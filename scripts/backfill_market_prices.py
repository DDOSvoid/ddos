#!/usr/bin/env python
"""Backfill resumable train-only stock/index daily bars for event ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.database.engine import get_engine, init_db
from src.database.models import DailyPrice
from src.pipeline.fetcher import TushareClient
from src.prediction.data_download import load_train_universe_state
from src.prediction.splits import load_prediction_split_contract

CONTRACT = "causal-market-price-backfill-v2"


def _optional_float(value) -> float | None:
    return None if pd.isna(value) else float(value)


def _codes_sha256(codes: list[str]) -> str:
    return hashlib.sha256("\n".join(codes).encode()).hexdigest()


def _write_state(path: Path, state: dict[str, object]) -> None:
    state["updated_at"] = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_or_create_state(
    *,
    path: Path,
    codes: list[str],
    start: date,
    end: date,
    benchmark_code: str,
) -> dict[str, object]:
    selection = {
        "dataset_role": "train",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "codes": len(codes),
        "codes_sha256": _codes_sha256(codes),
        "benchmark_code": benchmark_code,
    }
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("contract") != CONTRACT:
            raise ValueError("market price state contract mismatch")
        if state.get("selection") != selection:
            raise ValueError("market price state selection mismatch")
        if state.get("official_test_queried") is not False:
            raise PermissionError("market price state violates test isolation")
        return state
    return {
        "contract": CONTRACT,
        "created_at": datetime.now(UTC).isoformat(),
        "official_test_queried": False,
        "selection": selection,
        "completed": {},
        "failures": {},
        "benchmark": None,
        "stock_rows": 0,
    }


def upsert_bars(
    session: Session,
    frame: pd.DataFrame,
    *,
    instrument_type: str,
    fetched_at: datetime,
) -> int:
    if frame.empty:
        return 0
    rows = []
    for item in frame.to_dict("records"):
        rows.append(
            {
                "instrument_code": str(item["ts_code"]),
                "instrument_type": instrument_type,
                "trade_date": date.fromisoformat(
                    f"{item['trade_date'][:4]}-{item['trade_date'][4:6]}-{item['trade_date'][6:8]}"
                ),
                "open_price": float(item["open"]),
                "high_price": float(item["high"]),
                "low_price": float(item["low"]),
                "close_price": float(item["close"]),
                "pre_close": _optional_float(item.get("pre_close")),
                "pct_change": _optional_float(item.get("pct_chg")),
                "volume": _optional_float(item.get("vol")),
                "amount": _optional_float(item.get("amount")),
                "source": "tushare",
                "fetched_at": fetched_at,
            }
        )
    statement = sqlite_insert(DailyPrice).values(rows)
    statement = statement.on_conflict_do_update(
        index_elements=["instrument_code", "trade_date"],
        set_={
            "instrument_type": statement.excluded.instrument_type,
            "open_price": statement.excluded.open_price,
            "high_price": statement.excluded.high_price,
            "low_price": statement.excluded.low_price,
            "close_price": statement.excluded.close_price,
            "pre_close": statement.excluded.pre_close,
            "pct_change": statement.excluded.pct_change,
            "volume": statement.excluded.volume,
            "amount": statement.excluded.amount,
            "source": statement.excluded.source,
            "fetched_at": statement.excluded.fetched_at,
        },
    )
    session.execute(statement)
    return len(rows)


def backfill(
    *,
    start_date: date,
    end_date: date,
    benchmark_code: str,
    universe_state: Path,
    state_path: Path,
    limit: int,
) -> dict[str, object]:
    split = load_prediction_split_contract()
    if end_date > split.train.end:
        raise PermissionError("market price download may not cross the train cutoff")
    if start_date >= split.train.start:
        raise ValueError("market price download must include causal prehistory")

    codes = load_train_universe_state(universe_state)
    if limit > 0:
        codes = codes[:limit]
    state = _load_or_create_state(
        path=state_path,
        codes=codes,
        start=start_date,
        end=end_date,
        benchmark_code=benchmark_code,
    )
    completed = state.setdefault("completed", {})
    failures = state.setdefault("failures", {})

    init_db()
    engine = get_engine()
    client = TushareClient()
    start_text = start_date.strftime("%Y%m%d")
    end_text = end_date.strftime("%Y%m%d")

    if not state.get("benchmark"):
        benchmark = client.get_index_daily(benchmark_code, start_text, end_text)
        if benchmark.empty:
            raise ValueError(f"benchmark prices missing: {benchmark_code}")
        with Session(engine) as session:
            benchmark_rows = upsert_bars(
                session,
                benchmark,
                instrument_type="index",
                fetched_at=datetime.now(UTC),
            )
            session.commit()
        state["benchmark"] = {
            "code": benchmark_code,
            "rows": benchmark_rows,
            "empty": benchmark.empty,
        }
        _write_state(state_path, state)

    for index, code in enumerate(codes, 1):
        if code in completed:
            continue
        try:
            frame = client.get_daily(
                ts_code=code,
                start_date=start_text,
                end_date=end_text,
            )
            with Session(engine) as session:
                rows = upsert_bars(
                    session,
                    frame,
                    instrument_type="stock",
                    fetched_at=datetime.now(UTC),
                )
                session.commit()
            completed[code] = {"rows": rows, "empty": frame.empty}
            state["stock_rows"] = int(state.get("stock_rows") or 0) + rows
            failures.pop(code, None)
        except Exception as error:  # keep the queue resumable across provider errors
            failures[code] = {
                "error_type": type(error).__name__,
                "message": str(error)[:500],
                "failed_at": datetime.now(UTC).isoformat(),
            }
        _write_state(state_path, state)
        done = len(completed)
        if done % 20 == 0 or index == len(codes):
            print(
                f"stock progress {done}/{len(codes)}; "
                f"rows={state['stock_rows']}; failures={len(failures)}",
                flush=True,
            )

    if len(completed) == len(codes) and not failures:
        state["finished_at"] = datetime.now(UTC).isoformat()
    else:
        state.pop("finished_at", None)
    _write_state(state_path, state)
    result = {
        "contract": CONTRACT,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "stock_codes": len(codes),
        "completed_codes": len(completed),
        "stock_rows": int(state.get("stock_rows") or 0),
        "failures": len(failures),
        "benchmark": state.get("benchmark"),
        "official_test_queried": False,
        "finished": bool(state.get("finished_at")),
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


def main() -> None:
    root = PROJECT_ROOT / "data" / "backfills" / "event_ranking_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2022, 9, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--universe-state", type=Path, default=root / "train_universe.json")
    parser.add_argument("--state", type=Path, default=root / "market_prices.state.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    backfill(
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark_code=args.benchmark,
        universe_state=args.universe_state,
        state_path=args.state,
        limit=max(args.limit, 0),
    )


if __name__ == "__main__":
    main()

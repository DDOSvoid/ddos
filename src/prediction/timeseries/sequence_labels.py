"""Recompute physically separate train-only LSTM labels from cached daily bars."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from datetime import UTC, date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.prediction.development_contract import load_causal_development_contract
from src.prediction.splits import load_prediction_split_contract
from src.prediction.timeseries.contract import load_timeseries_model_contract
from src.prediction.timeseries.coverage import _read_only_connection
from src.prediction.timeseries.lstm_contract import (
    LstmTimeseriesContract,
    load_lstm_timeseries_contract,
)
from src.prediction.timeseries.sequence_artifacts import (
    _file_sha256,
    _stable_rows_sha256,
    load_label_free_train_sources,
)
from src.prediction.timeseries.sequence_identity import canonical_sample_id


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _available_at(exit_date: date, contract: LstmTimeseriesContract) -> datetime:
    from zoneinfo import ZoneInfo

    return datetime.combine(
        exit_date,
        time(15, 0),
        tzinfo=ZoneInfo("Asia/Shanghai"),
    ).astimezone(UTC)


def _prepare_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["trade_date"] = pd.to_datetime(
        working["trade_date"], errors="raise"
    ).dt.date
    working = working.sort_values("trade_date", kind="mergesort").reset_index(
        drop=True
    )
    if working["trade_date"].duplicated().any():
        raise ValueError("duplicate daily price date in label source")
    return working


def build_train_sequence_labels(
    company_days: pd.DataFrame,
    *,
    stock_bars_by_code: dict[str, pd.DataFrame],
    benchmark_bars: pd.DataFrame,
    contract: LstmTimeseriesContract,
) -> pd.DataFrame:
    """Build only labels whose outcomes mature inside the train partition."""
    benchmark = _prepare_price_frame(benchmark_bars)
    benchmark_by_date = {
        row.trade_date: row for row in benchmark.itertuples(index=False)
    }
    prepared_stock = {
        code: _prepare_price_frame(frame)
        for code, frame in stock_bars_by_code.items()
    }
    records: list[dict[str, object]] = []
    for identity in company_days.sort_values(
        ["published_date", "stock_code"], kind="mergesort"
    ).itertuples(index=False):
        stock = prepared_stock.get(str(identity.stock_code))
        if stock is None or stock.empty:
            continue
        stock_dates = stock["trade_date"].tolist()
        start = bisect.bisect_right(stock_dates, identity.published_date)
        future = stock.iloc[start:]
        for horizon in contract.horizons_sessions:
            if len(future) < horizon:
                continue
            path = future.iloc[:horizon]
            entry = path.iloc[0]
            exit_bar = path.iloc[-1]
            outcome_available_at = _available_at(exit_bar["trade_date"], contract)
            if exit_bar["trade_date"] > contract.training_end:
                continue
            benchmark_path = [
                benchmark_by_date.get(trade_day) for trade_day in path["trade_date"]
            ]
            if any(value is None for value in benchmark_path):
                continue
            benchmark_entry = benchmark_by_date[entry["trade_date"]]
            benchmark_exit = benchmark_by_date[exit_bar["trade_date"]]
            stock_return = float(exit_bar["close_price"] / entry["open_price"] - 1.0)
            benchmark_return = float(
                benchmark_exit.close_price / benchmark_entry.open_price - 1.0
            )
            excess_return = stock_return - benchmark_return
            actual_direction = 1 if excess_return > 0 else -1 if excess_return < 0 else 0
            stock_session_returns = path["close_price"].to_numpy(dtype=float) / np.where(
                np.arange(horizon) == 0,
                path["open_price"].to_numpy(dtype=float),
                path["pre_close"].to_numpy(dtype=float),
            ) - 1.0
            benchmark_session_returns = np.asarray(
                [value.close_price for value in benchmark_path], dtype=float
            ) / np.where(
                np.arange(horizon) == 0,
                np.asarray([value.open_price for value in benchmark_path], dtype=float),
                np.asarray([value.pre_close for value in benchmark_path], dtype=float),
            ) - 1.0
            future_realized_volatility = float(
                np.sqrt(
                    np.square(stock_session_returns - benchmark_session_returns).sum()
                )
            )
            evidence = {
                "company_day_id": identity.company_day_id,
                "stock_code": identity.stock_code,
                "published_date": identity.published_date.isoformat(),
                "horizon_sessions": horizon,
                "entry_date": entry["trade_date"].isoformat(),
                "exit_date": exit_bar["trade_date"].isoformat(),
                "stock_entry_open": float(entry["open_price"]),
                "stock_exit_close": float(exit_bar["close_price"]),
                "benchmark_entry_open": float(benchmark_entry.open_price),
                "benchmark_exit_close": float(benchmark_exit.close_price),
            }
            records.append(
                {
                    "sample_id": canonical_sample_id(
                        identity.company_day_id, horizon
                    ),
                    "company_day_id": identity.company_day_id,
                    "stock_code": identity.stock_code,
                    "published_date": identity.published_date,
                    "horizon_sessions": horizon,
                    "dataset_role": "train",
                    "entry_date": entry["trade_date"],
                    "exit_date": exit_bar["trade_date"],
                    "outcome_available_at": outcome_available_at,
                    "future_excess_return": excess_return,
                    "future_absolute_excess_return": abs(excess_return),
                    "future_downside_return": min(excess_return, 0.0),
                    "future_realized_volatility": future_realized_volatility,
                    "actual_direction": actual_direction,
                    "target_evidence_sha256": _stable_sha256(evidence),
                }
            )
    labels = pd.DataFrame.from_records(records)
    if labels.empty:
        raise ValueError("no train-only LSTM labels could be reconstructed")
    return labels.sort_values(
        ["horizon_sessions", "published_date", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_and_write_train_labels(
    database_path: Path,
    output_directory: Path,
    *,
    contract: LstmTimeseriesContract,
    force: bool = False,
) -> dict[str, object]:
    output = output_directory.resolve()
    if not output.is_relative_to(contract.data_directory):
        raise ValueError("LSTM labels must remain under the registered data boundary")
    label_path = output / "labels" / "train_labels.parquet"
    manifest_path = output / "manifests" / "train_labels_manifest.json"
    existing = [path for path in (label_path, manifest_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(f"LSTM label artifact already exists: {existing[0]}")
    with _read_only_connection(database_path) as connection:
        company_days, stock_frames, benchmark, source = load_label_free_train_sources(
            connection,
            contract=contract,
        )
    labels = build_train_sequence_labels(
        company_days,
        stock_bars_by_code=stock_frames,
        benchmark_bars=benchmark,
        contract=contract,
    )
    _write_parquet_atomic(labels, label_path)
    manifest: dict[str, object] = {
        "artifact": "causal-timeseries-lstm-daily-v1-train-labels",
        "created_at": datetime.now(UTC).isoformat(),
        "contract": contract.contract,
        "contract_sha256": contract.source_sha256,
        "split_contract": contract.split_contract,
        "split_contract_sha256": contract.split_contract_sha256,
        "dataset_role_read": "train",
        "official_test_read": False,
        "quarantine_read": False,
        "forward_validation_read": False,
        "target_source": "recomputed_from_train_partition_cached_daily_prices",
        "source_target_table_queried": False,
        "rows": len(labels),
        "rows_by_horizon": {
            str(key): int(value)
            for key, value in labels.groupby("horizon_sessions").size().items()
        },
        "identity_source_sha256": source["identity_source_sha256"],
        "price_source_sha256": source["price_source_sha256"],
        "labels_file": label_path.relative_to(output).as_posix(),
        "labels_sha256": _file_sha256(label_path),
        "label_rows_sha256": _stable_rows_sha256(
            [tuple(row) for row in labels.astype(str).itertuples(index=False, name=None)]
        ),
    }
    _write_json_atomic(manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "ddos.db",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "timeseries" / "lstm_daily_v1",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
    parent = load_timeseries_model_contract(split=split, development=development)
    contract = load_lstm_timeseries_contract(
        split=split,
        development=development,
        parent=parent,
    )
    manifest = build_and_write_train_labels(
        args.database,
        args.output_dir,
        contract=contract,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

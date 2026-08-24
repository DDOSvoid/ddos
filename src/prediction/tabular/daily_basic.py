"""Resumable train-only Tushare daily-basic valuation backfill."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.pipeline.fetcher import TushareClient
from src.prediction.splits import PredictionSplitContract
from src.prediction.tabular.contract import TabularModelContract

SHANGHAI = ZoneInfo("Asia/Shanghai")
DAILY_BASIC_CONTRACT = "tabular-daily-basic-backfill-v1"
DAILY_BASIC_FIELDS = (
    "ts_code",
    "trade_date",
    "close",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", index=False)
    temporary.replace(path)
    return _sha256_file(path)


def load_intraday_train_universe(state_path: Path) -> list[str]:
    """Reuse the audited train universe without opening the business database."""
    state = json.loads(state_path.read_text(encoding="utf-8"))
    selection = state.get("selection", {})
    if selection.get("dataset_role") != "train":
        raise PermissionError("stock universe is not train-only")
    if state.get("official_test_queried") is not False:
        raise PermissionError("stock universe state does not prove test isolation")
    codes = sorted({str(key).split(":", 1)[0] for key in state.get("completed", {})})
    canonical = "\n".join(codes)
    if len(codes) != selection.get("codes"):
        raise ValueError("stock universe count differs from source state")
    if hashlib.sha256(canonical.encode()).hexdigest() != selection.get("codes_sha256"):
        raise ValueError("stock universe hash differs from source state")
    return codes


def normalize_daily_basic_frame(
    frame: pd.DataFrame,
    *,
    stock_code: str,
    start: date,
    end: date,
    fetched_at: datetime,
) -> pd.DataFrame:
    missing = set(DAILY_BASIC_FIELDS) - set(frame.columns)
    if missing:
        raise ValueError(f"daily_basic response missing columns: {sorted(missing)}")
    normalized = frame.loc[:, DAILY_BASIC_FIELDS].copy()
    if normalized.empty:
        return normalized
    if set(normalized["ts_code"].astype(str).unique()) != {stock_code}:
        raise ValueError("daily_basic response contains an unexpected stock code")
    normalized["trade_date"] = pd.to_datetime(
        normalized["trade_date"], format="%Y%m%d", errors="raise"
    )
    normalized = normalized.sort_values("trade_date", kind="mergesort").reset_index(
        drop=True
    )
    if normalized["trade_date"].duplicated().any():
        raise ValueError("daily_basic response contains duplicate trade dates")
    trade_dates = normalized["trade_date"].dt.date
    if trade_dates.min() < start or trade_dates.max() > end:
        raise ValueError("daily_basic response leaves the train partition")
    for column in DAILY_BASIC_FIELDS[2:]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    for column in ("close", "total_share", "total_mv", "circ_mv"):
        if normalized[column].isna().any() or (normalized[column] <= 0).any():
            raise ValueError(f"daily_basic contains missing or non-positive {column}")
    for column in ("turnover_rate", "turnover_rate_f", "volume_ratio"):
        present = normalized[column].dropna()
        if (present < 0).any():
            raise ValueError(f"daily_basic contains negative {column}")
    normalized["trade_date"] = trade_dates
    normalized["available_at_utc"] = (
        pd.to_datetime(normalized["trade_date"].astype(str) + " 18:00:00")
        .dt.tz_localize(SHANGHAI)
        .dt.tz_convert("UTC")
    )
    normalized["source"] = "tushare_daily_basic"
    normalized["fetched_at_utc"] = pd.Timestamp(fetched_at).tz_convert("UTC")
    normalized["unit_market_value"] = "CNY_10K"
    normalized["unit_share"] = "SHARES_10K"
    return normalized


class TushareDailyBasicClient:
    def __init__(self, *, minimum_interval_seconds: float = 0.15) -> None:
        self.pro = TushareClient().pro
        self.minimum_interval_seconds = max(float(minimum_interval_seconds), 0.0)
        self._last_call: float | None = None

    def fetch(self, *, stock_code: str, start: date, end: date) -> pd.DataFrame:
        if self._last_call is not None:
            remaining = self.minimum_interval_seconds - (time.monotonic() - self._last_call)
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()
        value = self.pro.daily_basic(
            ts_code=stock_code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            fields=",".join(DAILY_BASIC_FIELDS),
        )
        return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def download_daily_basic_train(
    *,
    codes: list[str],
    start: date,
    end: date,
    output_root: Path,
    state_path: Path,
    client: TushareDailyBasicClient,
    split: PredictionSplitContract,
    tabular: TabularModelContract,
    maximum_attempts: int = 4,
) -> dict:
    if start != split.train.start or end != split.train.end:
        raise PermissionError("daily_basic download must equal the complete train partition")
    selected_codes = sorted(set(codes))
    codes_sha256 = hashlib.sha256("\n".join(selected_codes).encode()).hexdigest()
    selection = {
        "dataset_role": "train",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "source": "tushare_daily_basic",
        "codes": len(selected_codes),
        "codes_sha256": codes_sha256,
        "fields": list(DAILY_BASIC_FIELDS),
        "availability_rule": "trade_date 18:00 Asia/Shanghai",
    }
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    if state.get("selection") not in (None, selection):
        raise ValueError("daily_basic state selection does not match this run")
    completed = dict(state.get("completed", {}))
    failures = dict(state.get("failures", {}))
    state.setdefault("started_at_utc", datetime.now(UTC).isoformat())
    state.update(
        {
            "contract": DAILY_BASIC_CONTRACT,
            "tabular_contract_sha256": tabular.source_sha256,
            "split_contract": split.contract,
            "split_source_sha256": split.source_sha256,
            "official_test_queried": False,
            "selection": selection,
        }
    )
    for index, stock_code in enumerate(selected_codes, 1):
        key = f"{stock_code}:{start.isoformat()}:{end.isoformat()}"
        if key in completed:
            continue
        last_error: Exception | None = None
        for attempt in range(1, maximum_attempts + 1):
            try:
                fetched_at = datetime.now(UTC)
                raw = client.fetch(stock_code=stock_code, start=start, end=end)
                if raw.empty:
                    result = {"status": "empty", "rows": 0, "path": None, "sha256": None}
                else:
                    normalized = normalize_daily_basic_frame(
                        raw,
                        stock_code=stock_code,
                        start=start,
                        end=end,
                        fetched_at=fetched_at,
                    )
                    relative = Path("raw") / "tushare_daily_basic" / f"{stock_code}.parquet"
                    target = output_root / relative
                    digest = _write_parquet_atomic(normalized, target)
                    result = {
                        "status": "downloaded",
                        "rows": len(normalized),
                        "path": relative.as_posix(),
                        "sha256": digest,
                        "first_trade_date": str(normalized["trade_date"].min()),
                        "last_trade_date": str(normalized["trade_date"].max()),
                    }
                completed[key] = result
                failures.pop(key, None)
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - live API behavior
                last_error = exc
                if attempt < maximum_attempts:
                    time.sleep(min(2**attempt, 15))
        if last_error is not None:
            failures[key] = {
                "error_type": type(last_error).__name__,
                "error": str(last_error),
                "updated_at_utc": datetime.now(UTC).isoformat(),
            }
        state.update(
            {
                "completed": completed,
                "failures": failures,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        _save_json_atomic(state_path, state)
        print(
            f"daily_basic progress {index}/{len(selected_codes)}; "
            f"completed={len(completed)}; failures={len(failures)}",
            flush=True,
        )
    return state

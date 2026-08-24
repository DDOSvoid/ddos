"""Resumable train-only intraday market-data download primitives."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import sqlite3
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from src.pipeline.fetcher import TushareClient
from src.prediction.splits import PredictionSplitContract
from src.prediction.tabular.contract import TabularModelContract

SHANGHAI = ZoneInfo("Asia/Shanghai")
REQUIRED_SOURCE_COLUMNS = (
    "ts_code",
    "trade_time",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
)


@dataclass(frozen=True)
class IntradayChunk:
    stock_code: str
    start: date
    end: date

    @property
    def key(self) -> str:
        return f"{self.stock_code}:{self.start.isoformat()}:{self.end.isoformat()}"

    @property
    def year(self) -> int:
        if self.start.year != self.end.year:
            raise ValueError("intraday chunk crosses a calendar-year boundary")
        return self.start.year


class IntradayClient(Protocol):
    source: str
    bar_timestamp_semantics: str

    def fetch(self, chunk: IntradayChunk, *, frequency: str) -> pd.DataFrame: ...


class RateLimitedTushareIntradayClient:
    """Fallback client; the live entitlement may tighten to one call per hour."""

    source = "tushare_stk_mins"
    bar_timestamp_semantics = "bar_end"

    def __init__(
        self,
        *,
        minimum_interval_seconds: float = 3601.0,
        maximum_attempts: int = 4,
    ) -> None:
        if minimum_interval_seconds < 60:
            raise ValueError("stk_mins limiter must wait at least 60 seconds")
        self.client = TushareClient()
        self.minimum_interval_seconds = float(minimum_interval_seconds)
        self.maximum_attempts = int(maximum_attempts)
        self._last_attempt_monotonic: float | None = None

    def _wait_for_slot(self) -> None:
        if self._last_attempt_monotonic is not None:
            elapsed = time.monotonic() - self._last_attempt_monotonic
            remaining = self.minimum_interval_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_attempt_monotonic = time.monotonic()

    def fetch(self, chunk: IntradayChunk, *, frequency: str) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.maximum_attempts + 1):
            self._wait_for_slot()
            try:
                result = self.client.pro.stk_mins(
                    ts_code=chunk.stock_code,
                    freq=frequency,
                    start_date=f"{chunk.start.isoformat()} 00:00:00",
                    end_date=f"{chunk.end.isoformat()} 23:59:59",
                )
                return result if isinstance(result, pd.DataFrame) else pd.DataFrame()
            except Exception as exc:  # pragma: no cover - live API behavior
                last_error = exc
                if attempt >= self.maximum_attempts:
                    break
        assert last_error is not None
        raise last_error


class AmazingDataIntradayClient:
    """Local AmazingData package adapter with one authenticated session."""

    source = "amazingdata_query_kline"
    bar_timestamp_semantics = "bar_start"

    def __init__(
        self,
        *,
        package_path: Path,
        username: str,
        password: str,
        host: str,
        port: int,
        calendar_end: date,
    ) -> None:
        package_text = str(package_path.resolve())
        if package_text not in sys.path:
            sys.path.insert(0, package_text)
        self._amazingdata = importlib.import_module("AmazingData")
        base_module = importlib.import_module("AmazingData.query_api.base_data")
        market_module = importlib.import_module("AmazingData.query_api.market_data")
        constant_module = importlib.import_module("AmazingData.utils.constant")
        self._username = username
        self._period_min15 = constant_module.Period.min15.value

        # The vendor package prints an ephemeral session token during login.
        # Capture Python-level stdout/stderr so credentials never enter our logs.
        captured = io.StringIO()
        with redirect_stdout(captured), redirect_stderr(captured):
            logged_in = self._amazingdata.login(
                username,
                password,
                host,
                int(port),
            )
        if not logged_in:
            raise ConnectionError("AmazingData login failed")
        try:
            calendar = base_module.BaseData().get_calendar(
                data_type="str",
                market="SH",
                date=int(calendar_end.strftime("%Y%m%d")),
            )
            calendar_int = [
                int(str(item).replace("-", ""))
                for item in calendar
            ]
            if not calendar_int:
                raise ValueError("AmazingData returned an empty trading calendar")
            self._market = market_module.MarketData(calendar_int)
        except Exception:
            self.close()
            raise

    def fetch(self, chunk: IntradayChunk, *, frequency: str) -> pd.DataFrame:
        if frequency != "15min":
            raise ValueError("AmazingData adapter currently supports only 15min")
        result = self._market.query_kline(
            [chunk.stock_code],
            begin_date=int(chunk.start.strftime("%Y%m%d")),
            end_date=int(chunk.end.strftime("%Y%m%d")),
            period=self._period_min15,
        )
        frame = result.get(chunk.stock_code)
        if frame is None or frame.empty:
            return pd.DataFrame()
        return frame.rename(
            columns={
                "code": "ts_code",
                "kline_time": "trade_time",
                "volume": "vol",
            }
        )

    def close(self) -> None:
        if getattr(self, "_amazingdata", None) is None:
            return
        captured = io.StringIO()
        with redirect_stdout(captured), redirect_stderr(captured):
            self._amazingdata.logout(self._username)
        self._amazingdata = None


def yearly_chunks(
    stock_code: str,
    *,
    start: date,
    end: date,
) -> tuple[IntradayChunk, ...]:
    if start > end:
        raise ValueError("intraday start date is after end date")
    chunks = []
    for year in range(start.year, end.year + 1):
        chunk_start = max(start, date(year, 1, 1))
        chunk_end = min(end, date(year, 12, 31))
        chunks.append(IntradayChunk(stock_code, chunk_start, chunk_end))
    return tuple(chunks)


def require_train_only_range(
    *,
    start: date,
    end: date,
    split: PredictionSplitContract,
) -> None:
    if start < split.train.start or end > split.train.end or start > end:
        raise PermissionError(
            "intraday development download is restricted to the 2023-2024 train partition"
        )


def load_train_stock_codes(
    database_path: Path,
    *,
    start: date,
    end: date,
) -> list[str]:
    """Read the accepted core-event universe without opening a writable database."""
    uri = f"file:{database_path.resolve().as_posix()}?mode=ro"
    query = """
        SELECT DISTINCT c.stock_code
        FROM companies c
        JOIN announcements a ON a.company_id = c.id
        JOIN classifications cl ON cl.announcement_id = a.id
        WHERE a.published_date BETWEEN ? AND ?
          AND cl.needs_review = 0
          AND cl.relevance = 'core_event'
        ORDER BY c.stock_code
    """
    with sqlite3.connect(uri, uri=True, timeout=30) as connection:
        return [
            str(row[0])
            for row in connection.execute(query, (start.isoformat(), end.isoformat()))
        ]


def normalize_intraday_frame(
    frame: pd.DataFrame,
    *,
    chunk: IntradayChunk,
    frequency: str,
    fetched_at: datetime,
    source: str = "tushare_stk_mins",
    bar_timestamp_semantics: str = "bar_end",
) -> pd.DataFrame:
    missing = set(REQUIRED_SOURCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"intraday response missing columns: {sorted(missing)}")
    normalized = frame.loc[:, REQUIRED_SOURCE_COLUMNS].copy()
    if normalized.empty:
        return normalized
    if set(normalized["ts_code"].astype(str).unique()) != {chunk.stock_code}:
        raise ValueError("intraday response contains an unexpected stock code")

    normalized["trade_time"] = pd.to_datetime(
        normalized["trade_time"], errors="raise"
    )
    if normalized["trade_time"].dt.tz is not None:
        raise ValueError("Tushare trade_time unexpectedly contains a timezone")
    normalized = normalized.sort_values("trade_time", kind="mergesort").reset_index(
        drop=True
    )
    if normalized["trade_time"].duplicated().any():
        raise ValueError("intraday response contains duplicate trade_time values")
    trade_dates = normalized["trade_time"].dt.date
    if trade_dates.min() < chunk.start or trade_dates.max() > chunk.end:
        raise ValueError("intraday response leaves its requested train-only chunk")

    numeric = ("open", "high", "low", "close", "vol", "amount")
    for column in numeric:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        if normalized[column].isna().any():
            raise ValueError(f"intraday response contains missing {column}")
    if (normalized[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("intraday response contains non-positive prices")
    if (normalized[["vol", "amount"]] < 0).any().any():
        raise ValueError("intraday response contains negative volume or amount")
    if (
        normalized["high"]
        < normalized[["open", "low", "close"]].max(axis=1)
    ).any():
        raise ValueError("intraday high violates OHLC ordering")
    if (
        normalized["low"]
        > normalized[["open", "high", "close"]].min(axis=1)
    ).any():
        raise ValueError("intraday low violates OHLC ordering")

    local_times = normalized["trade_time"].dt.tz_localize(SHANGHAI)
    if bar_timestamp_semantics == "bar_start":
        availability = local_times + pd.Timedelta(frequency)
    elif bar_timestamp_semantics == "bar_end":
        availability = local_times
    else:
        raise ValueError(
            f"unsupported bar timestamp semantics: {bar_timestamp_semantics}"
        )
    normalized["trade_date"] = trade_dates
    normalized["available_at_utc"] = availability.dt.tz_convert("UTC")
    normalized["source"] = source
    normalized["frequency"] = frequency
    normalized["bar_timestamp_semantics"] = bar_timestamp_semantics
    normalized["price_adjustment"] = "raw"
    normalized["fetched_at_utc"] = pd.Timestamp(fetched_at).tz_convert("UTC")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", index=False)
    temporary.replace(path)
    return _sha256_file(path)


def download_intraday_train(
    *,
    codes: list[str],
    start: date,
    end: date,
    output_root: Path,
    state_path: Path,
    client: IntradayClient,
    split: PredictionSplitContract,
    tabular: TabularModelContract,
    limit_codes: int = 0,
) -> dict:
    require_train_only_range(start=start, end=end, split=split)
    if tabular.raw_intraday_frequency != "15min":
        raise ValueError("download frequency and tabular contract do not match")
    selected_codes = sorted(set(codes))
    if limit_codes > 0:
        selected_codes = selected_codes[:limit_codes]
    canonical_codes = "\n".join(selected_codes)
    selection = {
        "dataset_role": "train",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "frequency": tabular.raw_intraday_frequency,
        "usage": tabular.raw_intraday_usage,
        "source": client.source,
        "bar_timestamp_semantics": client.bar_timestamp_semantics,
        "codes": len(selected_codes),
        "codes_sha256": hashlib.sha256(canonical_codes.encode()).hexdigest(),
    }
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    if state.get("selection") not in (None, selection):
        raise ValueError("intraday state selection does not match this run")
    completed = dict(state.get("completed", {}))
    failures = dict(state.get("failures", {}))
    state.setdefault("started_at", datetime.now(UTC).isoformat())
    state.update(
        {
            "contract": "tabular-intraday-backfill-v1",
            "tabular_contract_sha256": tabular.source_sha256,
            "split_contract": split.contract,
            "split_source_sha256": split.source_sha256,
            "official_test_queried": False,
            "selection": selection,
        }
    )

    plan = [
        chunk
        for code in selected_codes
        for chunk in yearly_chunks(code, start=start, end=end)
    ]
    for index, chunk in enumerate(plan, 1):
        if chunk.key in completed:
            continue
        try:
            fetched_at = datetime.now(UTC)
            raw = client.fetch(chunk, frequency=tabular.raw_intraday_frequency)
            if raw.empty:
                result = {
                    "status": "empty",
                    "rows": 0,
                    "path": None,
                    "sha256": None,
                }
            else:
                normalized = normalize_intraday_frame(
                    raw,
                    chunk=chunk,
                    frequency=tabular.raw_intraday_frequency,
                    fetched_at=fetched_at,
                    source=client.source,
                    bar_timestamp_semantics=client.bar_timestamp_semantics,
                )
                relative = (
                    Path("raw")
                    / client.source
                    / chunk.stock_code
                    / f"{chunk.year}.parquet"
                )
                target = output_root / relative
                digest = _write_parquet_atomic(normalized, target)
                result = {
                    "status": "downloaded",
                    "rows": len(normalized),
                    "path": relative.as_posix(),
                    "sha256": digest,
                    "first_trade_time": str(normalized["trade_time"].min()),
                    "last_trade_time": str(normalized["trade_time"].max()),
                }
            completed[chunk.key] = result
            failures.pop(chunk.key, None)
        except Exception as exc:
            failures[chunk.key] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "updated_at": datetime.now(UTC).isoformat(),
            }
        state.update(
            {
                "completed": completed,
                "failures": failures,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        _save_json_atomic(state_path, state)
        print(
            f"intraday progress {index}/{len(plan)}; "
            f"completed={len(completed)}; failures={len(failures)}",
            flush=True,
        )
    return state

"""Causal daily features aggregated from complete 15-minute trading days."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.splits import PredictionSplitContract
from src.prediction.tabular.contract import TabularModelContract
from src.prediction.tabular.features import FeatureSpec

AGGREGATION_VERSION = "intraday-daily-aggregates-v1"
EXPECTED_BAR_LABELS = (
    "09:30",
    "09:45",
    "10:00",
    "10:15",
    "10:30",
    "10:45",
    "11:00",
    "11:15",
    "13:00",
    "13:15",
    "13:30",
    "13:45",
    "14:00",
    "14:15",
    "14:30",
    "14:45",
)
INTRADAY_FEATURE_COLUMNS = (
    "intraday_open_to_close_return",
    "intraday_realized_volatility",
    "intraday_downside_volatility",
    "intraday_high_low_range",
    "intraday_close_location",
    "intraday_opening_30m_return",
    "intraday_last_30m_return",
    "intraday_last_60m_return",
    "intraday_opening_30m_amount_share",
    "intraday_last_30m_amount_share",
    "intraday_last_30m_volume_share",
    "intraday_amount_hhi",
    "intraday_volume_hhi",
    "intraday_max_abs_15m_return",
    "intraday_amihud_15m",
)
REQUIRED_INTRADAY_COLUMNS = frozenset(
    {
        "ts_code",
        "trade_time",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "available_at_utc",
        "source",
        "frequency",
        "bar_timestamp_semantics",
        "price_adjustment",
    }
)


def intraday_feature_schema_sha256() -> str:
    payload = "\n".join((AGGREGATION_VERSION, *INTRADAY_FEATURE_COLUMNS))
    return hashlib.sha256(payload.encode()).hexdigest()


def intraday_feature_specs() -> tuple[FeatureSpec, ...]:
    return tuple(
        FeatureSpec(
            name=name,
            source="market_intraday_aggregate",
            available_at_column="intraday_features_available_at_utc",
            observation_date_column="observation_date",
        )
        for name in INTRADAY_FEATURE_COLUMNS
    )


def _safe_share(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0:
        return np.zeros_like(values, dtype=float)
    return values.astype(float) / total


def aggregate_intraday_days(
    frame: pd.DataFrame,
    *,
    source_file_sha256: str,
    tabular: TabularModelContract,
    split: PredictionSplitContract,
) -> pd.DataFrame:
    """Aggregate only complete A-share trading days with proven availability."""
    missing = REQUIRED_INTRADAY_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"normalized intraday data missing columns: {sorted(missing)}")
    if frame.empty:
        return pd.DataFrame()
    if len(source_file_sha256) != 64:
        raise ValueError("source intraday file requires a SHA-256")
    if set(frame["frequency"].astype(str).unique()) != {"15min"}:
        raise ValueError("aggregate source must contain only 15min bars")
    if set(frame["bar_timestamp_semantics"].astype(str).unique()) != {"bar_start"}:
        raise ValueError("aggregate source must use bar-start timestamp semantics")
    if set(frame["price_adjustment"].astype(str).unique()) != {"raw"}:
        raise ValueError("intraday source price-adjustment metadata changed")
    if set(frame["source"].astype(str).unique()) != {"amazingdata_query_kline"}:
        raise ValueError("unexpected intraday source")

    working = frame.copy()
    working["trade_time"] = pd.to_datetime(working["trade_time"], errors="raise")
    working["trade_date"] = pd.to_datetime(
        working["trade_date"], errors="raise"
    ).dt.date
    working["available_at_utc"] = pd.to_datetime(
        working["available_at_utc"], errors="raise", utc=True
    )
    stock_codes = set(working["ts_code"].astype(str).unique())
    if len(stock_codes) != 1:
        raise ValueError("one aggregate source file must contain exactly one stock")
    stock_code = next(iter(stock_codes))

    records = []
    for trade_date, day in working.groupby("trade_date", sort=True):
        day = day.sort_values("trade_time", kind="mergesort").reset_index(drop=True)
        labels = tuple(day["trade_time"].dt.strftime("%H:%M"))
        if labels != EXPECTED_BAR_LABELS:
            raise ValueError(
                f"incomplete or unexpected 15min trading day: {stock_code} {trade_date}"
            )
        expected_availability = (
            day["trade_time"].dt.tz_localize("Asia/Shanghai")
            + pd.Timedelta(minutes=15)
        ).dt.tz_convert("UTC")
        if not (day["available_at_utc"] == expected_availability).all():
            raise ValueError(
                f"bar availability semantics changed: {stock_code} {trade_date}"
            )

        opens = day["open"].to_numpy(dtype=float)
        highs = day["high"].to_numpy(dtype=float)
        lows = day["low"].to_numpy(dtype=float)
        closes = day["close"].to_numpy(dtype=float)
        volumes = day["vol"].to_numpy(dtype=float)
        amounts = day["amount"].to_numpy(dtype=float)
        return_bases = np.concatenate(([opens[0]], closes[:-1]))
        log_returns = np.log(closes / return_bases)
        amount_shares = _safe_share(amounts)
        volume_shares = _safe_share(volumes)
        day_high = float(highs.max())
        day_low = float(lows.min())
        price_range = day_high - day_low

        records.append(
            {
                "stock_code": stock_code,
                "observation_date": trade_date,
                "intraday_features_available_at_utc": day[
                    "available_at_utc"
                ].max(),
                "bar_count": len(day),
                "intraday_open_to_close_return": closes[-1] / opens[0] - 1.0,
                "intraday_realized_volatility": float(
                    np.sqrt(np.square(log_returns).sum())
                ),
                "intraday_downside_volatility": float(
                    np.sqrt(np.square(np.minimum(log_returns, 0.0)).sum())
                ),
                "intraday_high_low_range": day_high / day_low - 1.0,
                "intraday_close_location": (
                    0.5 if price_range == 0 else (closes[-1] - day_low) / price_range
                ),
                "intraday_opening_30m_return": closes[1] / opens[0] - 1.0,
                "intraday_last_30m_return": closes[-1] / opens[-2] - 1.0,
                "intraday_last_60m_return": closes[-1] / opens[-4] - 1.0,
                "intraday_opening_30m_amount_share": float(amount_shares[:2].sum()),
                "intraday_last_30m_amount_share": float(amount_shares[-2:].sum()),
                "intraday_last_30m_volume_share": float(volume_shares[-2:].sum()),
                "intraday_amount_hhi": float(np.square(amount_shares).sum()),
                "intraday_volume_hhi": float(np.square(volume_shares).sum()),
                "intraday_max_abs_15m_return": float(
                    np.abs(log_returns).max()
                ),
                "intraday_amihud_15m": float(
                    (np.abs(log_returns) / np.maximum(amounts, 1.0)).mean()
                    * 100_000_000.0
                ),
                "source": "amazingdata_query_kline",
                "frequency": "15min_to_daily",
                "price_adjustment": "raw_within_day_only",
                "aggregation_version": AGGREGATION_VERSION,
                "feature_schema_sha256": intraday_feature_schema_sha256(),
                "source_file_sha256": source_file_sha256,
                "tabular_contract_sha256": tabular.source_sha256,
                "split_contract": split.contract,
                "split_source_sha256": split.source_sha256,
            }
        )
    return pd.DataFrame.from_records(records)


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


def build_intraday_aggregate_dataset(
    *,
    raw_root: Path,
    raw_state_path: Path,
    output_root: Path,
    state_path: Path,
    tabular: TabularModelContract,
    split: PredictionSplitContract,
) -> dict:
    raw_state = json.loads(raw_state_path.read_text(encoding="utf-8"))
    selection = raw_state.get("selection", {})
    if raw_state.get("official_test_queried") is not False:
        raise PermissionError("intraday raw state does not prove official-test isolation")
    if selection.get("dataset_role") != "train":
        raise PermissionError("intraday aggregates accept train raw data only")
    if selection.get("start") != split.train.start.isoformat():
        raise PermissionError("intraday raw start does not match train partition")
    if selection.get("end") != split.train.end.isoformat():
        raise PermissionError("intraday raw end does not match train partition")
    if selection.get("source") != "amazingdata_query_kline":
        raise ValueError("aggregate source state is not AmazingData")
    if raw_state.get("tabular_contract_sha256") != tabular.source_sha256:
        raise ValueError("raw data and current tabular contracts do not match")

    aggregate_selection = {
        "dataset_role": "train",
        "raw_state_path": str(raw_state_path.resolve()),
        "raw_codes_sha256": selection.get("codes_sha256"),
        "aggregation_version": AGGREGATION_VERSION,
        "feature_schema_sha256": intraday_feature_schema_sha256(),
    }
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    if state.get("selection") not in (None, aggregate_selection):
        raise ValueError("intraday aggregate state selection does not match this run")
    completed = dict(state.get("completed", {}))
    failures = dict(state.get("failures", {}))
    state.setdefault("started_at", datetime.now(UTC).isoformat())
    state.update(
        {
            "contract": AGGREGATION_VERSION,
            "tabular_contract_sha256": tabular.source_sha256,
            "split_contract": split.contract,
            "split_source_sha256": split.source_sha256,
            "official_test_queried": False,
            "selection": aggregate_selection,
        }
    )

    raw_completed = raw_state.get("completed", {})
    for index, (key, item) in enumerate(sorted(raw_completed.items()), 1):
        if item.get("status") == "empty":
            completed[key] = {
                "status": "empty_source",
                "rows": 0,
                "path": None,
                "sha256": None,
                "source_file_sha256": None,
            }
            failures.pop(key, None)
        elif item.get("status") == "downloaded":
            source_sha256 = str(item["sha256"])
            previous = completed.get(key)
            previous_path = (
                output_root / previous["path"]
                if previous and previous.get("path")
                else None
            )
            if (
                previous
                and previous.get("source_file_sha256") == source_sha256
                and previous_path is not None
                and previous_path.exists()
                and _sha256_file(previous_path) == previous.get("sha256")
            ):
                continue
            try:
                source_path = raw_root / item["path"]
                if _sha256_file(source_path) != source_sha256:
                    raise ValueError(f"raw source hash mismatch: {source_path}")
                aggregate = aggregate_intraday_days(
                    pd.read_parquet(source_path),
                    source_file_sha256=source_sha256,
                    tabular=tabular,
                    split=split,
                )
                source_relative = Path(item["path"])
                relative = (
                    Path("aggregates")
                    / source_relative.parent.relative_to("raw")
                    / source_relative.name
                )
                target = output_root / relative
                digest = _write_parquet_atomic(aggregate, target)
                completed[key] = {
                    "status": "aggregated",
                    "rows": len(aggregate),
                    "path": relative.as_posix(),
                    "sha256": digest,
                    "source_file_sha256": source_sha256,
                    "first_observation_date": str(
                        aggregate["observation_date"].min()
                    ),
                    "last_observation_date": str(
                        aggregate["observation_date"].max()
                    ),
                }
                failures.pop(key, None)
            except Exception as exc:
                failures[key] = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "updated_at": datetime.now(UTC).isoformat(),
                }
        state.update(
            {
                "completed": completed,
                "failures": failures,
                "raw_completed_seen": len(raw_completed),
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        _save_json_atomic(state_path, state)
        if index % 20 == 0 or index == len(raw_completed):
            print(
                f"aggregate progress {index}/{len(raw_completed)}; "
                f"completed={len(completed)}; failures={len(failures)}",
                flush=True,
            )
    return state

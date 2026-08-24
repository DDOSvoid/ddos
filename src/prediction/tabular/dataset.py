"""Build physically separated point-in-time train features and outcome labels."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.prediction.development_contract import CausalDevelopmentContract
from src.prediction.splits import PredictionSplitContract
from src.prediction.tabular.aggregates import INTRADAY_FEATURE_COLUMNS
from src.prediction.tabular.contract import TabularModelContract
from src.prediction.tabular.features import (
    FeatureSpec,
    require_train_only,
    validate_and_select_features,
)
from src.prediction.tabular.snapshot import SNAPSHOT_CONTRACT

SHANGHAI = ZoneInfo("Asia/Shanghai")
DATASET_CONTRACT = "causal-tabular-train-dataset-v1"
BENCHMARK_CODE = "000300.SH"

EVENT_FEATURE_COLUMNS = (
    "horizon_sessions",
    "announcement_count",
    "unique_major_category_count",
    "unique_sub_category_count",
    "title_character_count",
    "title_numeric_token_count",
    "title_percent_token_count",
    "title_currency_token_count",
    "primary_major_category",
    "primary_sub_category",
    "event_count_a",
    "event_count_b",
    "event_count_c",
    "event_count_d",
    "event_count_e",
    "event_count_f",
    "event_count_g",
)

DAILY_MARKET_FEATURE_COLUMNS = (
    "market_daily_return_1d",
    "market_overnight_gap_1d",
    "market_high_low_range_1d",
    "market_momentum_3d",
    "market_momentum_5d",
    "market_momentum_10d",
    "market_momentum_20d",
    "market_momentum_60d",
    "market_realized_volatility_5d",
    "market_realized_volatility_10d",
    "market_realized_volatility_20d",
    "market_realized_volatility_60d",
    "market_downside_volatility_5d",
    "market_downside_volatility_10d",
    "market_downside_volatility_20d",
    "market_downside_volatility_60d",
    "market_max_drawdown_20d",
    "market_max_drawdown_60d",
    "market_amount_ratio_5d_20d",
    "market_volume_ratio_5d_20d",
    "market_average_range_5d",
    "market_average_range_20d",
    "benchmark_momentum_5d",
    "benchmark_momentum_20d",
    "benchmark_momentum_60d",
    "benchmark_volatility_5d",
    "benchmark_volatility_20d",
    "benchmark_volatility_60d",
    "excess_momentum_5d",
    "excess_momentum_20d",
    "excess_momentum_60d",
)

FEATURE_COLUMNS = (
    *EVENT_FEATURE_COLUMNS,
    *DAILY_MARKET_FEATURE_COLUMNS,
    *INTRADAY_FEATURE_COLUMNS,
)

CATEGORICAL_FEATURE_COLUMNS = (
    "primary_major_category",
    "primary_sub_category",
)

FEATURE_METADATA_COLUMNS = (
    "sample_id",
    "company_day_id",
    "stock_code",
    "published_date",
    "prediction_as_of",
    "dataset_role",
    "event_features_available_at_utc",
    "market_observation_date",
    "market_features_available_at_utc",
    "intraday_observation_date",
    "intraday_features_available_at_utc",
)

LABEL_COLUMNS = (
    "sample_id",
    "company_day_id",
    "stock_code",
    "published_date",
    "horizon_sessions",
    "dataset_role",
    "entry_date",
    "exit_date",
    "outcome_available_at",
    "future_excess_return",
    "future_absolute_excess_return",
    "future_downside_return",
    "future_realized_volatility",
    "actual_direction",
    "target_evidence_sha256",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha256(*parts: object) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode()).hexdigest()


def _local_timestamp(dates: pd.Series, local_time: str) -> pd.Series:
    return (
        pd.to_datetime(dates.astype(str) + " " + local_time, errors="raise")
        .dt.tz_localize(SHANGHAI)
        .dt.tz_convert("UTC")
    )


def tabular_feature_specs() -> tuple[FeatureSpec, ...]:
    specs = [
        FeatureSpec(
            name=name,
            source="prediction_context" if name == "horizon_sessions" else "announcement",
            available_at_column="event_features_available_at_utc",
        )
        for name in EVENT_FEATURE_COLUMNS
    ]
    specs.extend(
        FeatureSpec(
            name=name,
            source="market_daily",
            available_at_column="market_features_available_at_utc",
            observation_date_column="market_observation_date",
        )
        for name in DAILY_MARKET_FEATURE_COLUMNS
    )
    specs.extend(
        FeatureSpec(
            name=name,
            source="market_intraday_aggregate",
            available_at_column="intraday_features_available_at_utc",
            observation_date_column="intraday_observation_date",
        )
        for name in INTRADAY_FEATURE_COLUMNS
    )
    return tuple(specs)


def _primary_value(values: pd.Series) -> str:
    counts = values.astype(str).value_counts()
    maximum = int(counts.max())
    return sorted(counts[counts == maximum].index)[0]


def build_event_features(announcements: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only facts directly visible in accepted announcement titles."""
    working = announcements.copy()
    working["published_date"] = pd.to_datetime(
        working["published_date"], errors="raise"
    ).dt.date
    records = []
    for (stock_code, published_date), group in working.groupby(
        ["stock_code", "published_date"], sort=True
    ):
        titles = group["title"].astype(str).tolist()
        combined = "；".join(titles)
        categories = group["major_category"].astype(str).str.upper()
        record = {
            "stock_code": str(stock_code),
            "published_date": published_date,
            "announcement_count": len(group),
            "unique_major_category_count": categories.nunique(),
            "unique_sub_category_count": group["sub_category"].nunique(),
            "title_character_count": len(combined),
            "title_numeric_token_count": len(
                re.findall(r"\d+(?:\.\d+)?", combined)
            ),
            "title_percent_token_count": len(
                re.findall(r"[%％]|百分之", combined)
            ),
            "title_currency_token_count": len(
                re.findall(r"(?:亿|万)?元", combined)
            ),
            "primary_major_category": _primary_value(categories),
            "primary_sub_category": _primary_value(group["sub_category"]),
        }
        for category in "ABCDEFG":
            record[f"event_count_{category.lower()}"] = int(
                (categories == category).sum()
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_daily_market_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute causal rolling features on each instrument's ordered daily history."""
    working = prices.copy()
    working["trade_date"] = pd.to_datetime(working["trade_date"], errors="raise")
    working = working.sort_values(
        ["instrument_code", "trade_date"], kind="mergesort"
    ).reset_index(drop=True)
    numeric = (
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "pre_close",
        "pct_change",
        "volume",
        "amount",
    )
    for column in numeric:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["market_daily_return_1d"] = working["pct_change"] / 100.0
    missing_return = working["market_daily_return_1d"].isna()
    working.loc[missing_return, "market_daily_return_1d"] = (
        working.loc[missing_return, "close_price"]
        / working.loc[missing_return, "pre_close"]
        - 1.0
    )
    working["market_overnight_gap_1d"] = (
        working["open_price"] / working["pre_close"] - 1.0
    )
    working["market_high_low_range_1d"] = (
        working["high_price"] / working["low_price"] - 1.0
    )
    grouped = working.groupby("instrument_code", sort=False)
    for window in (3, 5, 10, 20, 60):
        working[f"market_momentum_{window}d"] = grouped[
            "close_price"
        ].transform(lambda values, w=window: values.pct_change(w, fill_method=None))
    for window in (5, 10, 20, 60):
        working[f"market_realized_volatility_{window}d"] = grouped[
            "market_daily_return_1d"
        ].transform(
            lambda values, w=window: values.rolling(w, min_periods=w).std(ddof=0)
        )
        working[f"market_downside_volatility_{window}d"] = grouped[
            "market_daily_return_1d"
        ].transform(
            lambda values, w=window: np.sqrt(
                values.clip(upper=0.0).pow(2).rolling(w, min_periods=w).mean()
            )
        )
    for window in (20, 60):
        rolling_high = grouped["close_price"].transform(
            lambda values, w=window: values.rolling(w, min_periods=w).max()
        )
        working[f"market_max_drawdown_{window}d"] = (
            working["close_price"] / rolling_high - 1.0
        )
    amount_5 = grouped["amount"].transform(
        lambda values: values.rolling(5, min_periods=5).mean()
    )
    amount_20 = grouped["amount"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    volume_5 = grouped["volume"].transform(
        lambda values: values.rolling(5, min_periods=5).mean()
    )
    volume_20 = grouped["volume"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    working["market_amount_ratio_5d_20d"] = amount_5 / amount_20
    working["market_volume_ratio_5d_20d"] = volume_5 / volume_20
    for window in (5, 20):
        working[f"market_average_range_{window}d"] = grouped[
            "market_high_low_range_1d"
        ].transform(
            lambda values, w=window: values.rolling(w, min_periods=w).mean()
        )

    benchmark = working.loc[
        working["instrument_code"] == BENCHMARK_CODE,
        [
            "trade_date",
            "market_momentum_5d",
            "market_momentum_20d",
            "market_momentum_60d",
            "market_realized_volatility_5d",
            "market_realized_volatility_20d",
            "market_realized_volatility_60d",
        ],
    ].rename(
        columns={
            "market_momentum_5d": "benchmark_momentum_5d",
            "market_momentum_20d": "benchmark_momentum_20d",
            "market_momentum_60d": "benchmark_momentum_60d",
            "market_realized_volatility_5d": "benchmark_volatility_5d",
            "market_realized_volatility_20d": "benchmark_volatility_20d",
            "market_realized_volatility_60d": "benchmark_volatility_60d",
        }
    )
    working = working.merge(benchmark, on="trade_date", how="left", validate="many_to_one")
    for window in (5, 20, 60):
        working[f"excess_momentum_{window}d"] = (
            working[f"market_momentum_{window}d"]
            - working[f"benchmark_momentum_{window}d"]
        )
    working["market_observation_date"] = working["trade_date"].dt.date
    working["market_features_available_at_utc"] = _local_timestamp(
        working["market_observation_date"], "15:00:00"
    )
    return working.loc[
        working["instrument_code"] != BENCHMARK_CODE,
        [
            "instrument_code",
            "trade_date",
            "market_observation_date",
            "market_features_available_at_utc",
            *DAILY_MARKET_FEATURE_COLUMNS,
        ],
    ].rename(columns={"instrument_code": "stock_code"})


def _merge_strictly_prior_market(
    samples: pd.DataFrame, market: pd.DataFrame
) -> pd.DataFrame:
    pieces = []
    for stock_code, sample_group in samples.groupby("stock_code", sort=False):
        left = sample_group.copy()
        left["published_timestamp"] = pd.to_datetime(
            left["published_timestamp"], errors="raise"
        ).astype("datetime64[ns]")
        left = left.sort_values("published_date", kind="mergesort")
        right = market.loc[market["stock_code"] == stock_code].copy()
        right["trade_date"] = pd.to_datetime(
            right["trade_date"], errors="raise"
        ).astype("datetime64[ns]")
        right = right.sort_values(
            "trade_date", kind="mergesort"
        )
        if right.empty:
            merged = left.copy()
            for column in market.columns:
                if column not in {"stock_code", "trade_date"}:
                    merged[column] = pd.NA
        else:
            merged = pd.merge_asof(
                left,
                right.drop(columns="stock_code"),
                left_on="published_timestamp",
                right_on="trade_date",
                direction="backward",
                allow_exact_matches=False,
            )
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True).sort_values(
        "__row_order", kind="mergesort"
    )


def _next_prediction_as_of(
    published_dates: pd.Series, benchmark_dates: pd.Series
) -> pd.Series:
    calendar = np.sort(pd.to_datetime(benchmark_dates).unique())
    published = pd.to_datetime(published_dates).to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(calendar, published, side="right")
    if (positions >= len(calendar)).any():
        raise ValueError("benchmark calendar does not reach every train prediction date")
    next_dates = pd.Series(calendar[positions])
    return _local_timestamp(next_dates.dt.date, "08:30:00")


def _load_intraday_aggregates(
    *, root: Path, state_path: Path, audit_path: Path, tabular: TabularModelContract
) -> tuple[pd.DataFrame, dict]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True or audit.get("official_test_queried") is not False:
        raise PermissionError("intraday audit is absent, failed, or not train-only")
    if audit.get("tabular_contract_sha256") != tabular.source_sha256:
        raise ValueError("intraday audit and tabular contract hashes differ")
    if state.get("official_test_queried") is not False or state.get("failures"):
        raise PermissionError("intraday aggregate state is not complete and train-only")
    frames = []
    resolved_root = root.resolve()
    for key, item in sorted(state.get("completed", {}).items()):
        if item.get("status") == "empty_source":
            continue
        if item.get("status") != "aggregated":
            raise ValueError(f"unexpected aggregate status: {key}")
        path = (resolved_root / item["path"]).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"aggregate path escapes root: {key}") from exc
        if _sha256_file(path) != item.get("sha256"):
            raise ValueError(f"aggregate hash mismatch: {key}")
        frames.append(pd.read_parquet(path))
    if not frames:
        raise ValueError("no intraday aggregates available")
    combined = pd.concat(frames, ignore_index=True)
    combined["observation_date"] = pd.to_datetime(
        combined["observation_date"], errors="raise"
    ).dt.date
    if combined.duplicated(["stock_code", "observation_date"]).any():
        raise ValueError("duplicate stock-date in intraday aggregate inputs")
    return combined, audit


def _read_train_snapshot(path: Path, manifest_path: Path, split: PredictionSplitContract):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != SNAPSHOT_CONTRACT:
        raise ValueError("unexpected train snapshot contract")
    if manifest.get("official_test_queried") is not False:
        raise PermissionError("train snapshot does not prove official-test isolation")
    if manifest.get("split_source_sha256") != split.source_sha256:
        raise ValueError("train snapshot split hash mismatch")
    if _sha256_file(path) != manifest.get("snapshot_sha256"):
        raise ValueError("train snapshot file hash mismatch")
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    expected = {
        "announcements_train",
        "targets_train",
        "daily_prices_train",
        "snapshot_metadata",
    }
    if tables != expected:
        connection.close()
        raise PermissionError("snapshot contains tables outside the train-only schema")
    announcements = pd.read_sql_query("SELECT * FROM announcements_train", connection)
    targets = pd.read_sql_query("SELECT * FROM targets_train", connection)
    prices = pd.read_sql_query("SELECT * FROM daily_prices_train", connection)
    connection.close()
    return announcements, targets, prices, manifest


def _build_labels(announcements: pd.DataFrame, targets: pd.DataFrame, prices: pd.DataFrame):
    joined = announcements.merge(
        targets, on="announcement_id", how="inner", validate="one_to_many"
    )
    joined["published_date"] = pd.to_datetime(
        joined["published_date"], errors="raise"
    ).dt.date
    joined["entry_date"] = pd.to_datetime(joined["entry_date"], errors="raise").dt.date
    joined["exit_date"] = pd.to_datetime(joined["exit_date"], errors="raise").dt.date
    joined["outcome_available_at"] = pd.to_datetime(
        joined["outcome_available_at"], errors="raise", utc=True
    )
    records = []
    group_columns = ["stock_code", "published_date", "horizon_sessions"]
    for key, group in joined.groupby(group_columns, sort=True):
        for column in (
            "entry_date",
            "exit_date",
            "stock_return",
            "benchmark_return",
            "excess_return",
            "actual_direction",
            "outcome_available_at",
        ):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"inconsistent company-day target {column}: {key}")
        stock_code, published_date, horizon = key
        company_day_id = _stable_sha256(
            "company_day", stock_code, published_date
        )
        sample_id = _stable_sha256("sample", company_day_id, int(horizon))
        evidence_hash = _stable_sha256(*sorted(group["prices_sha256"].astype(str)))
        excess_return = float(group["excess_return"].iloc[0])
        records.append(
            {
                "sample_id": sample_id,
                "company_day_id": company_day_id,
                "stock_code": stock_code,
                "published_date": published_date,
                "horizon_sessions": int(horizon),
                "dataset_role": "train",
                "entry_date": group["entry_date"].iloc[0],
                "exit_date": group["exit_date"].iloc[0],
                "outcome_available_at": group["outcome_available_at"].iloc[0],
                "future_excess_return": excess_return,
                "future_absolute_excess_return": abs(excess_return),
                "future_downside_return": min(excess_return, 0.0),
                "future_realized_volatility": np.nan,
                "actual_direction": int(group["actual_direction"].iloc[0]),
                "target_evidence_sha256": evidence_hash,
            }
        )
    labels = pd.DataFrame.from_records(records)

    price_working = prices.copy()
    price_working["trade_date"] = pd.to_datetime(
        price_working["trade_date"], errors="raise"
    ).dt.date
    price_groups = {
        code: group.set_index("trade_date").sort_index()
        for code, group in price_working.groupby("instrument_code", sort=False)
    }
    benchmark = price_groups[BENCHMARK_CODE]
    for index, row in labels.loc[labels["horizon_sessions"].isin([3, 5])].iterrows():
        stock = price_groups.get(row["stock_code"])
        if stock is None:
            raise ValueError(f"risk-label stock prices missing: {row['stock_code']}")
        path = stock.loc[row["entry_date"] : row["exit_date"]]
        if len(path) != row["horizon_sessions"]:
            raise ValueError(f"risk-label path length mismatch: {row['sample_id']}")
        benchmark_path = benchmark.reindex(path.index)
        if benchmark_path["close_price"].isna().any():
            raise ValueError(f"risk-label benchmark path missing: {row['sample_id']}")
        stock_returns = path["close_price"].to_numpy(dtype=float) / np.where(
            np.arange(len(path)) == 0,
            path["open_price"].to_numpy(dtype=float),
            path["pre_close"].to_numpy(dtype=float),
        ) - 1.0
        benchmark_returns = benchmark_path["close_price"].to_numpy(dtype=float) / np.where(
            np.arange(len(path)) == 0,
            benchmark_path["open_price"].to_numpy(dtype=float),
            benchmark_path["pre_close"].to_numpy(dtype=float),
        ) - 1.0
        labels.at[index, "future_realized_volatility"] = float(
            np.sqrt(np.square(stock_returns - benchmark_returns).sum())
        )
    return labels.loc[:, LABEL_COLUMNS]


def build_tabular_train_dataset(
    *,
    snapshot_path: Path,
    snapshot_manifest_path: Path,
    intraday_root: Path,
    intraday_state_path: Path,
    intraday_audit_path: Path,
    tabular: TabularModelContract,
    split: PredictionSplitContract,
    development: CausalDevelopmentContract,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    announcements, targets, prices, snapshot_manifest = _read_train_snapshot(
        snapshot_path, snapshot_manifest_path, split
    )
    labels = _build_labels(announcements, targets, prices)
    require_train_only(labels)
    event_features = build_event_features(announcements)
    market_features = build_daily_market_features(prices)
    intraday, intraday_audit = _load_intraday_aggregates(
        root=intraday_root,
        state_path=intraday_state_path,
        audit_path=intraday_audit_path,
        tabular=tabular,
    )

    features = labels.loc[
        :, [
            "sample_id",
            "company_day_id",
            "stock_code",
            "published_date",
            "horizon_sessions",
            "dataset_role",
        ]
    ].copy()
    features = features.merge(
        event_features,
        on=["stock_code", "published_date"],
        how="left",
        validate="many_to_one",
    )
    if features.loc[:, EVENT_FEATURE_COLUMNS[1:]].isna().any().any():
        raise ValueError("event feature aggregation left missing values")
    features["event_features_available_at_utc"] = _local_timestamp(
        features["published_date"], "23:59:59"
    )
    benchmark_dates = prices.loc[
        prices["instrument_code"] == BENCHMARK_CODE, "trade_date"
    ]
    features["prediction_as_of"] = _next_prediction_as_of(
        features["published_date"], benchmark_dates
    )
    features["published_timestamp"] = pd.to_datetime(features["published_date"])
    features["__row_order"] = np.arange(len(features))
    features = _merge_strictly_prior_market(features, market_features)

    intraday_join = intraday.loc[
        :, [
            "stock_code",
            "observation_date",
            "intraday_features_available_at_utc",
            *INTRADAY_FEATURE_COLUMNS,
        ]
    ].rename(columns={"observation_date": "intraday_observation_date"})
    features = features.merge(
        intraday_join,
        left_on=["stock_code", "market_observation_date"],
        right_on=["stock_code", "intraday_observation_date"],
        how="left",
        validate="many_to_one",
    )
    features = features.drop(columns=["published_timestamp", "trade_date", "__row_order"])
    features = features.loc[:, [*FEATURE_METADATA_COLUMNS, *FEATURE_COLUMNS]]
    require_train_only(features)
    selected = validate_and_select_features(
        features,
        specs=tabular_feature_specs(),
        tabular=tabular,
        development=development,
    )
    if selected.columns.tolist() != list(FEATURE_COLUMNS):
        raise ValueError("validated feature schema order changed")
    if set(features["sample_id"]) != set(labels["sample_id"]):
        raise ValueError("feature and label sample identities differ")
    if features["sample_id"].duplicated().any() or labels["sample_id"].duplicated().any():
        raise ValueError("duplicate sample identity")

    schema_hash = _stable_sha256(*FEATURE_COLUMNS)
    manifest = {
        "contract": DATASET_CONTRACT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_role": "train",
        "official_test_queried": False,
        "train_range": {
            "start": split.train.start.isoformat(),
            "end": split.train.end.isoformat(),
        },
        "rows": len(features),
        "rows_by_horizon": {
            str(int(horizon)): int(count)
            for horizon, count in labels["horizon_sessions"].value_counts().sort_index().items()
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_feature_columns": list(CATEGORICAL_FEATURE_COLUMNS),
        "feature_schema_sha256": schema_hash,
        "label_columns": list(LABEL_COLUMNS),
        "risk_label_formula": (
            "sqrt(sum((stock_session_return - benchmark_session_return)^2)); "
            "first session uses close/open, later sessions use close/pre_close"
        ),
        "contracts": {
            "tabular": tabular.source_sha256,
            "split": split.source_sha256,
            "development": development.source_sha256,
        },
        "sources": {
            "snapshot_sha256": snapshot_manifest["snapshot_sha256"],
            "intraday_audit_created_at_utc": intraday_audit["created_at_utc"],
            "intraday_feature_schema_sha256": intraday_audit[
                "feature_schema_sha256"
            ],
        },
        "missing_fraction": {
            name: float(features[name].isna().mean()) for name in FEATURE_COLUMNS
        },
    }
    return features, labels, manifest

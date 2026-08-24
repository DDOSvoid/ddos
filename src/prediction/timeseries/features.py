"""Causal daily sequence and market-state feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from src.prediction.development_contract import CausalDevelopmentContract
from src.prediction.timeseries.alignment import (
    daily_bar_available_at,
    daily_v1_prediction_as_of,
    require_daily_v1_bar_usable,
)
from src.prediction.timeseries.contract import TimeseriesModelContract

PRICE_COLUMNS = (
    "trade_date",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "pre_close",
    "volume",
    "amount",
)

SEQUENCE_FEATURE_COLUMNS = (
    "stock_return_1d",
    "benchmark_return_1d",
    "excess_return_1d",
    "stock_range_1d",
    "stock_gap_1d",
    "stock_log_volume",
    "stock_log_amount",
    "stock_tradable_mask",
    "stock_volume_mask",
    "stock_amount_mask",
)

SUMMARY_METADATA_COLUMNS = (
    "sample_id",
    "stock_code",
    "published_date",
    "prediction_as_of",
    "last_bar_trade_date",
    "last_bar_available_at",
)


@dataclass(frozen=True)
class DailySequence:
    sample_id: str
    stock_code: str
    benchmark_code: str
    published_date: date
    prediction_as_of: datetime
    frame: pd.DataFrame


def _prepare_bars(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    missing = [column for column in PRICE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} bars missing columns: {missing}")
    bars = frame.loc[:, PRICE_COLUMNS].copy()
    bars["trade_date"] = pd.to_datetime(bars["trade_date"], errors="raise").dt.date
    if bars["trade_date"].duplicated().any():
        raise ValueError(f"{label} bars contain duplicate trade dates")
    bars = bars.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    for column in ("open_price", "high_price", "low_price", "close_price"):
        bars[column] = pd.to_numeric(bars[column], errors="raise")
        if (bars[column].dropna() <= 0).any():
            raise ValueError(f"{label} bars contain non-positive {column}")
    for column in ("pre_close", "volume", "amount"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    if (bars[["volume", "amount"]].dropna(how="all") < 0).any().any():
        raise ValueError(f"{label} bars contain negative volume or amount")
    return bars


def _one_day_return(close: pd.Series, pre_close: pd.Series) -> pd.Series:
    previous = pre_close.where(pre_close > 0, close.shift(1))
    return close / previous - 1.0


def _safe_log1p(values: pd.Series) -> pd.Series:
    return np.log1p(values.where(values >= 0))


def build_daily_sequence(
    *,
    sample_id: str,
    stock_code: str,
    benchmark_code: str,
    published_date: date,
    stock_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    contract: TimeseriesModelContract,
    development: CausalDevelopmentContract,
) -> DailySequence:
    """Build one target-free sequence using only bars available at daily-v1 as-of."""
    stock = _prepare_bars(stock_bars, label="stock")
    benchmark = _prepare_bars(benchmark_bars, label="benchmark")
    prediction_as_of = daily_v1_prediction_as_of(
        published_date,
        development=development,
    )

    # Keep an extra session so return fallback can use a prior close, then crop.
    eligible_benchmark = benchmark.loc[
        benchmark["trade_date"] < published_date
    ].tail(contract.maximum_sequence_sessions + 1)
    if len(eligible_benchmark) < contract.minimum_history_sessions:
        raise ValueError(
            "insufficient benchmark history before prediction: "
            f"{len(eligible_benchmark)} < {contract.minimum_history_sessions}"
        )
    eligible_stock = stock.loc[stock["trade_date"] < published_date]
    merged = eligible_benchmark.merge(
        eligible_stock,
        on="trade_date",
        how="left",
        suffixes=("_benchmark", "_stock"),
        validate="one_to_one",
    )

    merged["stock_return_1d"] = _one_day_return(
        merged["close_price_stock"], merged["pre_close_stock"]
    )
    merged["benchmark_return_1d"] = _one_day_return(
        merged["close_price_benchmark"], merged["pre_close_benchmark"]
    )
    merged["excess_return_1d"] = (
        merged["stock_return_1d"] - merged["benchmark_return_1d"]
    )
    merged["stock_range_1d"] = (
        merged["high_price_stock"] - merged["low_price_stock"]
    ) / merged["pre_close_stock"].where(merged["pre_close_stock"] > 0)
    merged["stock_gap_1d"] = (
        merged["open_price_stock"] / merged["pre_close_stock"].where(
            merged["pre_close_stock"] > 0
        )
        - 1.0
    )
    merged["stock_log_volume"] = _safe_log1p(merged["volume_stock"])
    merged["stock_log_amount"] = _safe_log1p(merged["amount_stock"])
    merged["stock_tradable_mask"] = merged["close_price_stock"].notna().astype(float)
    merged["stock_volume_mask"] = merged["volume_stock"].notna().astype(float)
    merged["stock_amount_mask"] = merged["amount_stock"].notna().astype(float)

    sequence = merged.tail(contract.maximum_sequence_sessions).copy()
    if len(sequence) < contract.minimum_history_sessions:
        raise ValueError(
            "insufficient sequence history before prediction: "
            f"{len(sequence)} < {contract.minimum_history_sessions}"
        )
    valid_stock_sessions = int(sequence["stock_tradable_mask"].sum())
    if valid_stock_sessions < contract.minimum_valid_stock_sessions:
        raise ValueError(
            "insufficient valid stock sessions before prediction: "
            f"{valid_stock_sessions} < {contract.minimum_valid_stock_sessions}"
        )

    available_at = []
    for trade_day in sequence["trade_date"]:
        bar_time = daily_bar_available_at(
            trade_day,
            contract=contract,
            development=development,
        )
        require_daily_v1_bar_usable(
            trade_date=trade_day,
            published_date=published_date,
            prediction_as_of=prediction_as_of,
            bar_available_at=bar_time,
            development=development,
        )
        available_at.append(bar_time)
    sequence["bar_available_at"] = available_at
    sequence = sequence.loc[
        :, ["trade_date", "bar_available_at", *SEQUENCE_FEATURE_COLUMNS]
    ].reset_index(drop=True)

    return DailySequence(
        sample_id=sample_id,
        stock_code=stock_code,
        benchmark_code=benchmark_code,
        published_date=published_date,
        prediction_as_of=prediction_as_of,
        frame=sequence,
    )


def _compounded_return(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return float("nan")
    return float(np.prod(1.0 + clean.to_numpy(dtype=float)) - 1.0)


def summary_feature_names(contract: TimeseriesModelContract) -> tuple[str, ...]:
    names = []
    for lookback in contract.lookbacks_sessions:
        suffix = f"{lookback}d"
        names.extend(
            (
                f"stock_momentum_{suffix}",
                f"benchmark_momentum_{suffix}",
                f"excess_momentum_{suffix}",
                f"stock_volatility_{suffix}",
                f"benchmark_volatility_{suffix}",
                f"mean_range_{suffix}",
                f"mean_gap_{suffix}",
                f"mean_log_volume_{suffix}",
                f"mean_log_amount_{suffix}",
                f"tradable_fraction_{suffix}",
            )
        )
    return tuple(names)


def summarize_daily_sequence(
    sequence: DailySequence,
    *,
    contract: TimeseriesModelContract,
) -> dict[str, object]:
    """Create causal endpoint baselines without fitting global statistics."""
    frame = sequence.frame
    result: dict[str, object] = {
        "sample_id": sequence.sample_id,
        "stock_code": sequence.stock_code,
        "published_date": sequence.published_date,
        "prediction_as_of": sequence.prediction_as_of,
        "last_bar_trade_date": frame["trade_date"].iloc[-1],
        "last_bar_available_at": frame["bar_available_at"].iloc[-1],
    }
    for lookback in contract.lookbacks_sessions:
        tail = frame.tail(lookback)
        suffix = f"{lookback}d"
        result[f"stock_momentum_{suffix}"] = _compounded_return(
            tail["stock_return_1d"]
        )
        result[f"benchmark_momentum_{suffix}"] = _compounded_return(
            tail["benchmark_return_1d"]
        )
        result[f"excess_momentum_{suffix}"] = _compounded_return(
            tail["excess_return_1d"]
        )
        result[f"stock_volatility_{suffix}"] = float(
            tail["stock_return_1d"].std(ddof=1)
        )
        result[f"benchmark_volatility_{suffix}"] = float(
            tail["benchmark_return_1d"].std(ddof=1)
        )
        result[f"mean_range_{suffix}"] = float(tail["stock_range_1d"].mean())
        result[f"mean_gap_{suffix}"] = float(tail["stock_gap_1d"].mean())
        result[f"mean_log_volume_{suffix}"] = float(
            tail["stock_log_volume"].mean()
        )
        result[f"mean_log_amount_{suffix}"] = float(
            tail["stock_log_amount"].mean()
        )
        result[f"tradable_fraction_{suffix}"] = float(
            tail["stock_tradable_mask"].mean()
        )
    for name in result:
        if name in SUMMARY_METADATA_COLUMNS:
            continue
        if not contract.feature_name_allowed(name):
            raise ValueError(f"forbidden time-series feature generated: {name}")
    return result


def _rolling_compounded_return(values: pd.Series, window: int) -> pd.Series:
    return np.expm1(
        np.log1p(values).rolling(window=window, min_periods=1).sum()
    )


def build_daily_feature_panel(
    *,
    stock_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    contract: TimeseriesModelContract,
) -> pd.DataFrame:
    """Precompute backward-only endpoint features on the benchmark calendar."""
    stock = _prepare_bars(stock_bars, label="stock")
    benchmark = _prepare_bars(benchmark_bars, label="benchmark")
    merged = benchmark.merge(
        stock,
        on="trade_date",
        how="left",
        suffixes=("_benchmark", "_stock"),
        validate="one_to_one",
    )
    merged["stock_return_1d"] = _one_day_return(
        merged["close_price_stock"], merged["pre_close_stock"]
    )
    merged["benchmark_return_1d"] = _one_day_return(
        merged["close_price_benchmark"], merged["pre_close_benchmark"]
    )
    merged["excess_return_1d"] = (
        merged["stock_return_1d"] - merged["benchmark_return_1d"]
    )
    merged["stock_range_1d"] = (
        merged["high_price_stock"] - merged["low_price_stock"]
    ) / merged["pre_close_stock"].where(merged["pre_close_stock"] > 0)
    merged["stock_gap_1d"] = (
        merged["open_price_stock"] / merged["pre_close_stock"].where(
            merged["pre_close_stock"] > 0
        )
        - 1.0
    )
    merged["stock_log_volume"] = _safe_log1p(merged["volume_stock"])
    merged["stock_log_amount"] = _safe_log1p(merged["amount_stock"])
    merged["stock_tradable_mask"] = merged["close_price_stock"].notna().astype(float)
    merged["calendar_history_sessions"] = np.arange(1, len(merged) + 1)
    for lookback in contract.lookbacks_sessions:
        suffix = f"{lookback}d"
        merged[f"stock_momentum_{suffix}"] = _rolling_compounded_return(
            merged["stock_return_1d"], lookback
        )
        merged[f"benchmark_momentum_{suffix}"] = _rolling_compounded_return(
            merged["benchmark_return_1d"], lookback
        )
        merged[f"excess_momentum_{suffix}"] = _rolling_compounded_return(
            merged["excess_return_1d"], lookback
        )
        merged[f"stock_volatility_{suffix}"] = merged["stock_return_1d"].rolling(
            window=lookback, min_periods=2
        ).std(ddof=1)
        merged[f"benchmark_volatility_{suffix}"] = merged[
            "benchmark_return_1d"
        ].rolling(window=lookback, min_periods=2).std(ddof=1)
        merged[f"mean_range_{suffix}"] = merged["stock_range_1d"].rolling(
            window=lookback, min_periods=1
        ).mean()
        merged[f"mean_gap_{suffix}"] = merged["stock_gap_1d"].rolling(
            window=lookback, min_periods=1
        ).mean()
        merged[f"mean_log_volume_{suffix}"] = merged["stock_log_volume"].rolling(
            window=lookback, min_periods=1
        ).mean()
        merged[f"mean_log_amount_{suffix}"] = merged["stock_log_amount"].rolling(
            window=lookback, min_periods=1
        ).mean()
        merged[f"tradable_fraction_{suffix}"] = merged[
            "stock_tradable_mask"
        ].rolling(window=lookback, min_periods=1).mean()
    merged["valid_stock_sessions_max_window"] = merged[
        "stock_tradable_mask"
    ].rolling(
        window=contract.maximum_sequence_sessions,
        min_periods=1,
    ).sum()
    return merged.loc[
        :,
        [
            "trade_date",
            "calendar_history_sessions",
            "valid_stock_sessions_max_window",
            *summary_feature_names(contract),
        ],
    ].copy()

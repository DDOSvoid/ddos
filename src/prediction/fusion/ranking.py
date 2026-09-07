"""Risk-aware company-day ranking and cross-sectional research metrics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def compute_rank_score(
    expected_excess_return: Iterable[float] | pd.Series,
    risk_scale: Iterable[float] | pd.Series,
    *,
    risk_aversion: float,
    round_trip_cost_bps: int,
) -> np.ndarray:
    if risk_aversion < 0 or round_trip_cost_bps < 0:
        raise ValueError("risk_aversion and round_trip_cost_bps must be non-negative")
    expected = np.asarray(expected_excess_return, dtype=float)
    risk = np.asarray(risk_scale, dtype=float)
    if expected.shape != risk.shape:
        raise ValueError("expected return and risk must have identical shapes")
    if np.isnan(expected).any() or np.isnan(risk).any() or (risk < 0).any():
        raise ValueError("rank inputs must be finite and risk must be non-negative")
    return expected - risk_aversion * risk - round_trip_cost_bps / 10_000.0


def rank_company_day_predictions(
    frame: pd.DataFrame,
    *,
    risk_aversion: float,
    round_trip_cost_bps: int,
) -> pd.DataFrame:
    required = {
        "stock_code",
        "eligible_entry_date",
        "horizon_sessions",
        "expected_excess_return",
        "risk_scale",
        "fusion_available",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ranking frame missing columns: {sorted(missing)}")
    output = frame.copy()
    identity = ["stock_code", "eligible_entry_date", "horizon_sessions"]
    if output.duplicated(identity).any():
        raise ValueError("ranking requires one prediction per company-day-horizon")
    output["rank_score"] = np.nan
    available = output["fusion_available"].astype(bool)
    output.loc[available, "rank_score"] = compute_rank_score(
        output.loc[available, "expected_excess_return"],
        output.loc[available, "risk_scale"],
        risk_aversion=risk_aversion,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    grouping = ["eligible_entry_date", "horizon_sessions"]
    output["daily_rank"] = output.groupby(grouping)["rank_score"].rank(
        method="first", ascending=False, na_option="keep"
    )
    group_sizes = output.groupby(grouping)["rank_score"].transform("count")
    output["rank_percentile"] = np.where(
        available & (group_sizes > 1),
        1.0 - (output["daily_rank"] - 1.0) / (group_sizes - 1.0),
        np.where(available, 1.0, np.nan),
    )
    return output.sort_values(grouping + ["daily_rank", "stock_code"]).reset_index(drop=True)


def _daily_turnover(groups: list[set[str]]) -> float:
    if len(groups) < 2:
        return 0.0
    values = []
    for previous, current in zip(groups, groups[1:]):
        denominator = max(len(previous), len(current), 1)
        values.append(1.0 - len(previous & current) / denominator)
    return float(np.mean(values))


def evaluate_daily_ranking(
    frame: pd.DataFrame,
    *,
    top_k: Iterable[int] = (5, 10, 20),
    round_trip_cost_bps: int = 20,
) -> dict[str, object]:
    required = {
        "stock_code",
        "eligible_entry_date",
        "horizon_sessions",
        "rank_score",
        "future_excess_return",
        "fusion_available",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ranking evaluation missing columns: {sorted(missing)}")
    ks = tuple(sorted(set(int(value) for value in top_k)))
    if not ks or any(value <= 0 for value in ks):
        raise ValueError("top_k must contain positive integers")

    results = []
    for horizon, horizon_frame in frame.groupby("horizon_sessions", sort=True):
        eligible = horizon_frame.loc[horizon_frame["fusion_available"].astype(bool)].copy()
        eligible["future_excess_return"] = pd.to_numeric(
            eligible["future_excess_return"], errors="raise"
        )
        daily_groups = list(eligible.groupby("eligible_entry_date", sort=True))
        daily_ics = []
        for _, group in daily_groups:
            if len(group) >= 2 and group["rank_score"].nunique() >= 2:
                daily_ics.append(
                    group["rank_score"].rank().corr(
                        group["future_excess_return"].rank(), method="pearson"
                    )
                )
        horizon_result: dict[str, object] = {
            "horizon_sessions": int(horizon),
            "eligible_rows": int(len(eligible)),
            "eligible_days": int(len(daily_groups)),
            "coverage": float(len(eligible) / len(horizon_frame)) if len(horizon_frame) else 0.0,
            "daily_spearman_rank_ic": float(np.nanmean(daily_ics)) if daily_ics else None,
            "top_k": {},
        }
        for value in ks:
            top_returns = []
            selected_hits = []
            remainder_returns = []
            selections: list[set[str]] = []
            for _, group in daily_groups:
                ordered = group.sort_values("rank_score", ascending=False)
                selected = ordered.head(value)
                remainder = ordered.iloc[len(selected) :]
                top_returns.append(float(selected["future_excess_return"].mean()))
                selected_hits.extend((selected["future_excess_return"] > 0).tolist())
                if not remainder.empty:
                    remainder_returns.append(float(remainder["future_excess_return"].mean()))
                selections.append(set(selected["stock_code"].astype(str)))
            top_mean = float(np.mean(top_returns)) if top_returns else None
            remainder_mean = float(np.mean(remainder_returns)) if remainder_returns else None
            turnover = _daily_turnover(selections)
            horizon_result["top_k"][str(value)] = {
                "mean_excess_return": top_mean,
                "precision": float(np.mean(selected_hits)) if selected_hits else None,
                "spread_over_remainder": (
                    top_mean - remainder_mean
                    if top_mean is not None and remainder_mean is not None
                    else None
                ),
                "turnover": turnover,
                "cost_adjusted_mean_excess_return": (
                    top_mean - round_trip_cost_bps / 10_000.0
                    if top_mean is not None
                    else None
                ),
            }
        results.append(horizon_result)
    return {"contract": "daily-cross-sectional-ranking-evaluation-v1", "horizons": results}

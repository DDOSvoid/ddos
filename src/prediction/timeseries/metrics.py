"""Train-period time-series evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.prediction.memory import wilson_interval


def probability_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    bullish_threshold: float = 0.5,
    constant_probability: np.ndarray | None = None,
) -> dict[str, object]:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if target.ndim != 1 or probability.ndim != 1 or len(target) != len(probability):
        raise ValueError("target and probability must be equal-length one-dimensional arrays")
    if len(target) == 0:
        raise ValueError("metrics require at least one sample")
    if not np.isin(target, (0, 1)).all():
        raise ValueError("direction target must contain only 0/1")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")

    bullish = probability >= bullish_threshold
    bullish_count = int(bullish.sum())
    bullish_hits = int(target[bullish].sum()) if bullish_count else 0
    lower, upper = wilson_interval(bullish_hits, bullish_count)
    auc = (
        None
        if len(np.unique(target)) < 2
        else float(roc_auc_score(target, probability))
    )
    result: dict[str, object] = {
        "samples": int(len(target)),
        "brier_score": float(brier_score_loss(target, probability)),
        "roc_auc": auc,
        "bullish_threshold": float(bullish_threshold),
        "bullish_signals": bullish_count,
        "bullish_hits": bullish_hits,
        "bullish_hit_rate": bullish_hits / bullish_count if bullish_count else None,
        "bullish_hit_rate_wilson_95pct": [lower, upper],
    }
    if constant_probability is not None:
        constant = np.asarray(constant_probability, dtype=float)
        if constant.shape != probability.shape:
            raise ValueError("constant baseline probabilities must match predictions")
        baseline_brier = float(brier_score_loss(target, constant))
        result["constant_base_rate_brier_score"] = baseline_brier
        result["brier_improvement_over_constant"] = (
            baseline_brier - result["brier_score"]
        )
    return result


def cost_adjusted_bullish_returns(
    excess_return: np.ndarray,
    probability: np.ndarray,
    *,
    bullish_threshold: float,
    round_trip_cost_bps: float,
) -> np.ndarray:
    returns = np.asarray(excess_return, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if returns.shape != probability.shape:
        raise ValueError("excess returns and probabilities must have matching shapes")
    if round_trip_cost_bps < 0:
        raise ValueError("round-trip cost cannot be negative")
    selected = returns[probability >= bullish_threshold]
    return selected - round_trip_cost_bps / 10_000.0

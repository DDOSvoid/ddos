"""Causal accuracy memory and bullish ranking.

This module deliberately does not decide whether an announcement is bullish. It
measures how often a frozen predictor was right using outcomes that had already
matured at the next prediction's cutoff.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class MaturedObservation:
    prediction_id: str
    memory_key: str
    horizon_sessions: int
    predicted_direction: int
    excess_return: float
    outcome_available_at: datetime

    @property
    def actual_direction(self) -> int:
        if self.excess_return > 0:
            return 1
        if self.excess_return < 0:
            return -1
        return 0

    @property
    def hit(self) -> bool:
        return self.predicted_direction == self.actual_direction


@dataclass(frozen=True)
class AccuracyMemory:
    memory_key: str
    horizon_sessions: int
    as_of: datetime
    sample_count: int
    hit_count: int
    hit_rate: float
    wilson_lower: float
    wilson_upper: float
    mean_excess_return: float
    source_outcomes_sha256: str


@dataclass(frozen=True)
class BullishRank:
    eligible: bool
    score: float | None
    degree: str
    reason: str


def _require_aware(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total < 0 or correct < 0 or correct > total:
        raise ValueError("correct/total counts are invalid")
    if total == 0:
        return 0.0, 0.0
    proportion = correct / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def build_accuracy_memory(
    observations: list[MaturedObservation],
    *,
    memory_key: str,
    horizon_sessions: int,
    as_of: datetime,
) -> AccuracyMemory:
    """Build memory from outcomes available no later than ``as_of``.

    Future outcomes are excluded rather than silently included. Neutral/abstained
    predictions are not directional trials and therefore do not affect accuracy.
    """
    cutoff = _require_aware(as_of, "as_of")
    eligible = []
    for item in observations:
        available_at = _require_aware(item.outcome_available_at, "outcome_available_at")
        if (
            item.memory_key == memory_key
            and item.horizon_sessions == horizon_sessions
            and item.predicted_direction in {-1, 1}
            and available_at <= cutoff
        ):
            eligible.append(item)
    eligible.sort(key=lambda item: item.prediction_id)

    hits = sum(item.hit for item in eligible)
    total = len(eligible)
    lower, upper = wilson_interval(hits, total)
    mean_excess = (
        sum(item.excess_return for item in eligible) / total if total else 0.0
    )
    source_payload = [
        {
            "prediction_id": item.prediction_id,
            "predicted_direction": item.predicted_direction,
            "excess_return": item.excess_return,
            "outcome_available_at": item.outcome_available_at.astimezone(UTC).isoformat(),
        }
        for item in eligible
    ]
    source_hash = hashlib.sha256(
        json.dumps(
            source_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return AccuracyMemory(
        memory_key=memory_key,
        horizon_sessions=horizon_sessions,
        as_of=cutoff,
        sample_count=total,
        hit_count=hits,
        hit_rate=hits / total if total else 0.0,
        wilson_lower=lower,
        wilson_upper=upper,
        mean_excess_return=mean_excess,
        source_outcomes_sha256=source_hash,
    )


def rank_bullish_prediction(
    predicted_direction: int,
    memory: AccuracyMemory,
    *,
    minimum_samples: int = 20,
    minimum_lower_bound: float = 0.5,
) -> BullishRank:
    """Rank solely by the lower confidence bound of historical hit accuracy."""
    if predicted_direction != 1:
        return BullishRank(False, None, "非利多", "当前预测方向不是利多")
    if memory.sample_count < minimum_samples:
        return BullishRank(
            False,
            None,
            "样本不足",
            f"历史已到期样本 {memory.sample_count} < {minimum_samples}",
        )
    score = memory.wilson_lower
    if score < minimum_lower_bound:
        return BullishRank(False, score, "未验证", "历史准确率下界未超过随机基线")
    if score >= 0.75:
        degree = "很强"
    elif score >= 0.65:
        degree = "强"
    elif score >= 0.55:
        degree = "中等"
    else:
        degree = "弱"
    return BullishRank(True, score, degree, "完全按历史命中率95%下界排名")

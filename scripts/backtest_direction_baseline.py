#!/usr/bin/env python
"""Strict walk-forward backtest for the existing event-direction baseline."""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import event_registry
from src.database.engine import get_engine
from src.database.models import (
    Announcement,
    AnnouncementMarketTarget,
    Classification,
)
from src.prediction.memory import (
    AccuracyMemory,
    rank_bullish_prediction,
    wilson_interval,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _direction(sub_category: str) -> int:
    major = event_registry.get_major(sub_category)
    if not major:
        return 0
    baseline = event_registry.categories[major].subcategories[sub_category].direction_baseline
    return 1 if baseline > 0 else -1 if baseline < 0 else 0


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def run_backtest(*, minimum_samples: int, output: Path) -> dict:
    engine = get_engine()
    with Session(engine) as session:
        rows = (
            session.query(
                Announcement.announcement_id,
                Announcement.published_date,
                Classification.sub_category,
                AnnouncementMarketTarget.horizon_sessions,
                AnnouncementMarketTarget.excess_return,
                AnnouncementMarketTarget.actual_direction,
                AnnouncementMarketTarget.outcome_available_at,
            )
            .join(Classification, Classification.announcement_id == Announcement.id)
            .join(
                AnnouncementMarketTarget,
                AnnouncementMarketTarget.announcement_id == Announcement.id,
            )
            .order_by(
                AnnouncementMarketTarget.horizon_sessions,
                Announcement.published_date,
                Announcement.id,
            )
            .all()
        )

    by_horizon = {}
    for horizon in (1, 3, 5):
        horizon_rows = [row for row in rows if row.horizon_sessions == horizon]
        pending_outcomes = []
        memory_stats = defaultdict(lambda: {"samples": 0, "hits": 0, "excess": 0.0})
        directional = 0
        correct = 0
        bullish = 0
        bullish_correct = 0
        ranked = 0
        ranked_correct = 0
        ranked_categories = Counter()
        insufficient = 0
        unverified = 0

        for row in horizon_rows:
            prediction_as_of = datetime.combine(
                row.published_date,
                time(23, 59, 59),
                tzinfo=SHANGHAI,
            ).astimezone(UTC)
            predicted_direction = _direction(row.sub_category)
            memory_key = f"subcategory:{row.sub_category}"
            while pending_outcomes and pending_outcomes[0][0] <= prediction_as_of:
                _, _, matured_key, matured_hit, matured_excess = heapq.heappop(
                    pending_outcomes
                )
                matured = memory_stats[matured_key]
                matured["samples"] += 1
                matured["hits"] += int(matured_hit)
                matured["excess"] += matured_excess
            stats = memory_stats[memory_key]
            lower, upper = wilson_interval(stats["hits"], stats["samples"])
            memory = AccuracyMemory(
                memory_key=memory_key,
                horizon_sessions=horizon,
                as_of=prediction_as_of,
                sample_count=stats["samples"],
                hit_count=stats["hits"],
                hit_rate=(
                    stats["hits"] / stats["samples"] if stats["samples"] else 0.0
                ),
                wilson_lower=lower,
                wilson_upper=upper,
                mean_excess_return=(
                    stats["excess"] / stats["samples"] if stats["samples"] else 0.0
                ),
                source_outcomes_sha256="walk-forward-runtime",
            )
            rank = rank_bullish_prediction(
                predicted_direction,
                memory,
                minimum_samples=minimum_samples,
            )

            if predicted_direction:
                directional += 1
                if predicted_direction == row.actual_direction:
                    correct += 1
            if predicted_direction == 1:
                bullish += 1
                if row.actual_direction == 1:
                    bullish_correct += 1
                if rank.eligible:
                    ranked += 1
                    ranked_categories[row.sub_category] += 1
                    if row.actual_direction == 1:
                        ranked_correct += 1
                elif rank.degree == "样本不足":
                    insufficient += 1
                else:
                    unverified += 1

            if predicted_direction:
                heapq.heappush(
                    pending_outcomes,
                    (
                        _aware_utc(row.outcome_available_at),
                        str(row.announcement_id),
                        memory_key,
                        predicted_direction == row.actual_direction,
                        row.excess_return,
                    ),
                )

        by_horizon[str(horizon)] = {
            "matured_targets": len(horizon_rows),
            "directional_predictions": directional,
            "directional_accuracy": correct / directional if directional else None,
            "bullish_candidates": bullish,
            "bullish_hit_rate": bullish_correct / bullish if bullish else None,
            "ranked_bullish": ranked,
            "ranked_bullish_hit_rate": ranked_correct / ranked if ranked else None,
            "bullish_rejected_insufficient_history": insufficient,
            "bullish_rejected_accuracy_lower_bound": unverified,
            "ranked_category_counts": dict(ranked_categories),
        }

    report = {
        "contract": "walk-forward-direction-baseline-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "predictor": "frozen event_types direction_baseline sign",
        "warning": "research baseline, not live historical predictions",
        "minimum_matured_samples_per_subcategory": minimum_samples,
        "ranking_rule": "bullish only; Wilson 95% historical hit-rate lower bound > 0.5",
        "by_horizon": by_horizon,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="现有事件方向基线的严格滚动回测")
    parser.add_argument("--minimum-samples", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/direction_baseline_walk_forward.json"),
    )
    args = parser.parse_args()
    run_backtest(
        minimum_samples=max(args.minimum_samples, 1),
        output=args.output,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Plan train-only source backfill needed for a time-stratified extraction audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session

from src.config import PROJECT_ROOT
from src.database.engine import get_engine
from src.database.models import Announcement, AnnouncementSourceArchive, Classification
from src.prediction.splits import load_prediction_split_contract

DEFAULT_PLAN = (
    PROJECT_ROOT / "data" / "backfills" / "structured_extraction_audit_priority.json"
)
SELECTION_SEED = "structured-extraction-audit-source-coverage-v1"


def _half_year(day) -> str:
    return f"{day.year}-H{1 if day.month <= 6 else 2}"


def _atomic_write(path: Path, value: dict) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def build_plan(
    *,
    minimum_per_half_year: int,
    excluded_announcement_ids: set[str],
    output: Path,
) -> dict:
    split = load_prediction_split_contract()
    expected = ["2023-H1", "2023-H2", "2024-H1", "2024-H2"]
    with Session(get_engine()) as session:
        accepted = (
            session.query(Announcement)
            .join(Classification, Classification.announcement_id == Announcement.id)
            .filter(Announcement.published_date.between(split.train.start, split.train.end))
            .filter(Classification.relevance == "core_event")
            .filter(Classification.needs_review.is_(False))
            .all()
        )
        archived_ids = {
            value
            for (value,) in session.query(AnnouncementSourceArchive.announcement_id)
            .filter(AnnouncementSourceArchive.dataset_role == "train")
            .filter(AnnouncementSourceArchive.content_complete.is_(True))
            .distinct()
        }

    archived_counts = Counter(
        _half_year(item.published_date) for item in accepted if item.id in archived_ids
    )
    missing_by_period = defaultdict(list)
    for item in accepted:
        period = _half_year(item.published_date)
        if (
            period in expected
            and item.id not in archived_ids
            and item.announcement_id not in excluded_announcement_ids
        ):
            missing_by_period[period].append(item)
    selected = []
    planned_counts = {}
    for period in expected:
        needed = max(minimum_per_half_year - archived_counts.get(period, 0), 0)
        choices = sorted(
            missing_by_period[period],
            key=lambda item: hashlib.sha256(
                f"{SELECTION_SEED}|{item.announcement_id}".encode()
            ).hexdigest(),
        )
        if len(choices) < needed:
            raise RuntimeError(f"not enough eligible train documents for {period}")
        selected.extend(choices[:needed])
        planned_counts[period] = needed
    announcement_ids = [str(item.announcement_id) for item in selected]
    canonical_ids = "\n".join(sorted(announcement_ids))
    plan = {
        "contract": "structured-extraction-audit-source-plan-v1",
        "dataset_role": "train",
        "market_targets_queried": False,
        "minimum_per_half_year": minimum_per_half_year,
        "existing_archived_counts": {
            period: archived_counts.get(period, 0) for period in expected
        },
        "planned_counts": planned_counts,
        "announcement_count": len(announcement_ids),
        "excluded_failure_count": len(excluded_announcement_ids),
        "announcement_ids_sha256": hashlib.sha256(canonical_ids.encode()).hexdigest(),
        "announcement_ids": announcement_ids,
    }
    _atomic_write(output, plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-per-half-year", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--exclude-state",
        type=Path,
        action="append",
        default=[],
        help="Backfill state whose unresolved failures must not be selected again",
    )
    args = parser.parse_args()
    excluded = set()
    for path in args.exclude_state:
        state = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(
            str(item["announcement_id"])
            for item in state.get("failures", [])
            if item.get("announcement_id")
        )
    plan = build_plan(
        minimum_per_half_year=max(args.minimum_per_half_year, 1),
        excluded_announcement_ids=excluded,
        output=args.output.resolve(),
    )
    summary = {key: value for key, value in plan.items() if key != "announcement_ids"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

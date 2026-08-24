#!/usr/bin/env python
"""Audit source-text and timestamp coverage without reading any market labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session

from src.config import PROJECT_ROOT
from src.database.engine import get_engine
from src.database.models import Announcement
from src.prediction.development_contract import load_causal_development_contract
from src.prediction.splits import load_prediction_split_contract


def parse_eitime(value: object) -> datetime | None:
    """Parse Eastmoney's candidate event timestamp without asserting its semantics."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _has_non_midnight_time(value: object) -> bool:
    text = str(value or "").strip()
    parsed = parse_eitime(text)
    return parsed is not None and parsed.time() != time(0, 0)


def audit(output: Path) -> dict:
    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
    by_role = defaultdict(Counter)
    timestamp = Counter()
    duplicate_keys = Counter()

    with Session(get_engine()) as session:
        query = session.query(
            Announcement.company_id,
            Announcement.announcement_id,
            Announcement.title,
            Announcement.full_text,
            Announcement.pdf_url,
            Announcement.published_date,
            Announcement.source_url,
            Announcement.raw_response,
        ).order_by(Announcement.id)
        for row in query.yield_per(2000):
            role = split.role_for(row.published_date)
            counts = by_role[role]
            counts["total"] += 1
            text_length = len((row.full_text or "").strip())
            counts["full_text_nonempty"] += int(text_length > 0)
            counts["full_text_at_least_200_chars"] += int(text_length >= 200)
            counts["pdf_url_nonempty"] += int(bool((row.pdf_url or "").strip()))
            counts["raw_response_nonempty"] += int(bool(row.raw_response))
            counts["source_is_http_url"] += int(
                str(row.source_url or "").startswith(("http://", "https://"))
            )
            duplicate_keys[(row.company_id, row.published_date, row.title)] += 1

            raw = {}
            if row.raw_response:
                try:
                    raw = json.loads(row.raw_response)
                except (TypeError, json.JSONDecodeError):
                    timestamp["raw_json_invalid"] += 1
            candidate = parse_eitime(raw.get("eiTime"))
            if raw.get("eiTime"):
                timestamp["eitime_present"] += 1
            if candidate is None:
                timestamp["eitime_missing_or_unparseable"] += 1
            else:
                timestamp["eitime_parseable"] += 1
                delta_days = (candidate.date() - row.published_date).days
                timestamp[f"eitime_date_delta_{delta_days}"] += 1
                timestamp["eitime_at_or_after_15_local"] += int(
                    candidate.time() >= time(15, 0)
                )
                timestamp["eitime_before_09_30_local"] += int(
                    candidate.time() < time(9, 30)
                )
            timestamp["notice_date_has_non_midnight_time"] += int(
                _has_non_midnight_time(raw.get("notice_date"))
            )

    report = {
        "contract": "announcement-source-coverage-audit-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "development_contract": development.contract,
        "development_contract_sha256": development.source_sha256,
        "split_contract": split.contract,
        "split_source_sha256": split.source_sha256,
        "market_target_table_queried": False,
        "market_direction_labels_read": False,
        "by_dataset_role": {
            role: dict(counts) for role, counts in sorted(by_role.items())
        },
        "timestamp_candidate_audit": dict(sorted(timestamp.items())),
        "duplicate_company_date_title_groups": sum(
            1 for count in duplicate_keys.values() if count > 1
        ),
        "duplicate_rows_beyond_first": sum(
            count - 1 for count in duplicate_keys.values() if count > 1
        ),
        "important_limitations": [
            "eiTime is only a candidate source timestamp and is not yet trusted as exchange publication time",
            "notice_date is generally date-level and must not be treated as a precise timestamp",
            "empty full_text and PDF fields require a separate immutable-content backfill",
            "no timestamp column will be populated until source semantics are validated",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "validation"
            / "announcement_source_coverage_20260821.json"
        ),
    )
    args = parser.parse_args()
    audit(args.output.resolve())


if __name__ == "__main__":
    main()

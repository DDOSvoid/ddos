#!/usr/bin/env python
"""Build a deterministic train-only PDF audit plan after title classification."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT

CONTRACT = "event-ranking-content-plan-v1"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_pdf_plan(
    *,
    database_path: Path,
    metadata_state_path: Path,
    classification_report_path: Path,
    output_path: Path,
    train_start: date,
    train_end: date,
    per_stratum: int,
    maximum_pdfs: int,
) -> dict[str, object]:
    metadata = json.loads(metadata_state_path.read_text(encoding="utf-8"))
    progress = metadata.get("progress") or {}
    if int(progress.get("completed") or 0) != int(progress.get("total") or 0) or not metadata.get(
        "finished_at"
    ):
        raise ValueError("announcement metadata download is not complete")
    if per_stratum < 1 or maximum_pdfs < 1:
        raise ValueError("PDF sampling bounds must be positive")

    with sqlite3.connect(database_path) as connection:
        train_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM announcements WHERE published_date BETWEEN ? AND ?",
                (train_start.isoformat(), train_end.isoformat()),
            ).fetchone()[0]
        )
        classified_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM announcements a "
                "JOIN classifications c ON c.announcement_id = a.id "
                "WHERE a.published_date BETWEEN ? AND ?",
                (train_start.isoformat(), train_end.isoformat()),
            ).fetchone()[0]
        )
        if classified_count != train_count:
            raise ValueError("title classification is incomplete")
        rows = connection.execute(
            "SELECT a.announcement_id, a.published_date, c.sub_category "
            "FROM announcements a "
            "JOIN classifications c ON c.announcement_id = a.id "
            "WHERE a.published_date BETWEEN ? AND ? "
            "AND c.needs_review = 0 AND c.relevance = 'core_event'",
            (train_start.isoformat(), train_end.isoformat()),
        ).fetchall()

    groups: dict[str, list[str]] = defaultdict(list)
    all_ids: list[str] = []
    for announcement_id, published_date, sub_category in rows:
        parsed = date.fromisoformat(str(published_date)[:10])
        half = "H1" if parsed.month <= 6 else "H2"
        key = f"{parsed.year}-{half}:{sub_category}"
        value = str(announcement_id)
        groups[key].append(value)
        all_ids.append(value)

    selected: set[str] = set()
    for key, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda value: _digest(f"{key}:{value}"))
        selected.update(ordered[:per_stratum])
    if len(selected) < maximum_pdfs:
        remainder = sorted(
            (value for value in all_ids if value not in selected),
            key=lambda value: _digest(f"pdf-fill:{value}"),
        )
        selected.update(remainder[: maximum_pdfs - len(selected)])
    ordered_selected = sorted(selected, key=lambda value: _digest(f"pdf-plan:{value}"))[
        :maximum_pdfs
    ]
    plan: dict[str, object] = {
        "contract": CONTRACT,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_role": "train",
        "official_test_queried": False,
        "market_targets_queried": False,
        "purpose": "content_addressed_pdf_source_audit",
        "body_policy": "all_auto_accepted_core_events_via_paginated_content_api",
        "pdf_policy": "deterministic_stratified_audit_not_all_attachments",
        "selection": {
            "start": train_start.isoformat(),
            "end": train_end.isoformat(),
            "eligible_core_events": len(all_ids),
            "strata": len(groups),
            "per_stratum": per_stratum,
            "maximum_pdfs": maximum_pdfs,
            "selected_pdfs": len(ordered_selected),
            "announcement_ids_sha256": hashlib.sha256(
                "\n".join(sorted(ordered_selected)).encode()
            ).hexdigest(),
        },
        "announcement_ids": ordered_selected,
        "metadata_state_sha256": _sha256(metadata_state_path),
        "classification_report_sha256": _sha256(classification_report_path),
    }
    _write_json_atomic(output_path, plan)
    return plan


def main() -> None:
    root = PROJECT_ROOT / "data" / "backfills" / "event_ranking_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "data" / "ddos.db")
    parser.add_argument(
        "--metadata-state",
        type=Path,
        default=root / "full_announcements_train_2023_2024.json",
    )
    parser.add_argument(
        "--classification-report",
        type=Path,
        default=root / "title_classification.json",
    )
    parser.add_argument("--output", type=Path, default=root / "pdf_audit_plan.json")
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--per-stratum", type=int, default=10)
    parser.add_argument("--maximum-pdfs", type=int, default=2000)
    args = parser.parse_args()
    plan = build_pdf_plan(
        database_path=args.database.resolve(),
        metadata_state_path=args.metadata_state.resolve(),
        classification_report_path=args.classification_report.resolve(),
        output_path=args.output.resolve(),
        train_start=args.start_date,
        train_end=args.end_date,
        per_stratum=args.per_stratum,
        maximum_pdfs=args.maximum_pdfs,
    )
    print(json.dumps(plan["selection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

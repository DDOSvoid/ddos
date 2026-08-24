#!/usr/bin/env python
"""Build a deterministic, train-only human extraction audit set.

No market-target table is imported or queried. Documents are balanced by
half-year and event sub-category, then ordered by a stable hash so reruns with
the same source archive produce the same selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict, deque
from datetime import UTC
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session

from src.config import PROJECT_ROOT
from src.database.engine import get_engine
from src.database.models import (
    Announcement,
    AnnouncementSourceArchive,
    Classification,
)
from src.prediction.splits import load_prediction_split_contract
from src.prediction.structured_extraction import (
    ExtractionSourceDocument,
    StructuredExtractionAuditItem,
    load_extraction_schema,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "human_validation" / "structured_extraction_v1.jsonl"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "human_validation"
    / "structured_extraction_v1.manifest.json"
)
SELECTION_SEED = "structured-extraction-audit-v1"


def _half_year(day) -> str:
    return f"{day.year}-H{1 if day.month <= 6 else 2}"


def _stable_key(row) -> str:
    value = f"{SELECTION_SEED}|{row.archive_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate_rows(session: Session):
    """Return one immutable (earliest retrieved) archive per train announcement."""
    rows = (
        session.query(AnnouncementSourceArchive, Announcement, Classification)
        .join(Announcement, Announcement.id == AnnouncementSourceArchive.announcement_id)
        .join(Classification, Classification.announcement_id == Announcement.id)
        .filter(AnnouncementSourceArchive.dataset_role == "train")
        .filter(AnnouncementSourceArchive.content_complete.is_(True))
        .filter(Classification.relevance == "core_event")
        .filter(Classification.needs_review.is_(False))
        .order_by(AnnouncementSourceArchive.id.asc())
        .all()
    )
    first_by_announcement = {}
    for archive, announcement, classification in rows:
        first_by_announcement.setdefault(
            announcement.id, (archive, announcement, classification)
        )
    return list(first_by_announcement.values())


def _round_robin_cells(cells: dict[tuple, list[tuple]], size: int) -> list[tuple]:
    queues = {
        cell: deque(sorted(items, key=lambda item: _stable_key(item[0])))
        for cell, items in cells.items()
    }
    selected = []
    ordered_cells = sorted(queues)
    while len(selected) < size:
        took_any = False
        for cell in ordered_cells:
            if queues[cell] and len(selected) < size:
                selected.append(queues[cell].popleft())
                took_any = True
        if not took_any:
            break
    return selected


def _stratified_select(
    rows: list[tuple],
    size: int,
    minimum_per_half_year: int,
) -> list[tuple]:
    by_period: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        _, announcement, _ = row
        by_period[_half_year(announcement.published_date)].append(row)
    selected = []
    for period in sorted(by_period):
        period_cells: dict[tuple[str], list[tuple]] = defaultdict(list)
        for row in by_period[period]:
            period_cells[(row[2].sub_category,)].append(row)
        selected.extend(_round_robin_cells(period_cells, minimum_per_half_year))

    selected_archive_ids = {row[0].archive_id for row in selected}
    cells: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for row in rows:
        _, announcement, classification = row
        if row[0].archive_id not in selected_archive_ids:
            cells[
                (_half_year(announcement.published_date), classification.sub_category)
            ].append(row)
    selected.extend(_round_robin_cells(cells, size - len(selected)))
    return selected[:size]


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def build(
    *,
    size: int,
    minimum_per_half_year: int,
    output: Path,
    manifest_path: Path,
) -> dict:
    if size < 300:
        raise ValueError("human extraction gate requires at least 300 documents")
    split = load_prediction_split_contract()
    schema = load_extraction_schema()
    with Session(get_engine()) as session:
        candidates = _candidate_rows(session)
        for archive, announcement, _ in candidates:
            if split.role_for(announcement.published_date) != "train":
                raise PermissionError(
                    f"non-train document reached extraction audit: {archive.archive_id}"
                )
        if len(candidates) < size:
            raise RuntimeError(
                f"only {len(candidates)} eligible archived train documents; need {size}"
            )
        candidate_half_year_counts = Counter(
            _half_year(announcement.published_date)
            for _, announcement, _ in candidates
        )
        expected_half_years = ["2023-H1", "2023-H2", "2024-H1", "2024-H2"]
        undercovered = {
            period: candidate_half_year_counts.get(period, 0)
            for period in expected_half_years
            if candidate_half_year_counts.get(period, 0) < minimum_per_half_year
        }
        if undercovered:
            raise RuntimeError(
                "eligible archive lacks temporal coverage; "
                f"need at least {minimum_per_half_year} per half-year, got {undercovered}"
            )
        selected = _stratified_select(candidates, size, minimum_per_half_year)
        selected_half_year_counts = Counter(
            _half_year(announcement.published_date)
            for _, announcement, _ in selected
        )
        selected_undercovered = {
            period: selected_half_year_counts.get(period, 0)
            for period in expected_half_years
            if selected_half_year_counts.get(period, 0) < minimum_per_half_year
        }
        if selected_undercovered:
            raise RuntimeError(
                "selected audit set violates temporal coverage: "
                f"{selected_undercovered}"
            )
        role_by_archive_id = {}
        h1_2023_seen = 0
        for archive, announcement, _ in selected:
            period = _half_year(announcement.published_date)
            if period == "2023-H1":
                h1_2023_seen += 1
                role = (
                    "prompt_development"
                    if h1_2023_seen <= 100
                    else "model_selection_validation"
                )
            elif period == "2023-H2":
                role = "model_selection_validation"
            else:
                role = "extraction_holdout"
            role_by_archive_id[archive.archive_id] = role
        items = []
        for archive, announcement, classification in selected:
            audit_item_id = hashlib.sha256(
                f"{schema.schema_version}|{archive.archive_id}".encode()
            ).hexdigest()
            source = ExtractionSourceDocument(
                audit_item_id=audit_item_id,
                archive_id=archive.archive_id,
                source_name=archive.source,
                source_art_code=archive.source_art_code,
                announcement_db_id=announcement.id,
                published_date=announcement.published_date,
                dataset_role=archive.dataset_role,
                major_category=classification.major_category,
                sub_category=classification.sub_category,
                title=announcement.title,
                content_text=archive.content_text,
                content_sha256=archive.content_sha256,
                retrieved_at=(
                    archive.retrieved_at
                    if archive.retrieved_at.tzinfo is not None
                    else archive.retrieved_at.replace(tzinfo=UTC)
                ),
                page_manifest=json.loads(archive.page_manifest_json),
                pdf_sha256=archive.pdf_sha256,
                pdf_storage_path=archive.pdf_storage_path,
                development_contract_sha256=archive.development_contract_sha256,
            )
            items.append(
                StructuredExtractionAuditItem(
                    schema_version=schema.schema_version,
                    evaluation_role=role_by_archive_id[archive.archive_id],
                    source=source,
                    requested_fields=schema.fields_for(classification.sub_category),
                )
            )

    jsonl = "".join(
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
        for item in items
    ).encode("utf-8")
    digest = hashlib.sha256(jsonl).hexdigest()
    half_year_counts = Counter(_half_year(item.source.published_date) for item in items)
    major_counts = Counter(item.source.major_category for item in items)
    sub_counts = Counter(item.source.sub_category for item in items)
    role_counts = Counter(item.evaluation_role.value for item in items)
    manifest = {
        "contract": "structured-extraction-human-audit-v1",
        "schema_version": schema.schema_version,
        "schema_sha256": schema.source_sha256,
        "selection_seed": SELECTION_SEED,
        "selection_method": "round_robin_half_year_x_sub_category_then_stable_hash",
        "dataset_role": "train",
        "market_targets_queried": False,
        "requested_size": size,
        "minimum_per_half_year": minimum_per_half_year,
        "selected_size": len(items),
        "candidate_size": len(candidates),
        "output_file": str(output.resolve()),
        "output_sha256": digest,
        "half_year_counts": dict(sorted(half_year_counts.items())),
        "major_category_counts": dict(sorted(major_counts.items())),
        "sub_category_counts": dict(sorted(sub_counts.items())),
        "evaluation_role_counts": dict(sorted(role_counts.items())),
        "evaluation_role_contract": {
            "prompt_development": "first 100 stratified 2023-H1 documents",
            "model_selection_validation": "remaining 2023 documents",
            "extraction_holdout": "all selected 2024 documents; locked until selection",
        },
        "review_status": "pending",
    }
    _atomic_write(output, jsonl)
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--minimum-per-half-year", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    result = build(
        size=args.size,
        minimum_per_half_year=max(args.minimum_per_half_year, 1),
        output=args.output.resolve(),
        manifest_path=args.manifest.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

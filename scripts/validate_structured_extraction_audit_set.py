#!/usr/bin/env python
"""Validate the frozen train-only structured extraction human-audit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.prediction.splits import load_prediction_split_contract
from src.prediction.structured_extraction import (
    StructuredExtractionAuditItem,
    load_extraction_schema,
)

DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "human_validation" / "structured_extraction_v1.jsonl"
)
DEFAULT_MANIFEST = DEFAULT_INPUT.with_suffix(".manifest.json")


def validate(*, input_path: Path, manifest_path: Path) -> dict:
    raw = input_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest["output_sha256"]:
        raise ValueError("audit JSONL hash does not match manifest")
    schema = load_extraction_schema()
    if manifest["schema_sha256"] != schema.source_sha256:
        raise ValueError("extraction schema hash does not match manifest")
    split = load_prediction_split_contract()
    items = [
        StructuredExtractionAuditItem.model_validate_json(line)
        for line in raw.splitlines()
        if line.strip()
    ]
    if len(items) != manifest["selected_size"]:
        raise ValueError("audit item count does not match manifest")
    audit_ids = [item.source.audit_item_id for item in items]
    archive_ids = [item.source.archive_id for item in items]
    if len(audit_ids) != len(set(audit_ids)) or len(archive_ids) != len(set(archive_ids)):
        raise ValueError("audit set contains duplicate identities")
    for item in items:
        if split.role_for(item.source.published_date) != "train":
            raise PermissionError("audit set contains a non-train date")
        if item.schema_version != schema.schema_version:
            raise ValueError("audit item schema version mismatch")
    periods = Counter(
        f"{item.source.published_date.year}-H"
        f"{1 if item.source.published_date.month <= 6 else 2}"
        for item in items
    )
    minimum = int(manifest["minimum_per_half_year"])
    for period in ("2023-H1", "2023-H2", "2024-H1", "2024-H2"):
        if periods.get(period, 0) < minimum:
            raise ValueError(f"audit set undercovers {period}")
    return {
        "valid": True,
        "items": len(items),
        "dataset_role": "train",
        "market_targets_queried": False,
        "output_sha256": digest,
        "schema_sha256": schema.source_sha256,
        "half_year_counts": dict(sorted(periods.items())),
        "source_counts": dict(
            sorted(Counter(item.source.source_name for item in items).items())
        ),
        "evaluation_role_counts": dict(
            sorted(Counter(item.evaluation_role.value for item in items).items())
        ),
        "pending_reviews": sum(item.review.status == "pending" for item in items),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    result = validate(
        input_path=args.input.resolve(),
        manifest_path=args.manifest.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

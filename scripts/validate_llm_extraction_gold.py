#!/usr/bin/env python
"""Validate a source-only LLM extraction gold draft before it is locked."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.prediction.llm_extraction import validate_model_response
from src.prediction.structured_extraction import StructuredExtractionAuditItem


DEFAULT_AUDIT = PROJECT_ROOT / "data" / "human_validation" / "structured_extraction_v1.jsonl"


def _load_audit(path: Path) -> dict[str, StructuredExtractionAuditItem]:
    items = [
        StructuredExtractionAuditItem.model_validate_json(line)
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]
    return {item.source.audit_item_id: item for item in items}


def validate(*, gold_path: Path, audit_path: Path, maximum_quote_length: int) -> dict:
    draft = yaml.safe_load(gold_path.read_text(encoding="utf-8"))
    audit = _load_audit(audit_path)
    documents = draft.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("gold draft documents must be a non-empty list")
    ids = [document.get("audit_item_id") for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("gold draft contains duplicate audit_item_ids")
    if draft.get("target_documents") != len(documents):
        raise ValueError("target_documents does not match documents")
    if draft.get("completed_documents") != len(documents):
        raise ValueError("completed_documents does not match documents")

    counts = Counter()
    quote_count = 0
    taxonomy_rejections = []
    candidate_validator_rejections = []
    for document in documents:
        item_id = document["audit_item_id"]
        if item_id not in audit:
            raise ValueError(f"unknown audit_item_id: {item_id}")
        item = audit[item_id]
        if item.evaluation_role.value != "model_selection_validation":
            raise ValueError(f"gold item is not model_selection_validation: {item_id}")
        fields = document.get("fields")
        if set(fields or {}) != set(item.requested_fields):
            raise ValueError(f"requested field mismatch: {item_id}")
        response_facts = []
        for field_name in item.requested_fields:
            gold = fields[field_name]
            status = gold.get("status")
            counts[status] += 1
            if status == "supported":
                quotes = gold.get("evidence_quotes")
                if not isinstance(quotes, list) or not quotes:
                    raise ValueError(f"supported field has no evidence: {item_id}/{field_name}")
                for quote in quotes:
                    quote_count += 1
                    if len(quote) > maximum_quote_length:
                        raise ValueError(
                            f"evidence quote exceeds {maximum_quote_length} chars: "
                            f"{item_id}/{field_name}/{quote!r}"
                        )
                    if quote not in item.source.content_text:
                        raise ValueError(f"evidence quote is not exact: {item_id}/{field_name}/{quote!r}")
                response_facts.append(
                    {
                        "field_name": field_name,
                        "status": status,
                        "raw_value": quotes[0],
                        "normalized_value": gold.get("normalized_value"),
                        "unit": gold.get("unit"),
                        "evidence": [
                            {"quote": quote, "occurrence": 1} for quote in quotes
                        ],
                        "abstention_reason": None,
                    }
                )
            elif status in {"absent", "ambiguous"}:
                if not gold.get("reason"):
                    raise ValueError(f"abstained field has no reason: {item_id}/{field_name}")
                response_facts.append(
                    {
                        "field_name": field_name,
                        "status": status,
                        "raw_value": None,
                        "normalized_value": None,
                        "unit": None,
                        "evidence": [],
                        "abstention_reason": gold["reason"],
                    }
                )
            else:
                raise ValueError(f"invalid status: {item_id}/{field_name}/{status!r}")
        validation_facts = list(response_facts)
        while True:
            try:
                validated = validate_model_response(item, {"facts": validation_facts})
                break
            except ValueError as exc:
                match = re.search(r"for field ([a-z][a-z0-9_]*)", str(exc))
                field_name = match.group(1) if match else None
                if not field_name or fields.get(field_name, {}).get("status") != "supported":
                    raise
                if any(
                    rejection["field_name"] == field_name
                    for rejection in candidate_validator_rejections
                    if rejection["audit_item_id"] == item_id
                ):
                    raise
                candidate_validator_rejections.append(
                    {
                        "audit_item_id": item_id,
                        "field_name": field_name,
                        "reason": str(exc),
                    }
                )
                validation_facts = [
                    (
                        {
                            "field_name": fact["field_name"],
                            "status": "ambiguous",
                            "raw_value": None,
                            "normalized_value": None,
                            "unit": None,
                            "evidence": [],
                            "abstention_reason": (
                                "gold_supported_but_candidate_validator_rejected: "
                                f"{exc}"
                            ),
                        }
                        if fact["field_name"] == field_name
                        else fact
                    )
                    for fact in validation_facts
                ]
        validated_by_name = {fact.field_name: fact for fact in validated}
        actual_statuses = {
            field_name: fact.status.value for field_name, fact in validated_by_name.items()
        }
        expected_statuses = {
            field_name: fields[field_name]["status"] for field_name in item.requested_fields
        }
        for field_name, expected in expected_statuses.items():
            actual = actual_statuses[field_name]
            if actual == expected:
                continue
            if any(
                rejection["audit_item_id"] == item_id
                and rejection["field_name"] == field_name
                for rejection in candidate_validator_rejections
            ):
                continue
            fact = validated_by_name[field_name]
            if (
                expected == "supported"
                and actual == "ambiguous"
                and (fact.abstention_reason or "").startswith("local_taxonomy_rejected:")
            ):
                taxonomy_rejections.append(
                    {
                        "audit_item_id": item_id,
                        "field_name": field_name,
                        "reason": fact.abstention_reason,
                    }
                )
                continue
            raise ValueError(
                f"gold fact structurally rejected by current validator: "
                f"{item_id}/{field_name}; expected={expected}; actual={actual}; "
                f"reason={fact.abstention_reason}"
            )

    if sum(counts.values()) != sum(
        len(audit[item_id].requested_fields) for item_id in ids
    ):
        raise ValueError("field count mismatch")
    return {
        "contract": "validated-blind-model-selection-gold-v1",
        "documents": len(documents),
        "fields": sum(counts.values()),
        "supported": counts["supported"],
        "absent": counts["absent"],
        "ambiguous": counts["ambiguous"],
        "evidence_quotes": quote_count,
        "maximum_quote_length": maximum_quote_length,
        "requested_field_sets_complete": True,
        "all_supported_quotes_exact": True,
        "all_supported_facts_pass_candidate_validator": not (
            candidate_validator_rejections or taxonomy_rejections
        ),
        "gold_supported_fields_rejected_by_candidate_validator": candidate_validator_rejections,
        "gold_supported_fields_rejected_by_frozen_taxonomy": taxonomy_rejections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", type=Path)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--maximum-quote-length", type=int, default=40)
    args = parser.parse_args()
    result = validate(
        gold_path=args.gold.resolve(),
        audit_path=args.audit.resolve(),
        maximum_quote_length=args.maximum_quote_length,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

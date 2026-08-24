#!/usr/bin/env python
"""Run stateless single-document LLM extraction without exposing gold reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, config
from src.ml.llm_client import LlmClient
from src.prediction.llm_extraction import (
    VALIDATOR_CONTRACT_VERSION,
    build_user_prompt,
    infer_document_role,
    load_prompt_contract,
    validate_model_response,
)
from src.prediction.structured_extraction import FactStatus, StructuredExtractionAuditItem

DEFAULT_AUDIT = (
    PROJECT_ROOT / "data" / "human_validation" / "structured_extraction_v1.jsonl"
)
MODEL_SELECTION_GOLD_LOCK = (
    PROJECT_ROOT
    / "data"
    / "human_validation"
    / "model_selection_gold_batch1.lock.json"
)


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_model_selection_gold_lock(
    *, audit_sha256: str, prompt_version: str, prompt_sha256: str
) -> tuple[str, ...]:
    if not MODEL_SELECTION_GOLD_LOCK.exists():
        raise PermissionError(
            "model-selection candidates are locked until blind gold batch 1 is frozen"
        )
    try:
        lock = json.loads(MODEL_SELECTION_GOLD_LOCK.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PermissionError("model-selection gold lock is unreadable") from exc
    required = {
        "contract": "blind-model-selection-gold-lock-v1",
        "eligible_for_model_selection_score": True,
        "candidate_outputs_seen": False,
        "model_requests_sent_before_gold_freeze": False,
        "audit_sha256": audit_sha256,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "validator_contract_version": VALIDATOR_CONTRACT_VERSION,
    }
    mismatches = {
        key: {"expected": expected, "actual": lock.get(key)}
        for key, expected in required.items()
        if lock.get(key) != expected
    }
    draft_relative = lock.get("gold_draft_path")
    if not isinstance(draft_relative, str):
        mismatches["gold_draft_path"] = {"expected": "project-relative path", "actual": draft_relative}
    else:
        draft_path = (PROJECT_ROOT / draft_relative).resolve()
        gold_root = (PROJECT_ROOT / "data" / "human_validation").resolve()
        try:
            draft_path.relative_to(gold_root)
        except ValueError:
            mismatches["gold_draft_path"] = {"expected": "inside data/human_validation", "actual": str(draft_path)}
        else:
            if not draft_path.exists() or lock.get("gold_draft_sha256") != _sha256_path(draft_path):
                mismatches["gold_draft_sha256"] = {"expected": "current draft hash", "actual": lock.get("gold_draft_sha256")}
    if mismatches:
        raise PermissionError(f"model-selection gold lock contract mismatch: {mismatches}")
    document_count = lock.get("document_count")
    field_count = lock.get("field_count")
    status_counts = [
        lock.get("supported_field_count"),
        lock.get("absent_field_count"),
        lock.get("ambiguous_field_count"),
    ]
    if not isinstance(document_count, int) or document_count <= 0:
        raise PermissionError("model-selection gold lock has an invalid document_count")
    if not isinstance(field_count, int) or field_count <= 0:
        raise PermissionError("model-selection gold lock has an invalid field_count")
    if (
        not all(isinstance(count, int) and count >= 0 for count in status_counts)
        or sum(status_counts) != field_count
    ):
        raise PermissionError("model-selection gold lock field status counts are inconsistent")
    audit_item_ids = lock.get("audit_item_ids")
    if (
        not isinstance(audit_item_ids, list)
        or len(audit_item_ids) != document_count
        or len(set(audit_item_ids)) != document_count
        or not all(isinstance(item_id, str) and len(item_id) == 64 for item_id in audit_item_ids)
    ):
        raise PermissionError("model-selection gold lock has an invalid audit_item_ids list")
    return tuple(audit_item_ids)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_items(path: Path) -> list[StructuredExtractionAuditItem]:
    return [
        StructuredExtractionAuditItem.model_validate_json(line)
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_candidates(path: Path, records: list[dict]) -> None:
    data = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode()
    _atomic_write(path, data)


def _write_failure(path: Path, record: dict) -> None:
    existing = _load_existing(path)
    existing = [
        item for item in existing if item.get("candidate_id") != record["candidate_id"]
    ]
    existing.append(record)
    _write_candidates(path, existing)


def run(
    *,
    audit_path: Path,
    output_path: Path,
    roles: set[str],
    model: str,
    limit: int,
    dry_run: bool,
    audit_item_ids: tuple[str, ...] = (),
    reuse_response_paths: tuple[Path, ...] = (),
) -> dict:
    if "extraction_holdout" in roles:
        raise PermissionError(
            "extraction holdout is locked until prompt and model selection are frozen"
        )
    if audit_item_ids and roles != {"prompt_development"}:
        raise PermissionError(
            "explicit audit-item selection is allowed only for prompt_development"
        )
    if len(set(audit_item_ids)) != len(audit_item_ids):
        raise ValueError("explicit audit-item selection contains duplicates")
    audit_bytes = audit_path.read_bytes()
    audit_sha256 = hashlib.sha256(audit_bytes).hexdigest()
    prompt = load_prompt_contract()
    locked_model_selection_ids: tuple[str, ...] = ()
    if "model_selection_validation" in roles:
        if roles != {"model_selection_validation"}:
            raise PermissionError("model-selection role must run alone")
        locked_model_selection_ids = _assert_model_selection_gold_lock(
            audit_sha256=audit_sha256,
            prompt_version=prompt.version,
            prompt_sha256=prompt.sha256,
        )
    items = [item for item in _load_items(audit_path) if item.evaluation_role.value in roles]
    if locked_model_selection_ids:
        by_id = {item.source.audit_item_id: item for item in items}
        missing = [item_id for item_id in locked_model_selection_ids if item_id not in by_id]
        if missing:
            raise PermissionError(f"locked model-selection items are missing from audit: {missing}")
        items = [by_id[item_id] for item_id in locked_model_selection_ids]
    elif audit_item_ids:
        by_id = {item.source.audit_item_id: item for item in items}
        missing = [item_id for item_id in audit_item_ids if item_id not in by_id]
        if missing:
            raise ValueError(
                f"explicit prompt-development items are missing from audit: {missing}"
            )
        items = [by_id[item_id] for item_id in audit_item_ids]
    if limit > 0:
        items = items[:limit]
    plan = {
        "contract": "stateless-single-document-llm-extraction-v1",
        "audit_sha256": audit_sha256,
        "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "validator_contract_version": VALIDATOR_CONTRACT_VERSION,
        "model": model,
        "thinking": "disabled",
        "temperature": 0.0,
        "max_tokens": config.models.extraction.max_tokens,
        "roles": sorted(roles),
        "explicit_audit_item_ids": list(audit_item_ids),
        "documents_selected": len(items),
        "market_targets_queried": False,
        "human_review_included_in_prompt": False,
        "conversation_history_included": False,
        "memory_included": False,
        "maximum_document_chars": max(
            (len(item.source.content_text) for item in items), default=0
        ),
        "document_role_counts": dict(
            Counter(infer_document_role(item.source.title) for item in items)
        ),
        "reuse_response_sources": [str(path) for path in reuse_response_paths],
    }
    if dry_run:
        plan["dry_run"] = True
        return plan
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured; no LLM requests were sent")
    existing = _load_existing(output_path)
    existing_ids = {record["candidate_id"] for record in existing}
    reusable_responses = {}
    for reuse_path in reuse_response_paths:
        for record in _load_existing(reuse_path):
            if record.get("raw_model_response") is None:
                continue
            key = (
                record.get("audit_item_id"),
                record.get("model_requested"),
                record.get("prompt_sha256"),
                record.get("input_sha256"),
            )
            reusable_responses[key] = record
    client = LlmClient(
        default_model=model,
        timeout=config.models.extraction.request_timeout,
        max_retries=0,
    )
    failure_path = output_path.with_suffix(".failures.jsonl")
    failure_records = _load_existing(failure_path)
    failure_by_id = {record["candidate_id"]: record for record in failure_records}
    consecutive_failures = 0
    counts = Counter()
    for item in items:
        user_prompt = build_user_prompt(item)
        input_sha256 = hashlib.sha256(user_prompt.encode()).hexdigest()
        candidate_id = hashlib.sha256(
            (
                f"{item.source.audit_item_id}|{model}|{prompt.sha256}|"
                f"{input_sha256}|{VALIDATOR_CONTRACT_VERSION}"
            ).encode()
        ).hexdigest()
        if candidate_id in existing_ids:
            counts["skipped_existing"] += 1
            continue
        response = None
        api_metadata = None
        try:
            reusable_candidate = reusable_responses.get(
                (item.source.audit_item_id, model, prompt.sha256, input_sha256)
            )
            reusable_failure = failure_by_id.get(candidate_id)
            if reusable_candidate:
                response = reusable_candidate["raw_model_response"]
                api_metadata = reusable_candidate.get("api_metadata")
                facts = validate_model_response(
                    item, response, abstain_invalid_supported_fields=True
                )
                counts["recovered_from_candidate_audit"] += 1
            elif reusable_failure and reusable_failure.get("model_response"):
                response = reusable_failure["model_response"]
                api_metadata = reusable_failure.get("api_metadata")
                facts = validate_model_response(
                    item, response, abstain_invalid_supported_fields=True
                )
                counts["recovered_from_failure_audit"] += 1
            else:
                response, api_metadata = client.complete_json_audited(
                    system_prompt=prompt.text,
                    user_prompt=user_prompt,
                    model=model,
                    max_tokens=config.models.extraction.max_tokens,
                    temperature=0.0,
                    thinking="disabled",
                )
                facts = validate_model_response(
                    item, response, abstain_invalid_supported_fields=True
                )
        except Exception as exc:
            _write_failure(
                failure_path,
                {
                    "contract": "stateless-single-document-llm-extraction-failure-v1",
                    "candidate_id": candidate_id,
                    "audit_item_id": item.source.audit_item_id,
                    "archive_id": item.source.archive_id,
                    "evaluation_role": item.evaluation_role.value,
                    "model_requested": model,
                    "prompt_version": prompt.version,
                    "prompt_sha256": prompt.sha256,
                    "input_sha256": input_sha256,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "model_response": response,
                    "api_metadata": api_metadata,
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            counts["failed"] += 1
            consecutive_failures += 1
            if consecutive_failures >= 3:
                counts["circuit_breaker_tripped"] += 1
                break
            continue
        record = {
            "contract": plan["contract"],
            "candidate_id": candidate_id,
            "audit_item_id": item.source.audit_item_id,
            "archive_id": item.source.archive_id,
            "evaluation_role": item.evaluation_role.value,
            "model_requested": model,
            "prompt_version": prompt.version,
            "prompt_sha256": prompt.sha256,
            "validator_contract_version": VALIDATOR_CONTRACT_VERSION,
            "input_sha256": input_sha256,
            "source_content_sha256": item.source.content_sha256,
            "document_role_hint": infer_document_role(item.source.title),
            "human_review_included_in_prompt": False,
            "market_targets_queried": False,
            "conversation_history_included": False,
            "memory_included": False,
            "raw_value_policy": "all_validated_source_evidence_for_text_first_span_otherwise",
            "facts": [fact.model_dump(mode="json") for fact in facts],
            "raw_model_response": response,
            "local_rejected_fields": [
                fact.field_name
                for fact in facts
                if fact.status == FactStatus.AMBIGUOUS
                and (fact.abstention_reason or "").startswith("local_")
            ],
            "api_metadata": api_metadata,
            "created_at": datetime.now(UTC).isoformat(),
        }
        existing.append(record)
        existing_ids.add(candidate_id)
        _write_candidates(output_path, existing)
        if candidate_id in failure_by_id:
            failure_records = [
                item
                for item in failure_records
                if item.get("candidate_id") != candidate_id
            ]
            failure_by_id.pop(candidate_id, None)
            _write_candidates(failure_path, failure_records)
        counts["created"] += 1
        consecutive_failures = 0
    plan["dry_run"] = False
    plan["counts"] = dict(counts)
    plan["output_path"] = str(output_path)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--role", action="append", default=[])
    parser.add_argument("--model", default=config.models.extraction.model)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--audit-item-id",
        action="append",
        default=[],
        help="select an exact prompt-development item; repeat for multiple items",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-responses-from", type=Path, action="append", default=[])
    args = parser.parse_args()
    roles = set(args.role or ["prompt_development"])
    prompt = load_prompt_contract()
    output = args.output or (
        PROJECT_ROOT
        / "data"
        / "llm_extraction"
        / f"{args.model}_{prompt.sha256[:12]}_{VALIDATOR_CONTRACT_VERSION}.jsonl"
    )
    result = run(
        audit_path=args.audit.resolve(),
        output_path=output.resolve(),
        roles=roles,
        model=args.model,
        limit=max(args.limit, 0),
        dry_run=args.dry_run,
        audit_item_ids=tuple(args.audit_item_id),
        reuse_response_paths=tuple(path.resolve() for path in args.reuse_responses_from),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

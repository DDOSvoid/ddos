"""Tests for lock-time source gold validation."""

import hashlib
from datetime import UTC, date, datetime

import pytest
import yaml

from scripts.validate_llm_extraction_gold import validate
from src.prediction.structured_extraction import (
    ExtractionSourceDocument,
    StructuredExtractionAuditItem,
)


def _write_asset_impairment_fixture(tmp_path):
    text = "本次计提资产减值准备将减少公司净利润3,901.81万元。"
    digest = hashlib.sha256(text.encode()).hexdigest()
    item = StructuredExtractionAuditItem(
        schema_version="announcement-structured-evidence-v2",
        evaluation_role="model_selection_validation",
        source=ExtractionSourceDocument(
            audit_item_id="a" * 64,
            archive_id="b" * 64,
            source_name="eastmoney",
            source_art_code="AN1",
            announcement_db_id=1,
            published_date=date(2023, 1, 1),
            dataset_role="train",
            major_category="E",
            sub_category="asset_impairment",
            title="资产减值公告",
            content_text=text,
            content_sha256=digest,
            retrieved_at=datetime(2026, 8, 21, tzinfo=UTC),
            page_manifest=[{
                "page_index": 1,
                "chars": len(text),
                "content_sha256": digest,
            }],
            development_contract_sha256="c" * 64,
        ),
        requested_fields=["profit_effect_cny"],
    )
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text(item.model_dump_json() + "\n", encoding="utf-8")
    gold_path = tmp_path / "gold.yaml"
    gold_path.write_text(
        yaml.safe_dump({
            "target_documents": 1,
            "completed_documents": 1,
            "documents": [{
                "audit_item_id": item.source.audit_item_id,
                "fields": {
                    "profit_effect_cny": {
                        "status": "supported",
                        "evidence_quotes": ["减少公司净利润3,901.81万元"],
                        "normalized_value": "-39018100",
                        "unit": "CNY",
                    }
                },
            }],
        }, allow_unicode=True),
        encoding="utf-8",
    )
    return gold_path, audit_path


def test_strict_gold_validation_catches_post_validator_normalized_sign_change(tmp_path):
    gold_path, audit_path = _write_asset_impairment_fixture(tmp_path)

    relaxed = validate(
        gold_path=gold_path,
        audit_path=audit_path,
        maximum_quote_length=40,
    )
    assert relaxed["documents"] == 1

    with pytest.raises(ValueError, match="normalized value/unit does not match"):
        validate(
            gold_path=gold_path,
            audit_path=audit_path,
            maximum_quote_length=40,
            require_normalized_value_match=True,
        )

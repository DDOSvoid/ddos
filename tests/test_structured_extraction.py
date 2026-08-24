"""Tests for evidence-bound structured extraction records."""

import hashlib
from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from src.prediction.structured_extraction import (
    EvidenceSpan,
    ExtractionSourceDocument,
    HumanExtractionReview,
    StructuredExtractionAuditItem,
    StructuredFact,
    load_extraction_schema,
)


def _source(text: str = "合同金额为1亿元。") -> ExtractionSourceDocument:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ExtractionSourceDocument(
        audit_item_id="a" * 64,
        archive_id="b" * 64,
        source_name="eastmoney",
        source_art_code="AN202401010000000001",
        announcement_db_id=1,
        published_date=date(2024, 1, 1),
        dataset_role="train",
        major_category="E",
        sub_category="major_contract",
        title="重大合同公告",
        content_text=text,
        content_sha256=digest,
        retrieved_at=datetime(2026, 8, 21, tzinfo=UTC),
        page_manifest=[
            {
                "page_index": 1,
                "chars": len(text),
                "content_sha256": digest,
            }
        ],
        development_contract_sha256="c" * 64,
    )


def test_supported_fact_requires_exact_source_evidence():
    source = _source()
    quote = "1亿元"
    start = source.content_text.index(quote)
    item = StructuredExtractionAuditItem(
        schema_version="announcement-structured-evidence-v1",
        evaluation_role="prompt_development",
        source=source,
        requested_fields=["contract_amount_cny"],
        review=HumanExtractionReview(
            status="complete",
            reviewer="human",
            completed_at=datetime(2026, 8, 21, tzinfo=UTC),
            facts=[
                StructuredFact(
                    field_name="contract_amount_cny",
                    status="supported",
                    raw_value=quote,
                    normalized_value="100000000",
                    unit="CNY",
                    evidence=[
                        EvidenceSpan(
                            start_char=start,
                            end_char=start + len(quote),
                            quote=quote,
                            page_index=1,
                        )
                    ],
                )
            ],
        ),
    )
    assert item.review.facts[0].normalized_value == "100000000"


def test_mismatched_quote_and_non_train_source_are_rejected():
    source = _source()
    with pytest.raises(ValidationError, match="quote does not match"):
        StructuredExtractionAuditItem(
            schema_version="announcement-structured-evidence-v1",
            evaluation_role="prompt_development",
            source=source,
            requested_fields=["contract_amount_cny"],
            review=HumanExtractionReview(
                status="complete",
                reviewer="human",
                completed_at=datetime(2026, 8, 21, tzinfo=UTC),
                facts=[
                    StructuredFact(
                        field_name="contract_amount_cny",
                        status="supported",
                        raw_value="1亿元",
                        evidence=[
                            EvidenceSpan(start_char=0, end_char=3, quote="错误值")
                        ],
                    )
                ],
            ),
        )

    data = _source().model_dump()
    data["dataset_role"] = "test"
    with pytest.raises(ValidationError, match="train only"):
        ExtractionSourceDocument(**data)


def test_abstention_and_schema_contract():
    fact = StructuredFact(
        field_name="annual_revenue_ratio_pct",
        status="absent",
        abstention_reason="公告未披露占上年度营业收入比例",
    )
    assert fact.evidence == []
    schema = load_extraction_schema()
    fields = schema.fields_for("major_contract")
    assert "contract_amount_cny" in fields
    assert schema.evidence_contract["unsupported_fact_must_abstain"] is True


def test_schema_forbids_future_direction_fields():
    schema = load_extraction_schema()
    all_fields = set(schema.common_fields)
    for fields in schema.event_fields.values():
        all_fields.update(fields)
    forbidden = {"direction", "future_return", "actual_direction", "price_target"}
    assert all_fields.isdisjoint(forbidden)
    assert len(schema.event_fields) == 66

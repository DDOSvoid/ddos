"""Evidence-bound structured facts for causal announcement research.

The models in this module deliberately exclude price direction, future returns,
and subjective impact labels.  A fact is usable only when its quote can be
reconstructed exactly from the immutable source archive identified by SHA-256.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import PROJECT_ROOT

DEFAULT_EXTRACTION_SCHEMA_PATH = (
    PROJECT_ROOT / "config" / "structured_extraction_schema.yaml"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FactStatus(str, Enum):
    SUPPORTED = "supported"
    ABSENT = "absent"
    AMBIGUOUS = "ambiguous"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    COMPLETE = "complete"


class ExtractionEvaluationRole(str, Enum):
    PROMPT_DEVELOPMENT = "prompt_development"
    MODEL_SELECTION_VALIDATION = "model_selection_validation"
    EXTRACTION_HOLDOUT = "extraction_holdout"


class NormalizationUnit(str, Enum):
    CNY = "CNY"
    CNY_PER_SHARE = "CNY_PER_SHARE"
    PCT = "PCT"
    SHARES = "SHARES"
    COUNT = "COUNT"
    DATE = "DATE"
    DAYS = "DAYS"
    MONTHS = "MONTHS"
    YEARS = "YEARS"
    TEXT = "TEXT"
    BOOLEAN = "BOOLEAN"


class EvidenceSpan(StrictModel):
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    quote: str = Field(min_length=1)
    page_index: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_order(self) -> EvidenceSpan:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        if len(self.quote) != self.end_char - self.start_char:
            raise ValueError("quote length does not match character offsets")
        return self


class StructuredFact(StrictModel):
    field_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    status: FactStatus
    raw_value: str | None = None
    normalized_value: str | None = None
    unit: NormalizationUnit | None = None
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def validate_support(self) -> StructuredFact:
        if self.status == FactStatus.SUPPORTED:
            if self.raw_value is None or not self.raw_value.strip():
                raise ValueError("supported fact requires raw_value")
            if not self.evidence:
                raise ValueError("supported fact requires evidence")
            if self.abstention_reason is not None:
                raise ValueError("supported fact cannot have abstention_reason")
        else:
            if not self.abstention_reason:
                raise ValueError("absent or ambiguous fact requires abstention_reason")
            if self.normalized_value is not None:
                raise ValueError("abstained fact cannot have normalized_value")
        return self


class ExtractionSourceDocument(StrictModel):
    audit_item_id: str = Field(min_length=64, max_length=64)
    archive_id: str = Field(min_length=64, max_length=64)
    source_name: str
    source_art_code: str
    announcement_db_id: int = Field(gt=0)
    published_date: date
    dataset_role: str
    major_category: str
    sub_category: str
    title: str
    content_text: str = Field(min_length=1)
    content_sha256: str = Field(min_length=64, max_length=64)
    retrieved_at: datetime
    page_manifest: list[dict[str, Any]]
    pdf_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    pdf_storage_path: str | None = None
    development_contract_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_training_source(self) -> ExtractionSourceDocument:
        if self.dataset_role != "train":
            raise ValueError("structured extraction audit source must be train only")
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise ValueError("retrieved_at must be timezone-aware")
        if (self.pdf_sha256 is None) != (self.pdf_storage_path is None):
            raise ValueError("PDF hash and storage path must be present together")
        actual = hashlib.sha256(self.content_text.encode("utf-8")).hexdigest()
        if actual != self.content_sha256:
            raise ValueError("content_text SHA-256 does not match source archive")
        offset = 0
        for expected_index, page in enumerate(self.page_manifest, 1):
            if page.get("page_index") != expected_index:
                raise ValueError("page manifest indices must be consecutive")
            chars = page.get("chars")
            if not isinstance(chars, int) or chars < 0:
                raise ValueError("page manifest chars must be non-negative integers")
            page_text = self.content_text[offset : offset + chars]
            expected_hash = page.get("content_sha256")
            page_hash = hashlib.sha256(page_text.encode("utf-8")).hexdigest()
            if expected_hash and page_hash != expected_hash:
                raise ValueError("page content SHA-256 does not match manifest")
            offset += chars
        if offset != len(self.content_text):
            raise ValueError("page manifest lengths do not reconstruct content")
        return self


class HumanExtractionReview(StrictModel):
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer: str | None = None
    completed_at: datetime | None = None
    facts: list[StructuredFact] = Field(default_factory=list)
    document_abstention_reason: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_completion(self) -> HumanExtractionReview:
        if self.status == ReviewStatus.PENDING:
            if self.reviewer or self.completed_at or self.facts:
                raise ValueError("pending review cannot contain completed annotations")
        elif not self.reviewer or self.completed_at is None:
            raise ValueError("completed review requires reviewer and completed_at")
        if self.status == ReviewStatus.COMPLETE and not (
            self.facts or self.document_abstention_reason
        ):
            raise ValueError("completed review requires facts or document abstention")
        return self


class StructuredExtractionAuditItem(StrictModel):
    schema_version: str
    evaluation_role: ExtractionEvaluationRole
    source: ExtractionSourceDocument
    requested_fields: list[str]
    review: HumanExtractionReview = Field(default_factory=HumanExtractionReview)

    @model_validator(mode="after")
    def validate_evidence(self) -> StructuredExtractionAuditItem:
        if len(self.requested_fields) != len(set(self.requested_fields)):
            raise ValueError("requested extraction fields contain duplicates")
        requested = set(self.requested_fields)
        for fact in self.review.facts:
            if fact.field_name not in requested:
                raise ValueError(f"fact field is not requested: {fact.field_name}")
            for span in fact.evidence:
                if span.end_char > len(self.source.content_text):
                    raise ValueError("evidence span leaves source document")
                actual = self.source.content_text[span.start_char : span.end_char]
                if actual != span.quote:
                    raise ValueError("evidence quote does not match source offsets")
                if span.page_index is not None:
                    page_start = 0
                    page_end = 0
                    for page in self.source.page_manifest:
                        page_end = page_start + int(page["chars"])
                        if page["page_index"] == span.page_index:
                            break
                        page_start = page_end
                    else:
                        raise ValueError("evidence page_index leaves page manifest")
                    if not (page_start <= span.start_char < span.end_char <= page_end):
                        raise ValueError("evidence offsets do not belong to page_index")
        return self


class ExtractionSchema(StrictModel):
    schema_version: str
    description: str
    common_fields: list[str]
    event_fields: dict[str, list[str]]
    normalization_units: list[str]
    evidence_contract: dict[str, Any]
    source_path: Path
    source_sha256: str

    def fields_for(self, sub_category: str) -> list[str]:
        return list(dict.fromkeys(self.common_fields + self.event_fields.get(sub_category, [])))


def load_extraction_schema(
    path: Path = DEFAULT_EXTRACTION_SCHEMA_PATH,
) -> ExtractionSchema:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    schema = ExtractionSchema(
        **raw,
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    if not schema.evidence_contract.get("exact_character_offsets_required"):
        raise ValueError("exact evidence offsets must be required")
    if not schema.evidence_contract.get("unsupported_fact_must_abstain"):
        raise ValueError("unsupported facts must abstain")
    if len(schema.common_fields) != len(set(schema.common_fields)):
        raise ValueError("common extraction fields contain duplicates")
    if set(schema.normalization_units) != {item.value for item in NormalizationUnit}:
        raise ValueError("normalization units do not match the code contract")
    for event, fields in schema.event_fields.items():
        if len(fields) != len(set(fields)):
            raise ValueError(f"event extraction fields contain duplicates: {event}")
    return schema

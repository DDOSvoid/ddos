"""Stateless, single-document LLM extraction candidate contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import PROJECT_ROOT
from src.prediction.structured_extraction import (
    EvidenceSpan,
    FactStatus,
    StructuredExtractionAuditItem,
    StructuredFact,
)

BASE_PROMPT_PATH = PROJECT_ROOT / "prompts" / "structured_extraction_v9.txt"
V10_APPENDIX_PATH = PROJECT_ROOT / "prompts" / "structured_extraction_v10_appendix.txt"
V11_APPENDIX_PATH = PROJECT_ROOT / "prompts" / "structured_extraction_v11_appendix.txt"
V12_APPENDIX_PATH = PROJECT_ROOT / "prompts" / "structured_extraction_v12_appendix.txt"
V13_APPENDIX_PATH = PROJECT_ROOT / "prompts" / "structured_extraction_v13_appendix.txt"
V14_APPENDIX_PATH = PROJECT_ROOT / "prompts" / "structured_extraction_v14_appendix.txt"
V15_APPENDIX_PATH = PROJECT_ROOT / "prompts" / "structured_extraction_v15_appendix.txt"
V16_APPENDIX_PATH = PROJECT_ROOT / "prompts" / "structured_extraction_v16_appendix.txt"
V17_APPENDIX_PATH = PROJECT_ROOT / "prompts" / "structured_extraction_v17_appendix.txt"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "structured_extraction_v18_appendix.txt"
VALIDATOR_CONTRACT_VERSION = "evidence-validator-v10"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RawEvidence(_StrictModel):
    quote: str = Field(min_length=1)
    occurrence: int = Field(default=1, ge=1)


class RawFact(_StrictModel):
    field_name: str
    status: FactStatus
    raw_value: str | None = None
    normalized_value: str | None = None
    unit: str | None = None
    evidence: list[RawEvidence] = Field(default_factory=list)
    abstention_reason: str | None = None

    @field_validator("normalized_value", mode="before")
    @classmethod
    def normalize_json_scalar(cls, value):
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float) and math.isfinite(value):
            return format(value, ".15g")
        raise ValueError("normalized_value must be a finite JSON scalar")

    @model_validator(mode="after")
    def validate_status_shape(self) -> RawFact:
        if self.status == FactStatus.SUPPORTED:
            if not self.raw_value or self.normalized_value is None or self.unit is None:
                raise ValueError("supported fact requires values and unit")
            if not self.evidence or self.abstention_reason is not None:
                raise ValueError("supported fact requires evidence and no abstention")
        elif (
            self.raw_value is not None
            or self.normalized_value is not None
            or self.unit is not None
            or self.evidence
            or not self.abstention_reason
        ):
            raise ValueError("abstained fact must contain only an abstention reason")
        return self


class RawExtractionResponse(_StrictModel):
    facts: list[RawFact]

    @model_validator(mode="after")
    def unique_fields(self) -> RawExtractionResponse:
        names = [fact.field_name for fact in self.facts]
        if len(names) != len(set(names)):
            raise ValueError("model returned duplicate fields")
        return self


@dataclass(frozen=True)
class PromptContract:
    version: str
    text: str
    sha256: str
    path: Path


def load_prompt_contract(path: Path = DEFAULT_PROMPT_PATH) -> PromptContract:
    data = path.read_bytes()
    if path.resolve() == DEFAULT_PROMPT_PATH.resolve():
        data = (
            BASE_PROMPT_PATH.read_bytes()
            + b"\n\n"
            + V10_APPENDIX_PATH.read_bytes()
            + b"\n\n"
            + V11_APPENDIX_PATH.read_bytes()
            + b"\n\n"
            + V12_APPENDIX_PATH.read_bytes()
            + b"\n\n"
            + V13_APPENDIX_PATH.read_bytes()
            + b"\n\n"
            + V14_APPENDIX_PATH.read_bytes()
            + b"\n\n"
            + V15_APPENDIX_PATH.read_bytes()
            + b"\n\n"
            + V16_APPENDIX_PATH.read_bytes()
            + b"\n\n"
            + V17_APPENDIX_PATH.read_bytes()
            + b"\n\n"
            + data
        )
    return PromptContract(
        version="structured-extraction-prompt-v18",
        text=data.decode("utf-8"),
        sha256=hashlib.sha256(data).hexdigest(),
        path=path.resolve(),
    )


def infer_document_role(title: str) -> str:
    """Infer a routing hint from metadata without treating the title as evidence."""
    compact = _compact(title)
    if any(marker in compact for marker in ("独立财务顾问报告", "上市保荐书", "法律意见书")):
        return "external_adviser_report"
    if "报告摘要" in compact:
        return "financial_report_summary"
    if "审核问询函" in compact and "回复" in compact:
        return "inquiry_response_supporting_document"
    if "自查表" in compact or "自查报告" in compact:
        return "compliance_self_check_supporting_document"
    if "资产评估报告" in compact or "资产评估说明" in compact:
        return "valuation_report_supporting_document"
    if "提示性公告" in compact and any(
        marker in compact for marker in ("更新", "申请文件", "修订")
    ):
        return "supporting_update_notice"
    if re.search(r"(?:管理制度|管理办法|工作制度|实施细则)[》）)]?$", compact):
        return "policy_document"
    return "core_event_announcement"


def document_role_instruction(role: str) -> str:
    instructions = {
        "external_adviser_report": (
            "Use only the current adviser conclusion for event_stage; treat issuer board actions "
            "and dates as history; set the report author as counterparty and the issuer as subject_name."
        ),
        "policy_document": (
            "Extract the current formulation/revision/approval of the policy; generic permissions "
            "and limits are not current transaction facts."
        ),
        "supporting_update_notice": (
            "Extract the current update/revision/disclosure action, not the underlying historical issuance."
        ),
        "inquiry_response_supporting_document": (
            "Extract the current inquiry-response action and responding entity, not historical proposal stages."
        ),
        "financial_report_summary": (
            "Use the current report role and its cumulative reporting-period figures."
        ),
        "compliance_self_check_supporting_document": (
            "Use the current self-check conclusion; issuer proposal approvals are historical context."
        ),
        "valuation_report_supporting_document": (
            "Use the current valuation scope, explicitly labelled valuation base date, and conclusion; "
            "the appraisal firm is not a transaction counterparty, and a proposed acquisition is not completed."
        ),
        "core_event_announcement": "Extract the newly disclosed current event from this announcement.",
    }
    return instructions[role]


def build_user_prompt(item: StructuredExtractionAuditItem) -> str:
    """Build one request from one document; human review is intentionally excluded."""
    fields = json.dumps(item.requested_fields, ensure_ascii=False, separators=(",", ":"))
    role = infer_document_role(item.source.title)
    return (
        f"audit_item_id: {item.source.audit_item_id}\n"
        f"announcement_title: {item.source.title}\n"
        f"major_category: {item.source.major_category}\n"
        f"sub_category: {item.source.sub_category}\n"
        f"document_role_hint: {role}\n"
        f"document_role_instruction: {document_role_instruction(role)}\n"
        f"requested_fields: {fields}\n"
        "DOCUMENT_BEGIN\n"
        f"{item.source.content_text}\n"
        "DOCUMENT_END"
    )


def _nth_occurrence(text: str, quote: str, occurrence: int) -> tuple[int, int]:
    start = -1
    cursor = 0
    for _ in range(occurrence):
        start = text.find(quote, cursor)
        if start < 0:
            break
        cursor = start + len(quote)
    if start >= 0:
        return start, start + len(quote)

    compact_text = []
    original_offsets = []
    for index, character in enumerate(text):
        if not character.isspace():
            compact_text.append(character)
            original_offsets.append(index)
    compact_quote = "".join(character for character in quote if not character.isspace())
    if not compact_quote:
        raise ValueError("model evidence quote is empty after whitespace removal")
    compact = "".join(compact_text)
    match_offsets = []
    match_cursor = 0
    while True:
        match = compact.find(compact_quote, match_cursor)
        if match < 0:
            break
        match_offsets.append(match)
        match_cursor = match + len(compact_quote)
    if len(match_offsets) == 1:
        compact_start = match_offsets[0]
        compact_end = compact_start + len(compact_quote)
        return original_offsets[compact_start], original_offsets[compact_end - 1] + 1
    compact_start = -1
    compact_cursor = 0
    for _ in range(occurrence):
        compact_start = compact.find(compact_quote, compact_cursor)
        if compact_start < 0:
            raise ValueError("model evidence quote does not occur as requested")
        compact_cursor = compact_start + len(compact_quote)
    compact_end = compact_start + len(compact_quote)
    return original_offsets[compact_start], original_offsets[compact_end - 1] + 1


def _page_for_span(item: StructuredExtractionAuditItem, start: int, end: int) -> int:
    page_start = 0
    for page in item.source.page_manifest:
        page_end = page_start + int(page["chars"])
        if page_start <= start < end <= page_end:
            return int(page["page_index"])
        page_start = page_end
    raise ValueError("model evidence crosses or leaves source page boundaries")


_NUMERIC_UNITS = {
    "CNY": ("元", "万元", "亿元", "人民币"),
    "CNY_PER_SHARE": ("元", "股"),
    "PCT": ("%", "％", "百分之"),
    "SHARES": ("股",),
    "COUNT": ("人", "名", "家", "项", "个", "户", "份", "笔", "次", "只", "套"),
    "DATE": ("年", "月", "日", "-"),
    "DAYS": ("日", "天", "交易日"),
    "MONTHS": ("月",),
    "YEARS": ("年",),
}

_SCALAR_NUMERIC_UNITS = {
    "CNY",
    "CNY_PER_SHARE",
    "PCT",
    "SHARES",
    "COUNT",
    "DAYS",
    "MONTHS",
    "YEARS",
}

_ENTITY_FIELDS = {
    "counterparty",
    "subject_name",
    "holder_name",
    "pledgee",
    "subscribers",
    "guarantor",
    "guaranteed_party",
    "related_party",
    "recipient",
    "financial_institution",
    "manager",
    "partner_name",
    "applicant",
    "court",
    "administrator",
    "authority",
    "investigating_authority",
    "investigated_subject",
    "creditor",
    "executive_name",
    "successor",
    "new_controller",
    "old_controller",
    "old_auditor",
    "new_auditor",
    "issuing_institution",
}

_TABLE_NUMERIC_FIELD_LABELS = {
    "revenue_cny": ("营业总收入", "营业收入"),
    "revenue_yoy_pct": ("营业总收入", "营业收入"),
    "net_profit_cny": (
        "归属上市公司股东的净利润",
        "归属于上市公司股东的净利润",
        "归属于上市公司股",
    ),
    "net_profit_yoy_pct": (
        "归属上市公司股东的净利润",
        "归属于上市公司股东的净利润",
        "归属于上市公司股",
    ),
    "total_assets_cny": ("总资产",),
    "net_assets_cny": (
        "归属上市公司股东的股东权",
        "归属于上市公司股东的所有者权益",
        "归属于上市公司股",
        "净资产",
    ),
    "book_value_cny": ("账面价值", "账面净资产", "净资产"),
}


def _compact(value: str) -> str:
    return "".join(character for character in value if not character.isspace())


def _page_bounds(
    item: StructuredExtractionAuditItem, page_index: int
) -> tuple[int, int]:
    start = 0
    for page in item.source.page_manifest:
        end = start + int(page["chars"])
        if int(page["page_index"]) == page_index:
            return start, end
        start = end
    raise ValueError(f"source page is missing from manifest: {page_index}")


def _span(item: StructuredExtractionAuditItem, start: int, end: int) -> EvidenceSpan:
    return EvidenceSpan(
        start_char=start,
        end_char=end,
        quote=item.source.content_text[start:end],
        page_index=_page_for_span(item, start, end),
    )


def _all_exact_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    result = []
    cursor = 0
    while needle:
        start = text.find(needle, cursor)
        if start < 0:
            break
        result.append((start, start + len(needle)))
        cursor = start + len(needle)
    return result


def _recover_table_numeric_evidence(
    item: StructuredExtractionAuditItem, fact: RawFact
) -> list[EvidenceSpan] | None:
    """Recover an exact unit/row/cell chain from a uniquely identified table value."""
    if (
        fact.status != FactStatus.SUPPORTED
        or fact.field_name not in _TABLE_NUMERIC_FIELD_LABELS
        or fact.unit not in {"CNY", "PCT", "SHARES", "COUNT"}
        or not fact.raw_value
        or fact.normalized_value is None
    ):
        return None
    number_match = re.search(r"-?[0-9][0-9,]*(?:\.[0-9]+)?", fact.raw_value)
    if not number_match:
        return None
    number_text = number_match.group(0)
    occurrences = _all_exact_occurrences(item.source.content_text, number_text)
    if len(occurrences) != 1:
        return None
    try:
        source_number = Decimal(number_text.replace(",", ""))
        normalized_number = Decimal(str(fact.normalized_value))
    except InvalidOperation:
        return None
    if fact.unit != "CNY" and source_number != normalized_number:
        return None

    cell_start, cell_end = occurrences[0]
    page_index = _page_for_span(item, cell_start, cell_end)
    page_start, _ = _page_bounds(item, page_index)
    row_window_start = max(page_start, cell_start - 260)
    row_prefix = item.source.content_text[row_window_start:cell_start]
    label_match = None
    for label in _TABLE_NUMERIC_FIELD_LABELS[fact.field_name]:
        position = row_prefix.rfind(label)
        if position >= 0 and (
            label_match is None or position > label_match[0]
        ):
            label_match = (position, label)
    if label_match is None:
        return None
    label_start = row_window_start + label_match[0]
    label_end = label_start + len(label_match[1])

    unit_window_start = max(page_start, label_start - 2000)
    unit_prefix = item.source.content_text[unit_window_start:label_start]
    unit_pattern = (
        r"(?:[（(]\s*)?单位\s*[：:]\s*(?:人民币\s*)?(?P<cny_scale>元|万元|亿元)(?:\s*[)）])?"
        if fact.unit == "CNY"
        else r"增减变动幅度\s*[（(]?[%％][)）]?"
        if fact.unit == "PCT"
        else None
    )
    if unit_pattern is None:
        return None
    unit_matches = list(re.finditer(unit_pattern, unit_prefix))
    if not unit_matches:
        return None
    unit_match = unit_matches[-1]
    if fact.unit == "CNY":
        multiplier = {
            "元": Decimal(1),
            "万元": Decimal(10_000),
            "亿元": Decimal(100_000_000),
        }[unit_match.group("cny_scale")]
        if source_number * multiplier != normalized_number:
            return None
    unit_start = unit_window_start + unit_match.start()
    unit_end = unit_window_start + unit_match.end()
    return [
        _span(item, cell_start, cell_end),
        _span(item, label_start, label_end),
        _span(item, unit_start, unit_end),
    ]


def _recover_split_entity_evidence(
    item: StructuredExtractionAuditItem, fact: RawFact
) -> list[EvidenceSpan] | None:
    """Recover a table entity split by intervening cells without crossing pages."""
    if (
        fact.status != FactStatus.SUPPORTED
        or not (fact.field_name in _ENTITY_FIELDS or fact.field_name.endswith("_name"))
        or not fact.normalized_value
        or len(fact.evidence) != 1
        or _compact(fact.evidence[0].quote) != _compact(fact.normalized_value)
    ):
        return None
    normalized = _compact(fact.normalized_value)
    candidates: list[tuple[int, int, int, int]] = []
    for split in range(4, len(normalized) - 2):
        left = normalized[:split]
        right = normalized[split:]
        for left_start, left_end in _all_exact_occurrences(
            item.source.content_text, left
        ):
            right_start = item.source.content_text.find(
                right, left_end, min(len(item.source.content_text), left_end + 401)
            )
            if right_start < 0:
                continue
            right_end = right_start + len(right)
            try:
                if _page_for_span(item, left_start, left_end) != _page_for_span(
                    item, right_start, right_end
                ):
                    continue
            except ValueError:
                continue
            candidates.append((left_start, left_end, right_start, right_end))
    if not candidates:
        return None
    candidates.sort(
        key=lambda value: (
            value[2] - value[1],
            -min(value[1] - value[0], value[3] - value[2]),
        )
    )
    best = candidates[0]
    best_score = (best[2] - best[1], -min(best[1] - best[0], best[3] - best[2]))
    if sum(
        (
            value[2] - value[1],
            -min(value[1] - value[0], value[3] - value[2]),
        )
        == best_score
        for value in candidates
    ) != 1:
        return None
    return [_span(item, best[0], best[1]), _span(item, best[2], best[3])]


def _canonicalize_period_scalar(fact: RawFact) -> RawFact:
    """Safely remove a redundant period unit without inferring or converting values."""
    if fact.status != FactStatus.SUPPORTED or fact.unit not in {"DAYS", "MONTHS", "YEARS"}:
        return fact
    normalized = _compact(fact.normalized_value or "")
    suffixes = {
        "DAYS": ("交易日", "日", "天"),
        "MONTHS": ("个月", "月"),
        "YEARS": ("年",),
    }[fact.unit]
    for suffix in suffixes:
        if normalized.endswith(suffix):
            numeric = normalized[: -len(suffix)]
            if re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", numeric):
                return fact.model_copy(update={"normalized_value": numeric})
    return fact


def _canonicalize_numeric_bounds(
    item: StructuredExtractionAuditItem, facts: list[StructuredFact]
) -> list[StructuredFact]:
    """Normalize signed forecast loss bounds and enforce numeric lower <= upper."""
    by_name = {fact.field_name: fact for fact in facts}
    forecast_type = by_name.get("forecast_type")
    is_loss_forecast = bool(
        item.source.sub_category == "forecast_performance"
        and forecast_type
        and forecast_type.status == FactStatus.SUPPORTED
        and "亏损" in _compact(str(forecast_type.normalized_value or ""))
    )
    replacements: dict[str, StructuredFact] = {}
    pairs = (
        ("net_profit_lower_cny", "net_profit_upper_cny"),
        ("yoy_lower_pct", "yoy_upper_pct"),
        ("plan_amount_lower_cny", "plan_amount_upper_cny"),
    )
    for lower_name, upper_name in pairs:
        lower = by_name.get(lower_name)
        upper = by_name.get(upper_name)
        if not lower or not upper:
            continue
        if lower.status != FactStatus.SUPPORTED or upper.status != FactStatus.SUPPORTED:
            continue
        try:
            lower_value = Decimal(str(lower.normalized_value))
            upper_value = Decimal(str(upper.normalized_value))
        except InvalidOperation:
            continue
        if is_loss_forecast and lower_name.startswith("net_profit_"):
            lower_value = -abs(lower_value)
            upper_value = -abs(upper_value)
        if lower_value > upper_value:
            lower_value, upper_value = upper_value, lower_value
        replacements[lower_name] = lower.model_copy(
            update={"normalized_value": format(lower_value, "f")}
        )
        replacements[upper_name] = upper.model_copy(
            update={"normalized_value": format(upper_value, "f")}
        )
    return [replacements.get(fact.field_name, fact) for fact in facts]


def _canonicalize_category_values(
    item: StructuredExtractionAuditItem, facts: list[StructuredFact]
) -> list[StructuredFact]:
    """Apply category-specific numeric conventions without inferring a value."""
    result = []
    for fact in facts:
        if (
            item.source.sub_category == "asset_impairment"
            and fact.field_name == "profit_effect_cny"
            and fact.status == FactStatus.SUPPORTED
            and fact.normalized_value is not None
        ):
            try:
                value = Decimal(str(fact.normalized_value))
            except InvalidOperation:
                result.append(fact)
                continue
            fact = fact.model_copy(update={"normalized_value": format(abs(value), "f")})
        result.append(fact)
    return result


def _validate_evidence_semantics(
    item: StructuredExtractionAuditItem,
    fact: RawFact,
    evidence: list[EvidenceSpan],
) -> None:
    """Apply deterministic support checks that do not infer missing semantics."""
    if fact.status != FactStatus.SUPPORTED:
        return
    compact_quotes = [_compact(span.quote) for span in evidence]
    normalized = _compact(fact.normalized_value or "")
    if fact.unit in _SCALAR_NUMERIC_UNITS and not re.fullmatch(
        r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", normalized
    ):
        raise ValueError(f"normalized numeric value is invalid for field {fact.field_name}")
    if fact.unit == "DATE" and not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", normalized):
        raise ValueError(f"normalized date value is invalid for field {fact.field_name}")
    if fact.unit == "DATE":
        combined_date_evidence = "".join(compact_quotes)
        has_explicit_day = bool(
            re.search(
                r"(?:[0-9]{4}|[〇零一二三四五六七八九十]{4})年"
                r"(?:[0-9]{1,2}|[一二三四五六七八九十]{1,3})月"
                r"(?:[0-9]{1,2}|[一二三四五六七八九十]{1,3})日",
                combined_date_evidence,
            )
            or re.search(
                r"[0-9]{4}[-./][0-9]{1,2}[-./][0-9]{1,2}",
                combined_date_evidence,
            )
        )
        if not has_explicit_day:
            raise ValueError(f"date evidence has no explicit day for field {fact.field_name}")
    if fact.unit == "BOOLEAN" and normalized not in {"true", "false"}:
        raise ValueError(f"normalized boolean value is invalid for field {fact.field_name}")
    if fact.unit in _NUMERIC_UNITS:
        expected_markers = _NUMERIC_UNITS[fact.unit]
        lexical_zero_cny = (
            fact.unit == "CNY"
            and normalized == "0"
            and any(
                re.search(r"(?:无|不存在|未发生).{0,8}(?:金额|逾期|担保|欠款|余额)", quote)
                for quote in compact_quotes
            )
        )
        lexical_zero_pct = (
            fact.unit == "PCT"
            and normalized == "0"
            and any(
                re.search(
                    r"(?:不再|未再|没有|无).{0,10}(?:持有|拥有).{0,8}(?:股份|股权)",
                    quote,
                )
                for quote in compact_quotes
            )
        )
        if lexical_zero_cny or lexical_zero_pct:
            has_self_contained_numeric_span = True
        elif fact.unit == "SHARES":
            has_self_contained_numeric_span = any(
                re.search(
                    r"[0-9零〇一二三四五六七八九十百千万亿,.]+股(?!权)", quote
                )
                for quote in compact_quotes
            )
        elif fact.unit == "CNY_PER_SHARE":
            has_self_contained_numeric_span = any(
                re.search(r"[0-9零〇一二三四五六七八九十百千万亿]", quote)
                and "元" in quote
                and any(marker in quote for marker in ("股", "份"))
                for quote in compact_quotes
            )
        else:
            has_self_contained_numeric_span = any(
                re.search(r"[0-9零〇一二三四五六七八九十百千万亿]", quote)
                and any(marker in quote for marker in expected_markers)
                for quote in compact_quotes
            )
        if not has_self_contained_numeric_span and len(evidence) >= 3:
            combined = "".join(compact_quotes)
            pages = {span.page_index for span in evidence}
            has_numeric_cell = any(
                re.search(r"[0-9零〇一二三四五六七八九十百千万亿]", quote)
                for quote in compact_quotes
            )
            has_unit_or_date_header = any(marker in combined for marker in expected_markers)
            has_distinct_row_label = any(
                not re.search(r"[0-9]", quote)
                and not any(marker in quote for marker in expected_markers)
                and len(quote) >= 2
                for quote in compact_quotes
            )
            has_self_contained_numeric_span = bool(
                len(pages) == 1
                and has_numeric_cell
                and has_unit_or_date_header
                and has_distinct_row_label
            )
        if not has_self_contained_numeric_span:
            raise ValueError(
                f"numeric evidence is not self-contained for field {fact.field_name}"
            )
    if fact.field_name in _ENTITY_FIELDS or fact.field_name.endswith("_name"):
        split_table_entity = bool(
            normalized
            and "".join(compact_quotes) == normalized
            and len({span.page_index for span in evidence}) == 1
            and all(
                later.start_char >= earlier.end_char
                and later.start_char - earlier.end_char <= 400
                for earlier, later in zip(evidence, evidence[1:])
            )
        )
        if normalized and not (
            any(normalized in quote for quote in compact_quotes) or split_table_entity
        ):
            raise ValueError(
                f"entity normalized value is not fully supported for field {fact.field_name}"
            )


def _apply_cross_field_taxonomy(
    item: StructuredExtractionAuditItem, facts: list[StructuredFact]
) -> list[StructuredFact]:
    """Reject deterministic category conflicts without inventing replacement facts."""
    def reject(fact: StructuredFact, reason: str) -> StructuredFact:
        return StructuredFact(
            field_name=fact.field_name,
            status=FactStatus.AMBIGUOUS,
            raw_value=None,
            normalized_value=None,
            unit=None,
            evidence=[],
            abstention_reason=f"local_taxonomy_rejected: {reason}",
        )

    def mark_absent(fact: StructuredFact, reason: str) -> StructuredFact:
        return StructuredFact(
            field_name=fact.field_name,
            status=FactStatus.ABSENT,
            raw_value=None,
            normalized_value=None,
            unit=None,
            evidence=[],
            abstention_reason=f"local_taxonomy_absent: {reason}",
        )

    result = []
    source_text = _compact(item.source.content_text)
    document_role = infer_document_role(item.source.title)
    for fact in facts:
        evidence_text = _compact("".join(span.quote for span in fact.evidence))
        evidence_context = _compact(
            "".join(
                item.source.content_text[
                    max(0, span.start_char - 120):
                    min(len(item.source.content_text), span.end_char + 120)
                ]
                for span in fact.evidence
            )
        )
        pending_approval_only = (
            item.source.sub_category == "auditor_change"
            and
            fact.field_name == "uncertainty"
            and fact.status == FactStatus.SUPPORTED
            and "尚需提交" in evidence_text
            and "审议" in evidence_text
            and not any(marker in evidence_text for marker in ("不确定", "风险", "能否"))
        )
        if pending_approval_only:
            result.append(reject(fact, "pending approval is not uncertainty"))
            continue
        if fact.status != FactStatus.SUPPORTED:
            result.append(fact)
            continue

        if (
            document_role == "external_adviser_report"
            and fact.field_name == "event_stage"
            and not any(
                marker in evidence_text
                for marker in ("本独立财务顾问认为", "经核查", "结论性意见", "同意推荐")
            )
        ):
            result.append(reject(fact, "issuer historical action is not the current adviser conclusion"))
            continue
        if (
            document_role == "external_adviser_report"
            and fact.field_name == "event_date"
            and any(marker in evidence_text for marker in ("董事会", "监事会", "会议审议"))
            and not any(marker in evidence_text for marker in ("报告出具日", "本报告出具日"))
        ):
            result.append(reject(fact, "issuer meeting date is not the adviser report date"))
            continue
        if (
            document_role == "policy_document"
            and fact.field_name
            not in {"event_stage", "event_date", "counterparty", "subject_name", "uncertainty"}
        ):
            result.append(mark_absent(fact, "generic policy provisions are not current transaction facts"))
            continue
        if (
            fact.field_name == "event_stage"
            and document_role != "financial_report_summary"
            and evidence_text.endswith("公告")
            and not any(
                marker in evidence_text
                for marker in ("审议", "通过", "完成", "收到", "属于", "发生", "披露", "现将", "签署", "推荐")
            )
        ):
            result.append(reject(fact, "document title is not a current event action or status"))
            continue
        if (
            fact.field_name == "event_stage"
            and _compact(item.source.title).endswith("通知债权人的公告")
            and "通知债权人" not in evidence_text
        ):
            result.append(reject(fact, "historical approval is not the current creditor-notification stage"))
            continue
        if (
            fact.field_name == "event_stage"
            and evidence_text in {"签署", "签署了", "通过", "通过了", "收到", "完成"}
        ):
            result.append(reject(fact, "event action verb has no identifiable object or status"))
            continue

        if (
            document_role == "valuation_report_supporting_document"
            and fact.field_name == "event_date"
            and "评估基准日" not in evidence_text
            and "基准日" not in evidence_text
        ):
            result.append(reject(fact, "valuation report event_date must use the labelled valuation base date"))
            continue
        if (
            fact.field_name == "event_date"
            and _compact(item.source.title).endswith("通知债权人的公告")
            and any(marker in evidence_text for marker in ("董事会", "监事会", "会议"))
        ):
            result.append(reject(fact, "historical approval date is not the creditor-notification date"))
            continue
        if (
            fact.field_name == "event_date"
            and fact.evidence
            and min(span.start_char for span in fact.evidence)
            >= max(0, len(item.source.content_text) - 80)
            and not (
                document_role == "valuation_report_supporting_document"
                and any(marker in evidence_text for marker in ("评估基准日", "基准日"))
            )
        ):
            result.append(reject(fact, "footer-only date is not the current action date"))
            continue
        if (
            fact.field_name == "event_stage"
            and item.source.sub_category in {"private_placement", "convertible_bond"}
            and any(label in evidence_text for label in ("上市保荐书", "募集说明书"))
            and not any(
                action in evidence_text
                for action in (
                    "推荐", "审议", "通过", "注册", "受理", "问询", "发行",
                    "上市交易", "更新", "修订", "回复", "补充", "披露",
                )
            )
        ):
            result.append(reject(fact, "document title is not an event action"))
            continue
        if (
            fact.field_name == "uncertainty"
            and (
                any(marker in evidence_text for marker in ("尽快办理", "及时履行信息披露", "办理相应工商"))
                or (
                    "尽快" in evidence_text
                    and "办理" in evidence_text
                    and any(marker in evidence_text for marker in ("登记", "托管", "手续"))
                )
            )
            and not any(marker in evidence_text for marker in ("不确定", "风险", "能否", "可能", "尚未"))
        ):
            result.append(reject(fact, "routine follow-up processing is not material uncertainty"))
            continue
        if (
            fact.field_name == "uncertainty"
            and "尚需提交" in evidence_text
            and "股东大会" in evidence_text
            and "审议" in evidence_text
            and not any(marker in evidence_text for marker in ("不确定", "风险", "能否", "可能"))
        ):
            result.append(reject(fact, "pending shareholder approval is approval_status, not uncertainty"))
            continue
        if (
            fact.field_name == "funding_source"
            and not any(
                marker in evidence_text
                for marker in ("自有资金", "募集资金", "银行贷款", "银行借款", "自筹资金", "财政资金")
            )
        ):
            result.append(reject(fact, "payment or contribution method is not a funding source"))
            continue
        if (
            fact.field_name == "counterparty"
            and any(
                marker in evidence_text
                for marker in ("银行等金融机构", "相关金融机构", "符合资质的金融机构")
            )
            and not any(marker in evidence_text for marker in ("有限公司", "股份有限公司"))
        ):
            result.append(reject(fact, "generic financial-institution class is not a unique counterparty"))
            continue
        if (
            item.source.sub_category == "government_subsidy"
            and fact.field_name == "accounting_period"
            and any(marker in evidence_text for marker in ("收到", "到账", "下发"))
            and not any(
                marker in evidence_text
                for marker in ("使用寿命", "分期计入", "计入损益", "递延收益", "会计期间")
            )
        ):
            result.append(reject(fact, "subsidy receipt date is not the accounting recognition period"))
            continue
        if (
            fact.field_name == "risk_controls"
            and any(marker in evidence_text for marker in ("制定了", "管理制度"))
            and not any(
                marker in evidence_text
                for marker in (
                    "禁止", "限于", "授权", "审核", "审查", "审批", "监督",
                    "额度", "预测", "匹配", "资格", "保值为原则", "止损",
                )
            )
            and any(marker in source_text for marker in ("风险控制措施", "采取的风险控制"))
        ):
            result.append(reject(fact, "management-policy existence is not the substantive risk controls"))
            continue
        if (
            fact.field_name == "hedged_exposure"
            and not any(
                marker in evidence_text
                for marker in (
                    "外币收入", "外币收款", "外币付款", "外币收付", "外币收（付）款",
                    "出口", "应收", "应付", "外币借款", "风险敞口",
                )
            )
        ):
            result.append(reject(fact, "hedging notional amount is not the underlying exposure"))
            continue
        if (
            fact.field_name == "performance_uncertainty"
            and not any(
                marker in evidence_text
                for marker in (
                    "风险", "不确定", "无法", "不能履行", "可能影响", "违约",
                    "不可抗力", "能否", "回款", "执行",
                )
            )
        ):
            result.append(reject(fact, "business effect or audit timing is not performance uncertainty"))
            continue
        if (
            fact.field_name == "enforceability_uncertainty"
            and not any(
                marker in evidence_text
                for marker in ("执行", "可执行", "回款", "收回", "清偿", "偿付")
            )
        ):
            result.append(reject(fact, "general case uncertainty is not enforceability uncertainty"))
            continue
        if (
            item.source.sub_category == "litigation"
            and fact.field_name == "claim_amount_cny"
            and "财产保全" in evidence_context
            and any(marker in evidence_context for marker in ("价值", "设备", "资产"))
            and not any(marker in evidence_context for marker in ("诉讼请求", "请求判令", "欠款", "本金"))
        ):
            result.append(mark_absent(fact, "asset-preservation value is not a legal claim amount"))
            continue
        if (
            item.source.sub_category == "external_investment"
            and fact.field_name == "investment_amount_cny"
            and re.search(r"(?:公司|子公司).{0,20}(?:认缴|出资).{0,20}(?:万元|亿元)", source_text)
            and not (
                any(marker in evidence_text for marker in ("公司", "子公司", "本公司"))
                and any(marker in evidence_text for marker in ("认缴", "出资", "投资"))
            )
        ):
            result.append(reject(fact, "aggregate transaction amount is not the listed group's contribution"))
            continue
        if (
            document_role == "valuation_report_supporting_document"
            and fact.field_name == "counterparty"
            and any(marker in evidence_text for marker in ("资产评估有限公司", "评估事务所", "会计师事务所"))
        ):
            result.append(mark_absent(fact, "valuation or audit report author is not a transaction counterparty"))
            continue
        if (
            item.source.sub_category == "private_placement"
            and fact.field_name == "purpose"
            and (
                ("用于以下项目" in evidence_text and len(evidence_text) <= 30)
                or (
                    any(marker in evidence_text for marker in ("募集资金投资项目", "募投项目"))
                    and "主营业务" in evidence_text
                    and not any(marker in evidence_text for marker in ("建设", "扩产", "补充流动资金", "偿还"))
                )
            )
        ):
            result.append(reject(fact, "generic use-of-proceeds heading has no concrete purpose"))
            continue
        if (
            infer_document_role(item.source.title) == "policy_document"
            and fact.field_name == "event_stage"
            and "本制度自" in evidence_text
            and "之日起生效" in evidence_text
        ):
            result.append(reject(fact, "contingent policy effectiveness clause is not a completed current action"))
            continue
        if (
            fact.field_name in {"performance_targets", "vesting_schedule"}
            and "第一个归属期" in source_text
            and "第二个归属期" in source_text
            and not (
                "第一个归属期" in evidence_text and "第二个归属期" in evidence_text
            )
        ):
            result.append(reject(fact, "one scalar cannot select a single item from multiple vesting periods"))
            continue
        if (
            fact.field_name == "event_date"
            and item.source.sub_category == "abnormal_volatility"
            and "连续" in source_text
            and "交易日" in source_text
            and len(set(re.findall(r"[0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日", source_text))) >= 2
        ):
            result.append(reject(fact, "multi-day volatility period has no unique event_date"))
            continue
        if (
            fact.field_name == "counterparty"
            and any(
                marker in evidence_text
                for marker in (
                    "基金业协会",
                    "证券交易所",
                    "中国证监会",
                    "证券登记结算",
                    "人民法院",
                    "市场监督管理局",
                )
            )
        ):
            result.append(reject(fact, "administrative or judicial body is not a bilateral counterparty"))
            continue
        if (
            item.source.sub_category == "private_placement"
            and fact.field_name == "subscribers"
            and re.search(r"不超过[0-9一二三四五六七八九十]+名", evidence_text)
            and "最终发行对象" in source_text
            and "确定" in source_text
        ):
            result.append(reject(fact, "eligible investor range is not a final subscriber list"))
            continue
        if (
            item.source.sub_category == "related_transaction"
            and fact.field_name == "relationship"
            and "控制" in evidence_text
            and not any(marker in evidence_text for marker in ("控股股东", "实际控制人", "关联"))
        ):
            result.append(reject(fact, "relationship evidence omits the controlling-party basis"))
            continue
        result.append(fact)
    return result


def _recover_contract_performance_uncertainty(
    item: StructuredExtractionAuditItem, facts: list[StructuredFact]
) -> list[StructuredFact]:
    """Reuse explicit contract-execution risk already validated as general uncertainty."""
    if item.source.sub_category not in {"major_contract", "bid_win"}:
        return facts
    by_field = {fact.field_name: fact for fact in facts}
    target = by_field.get("performance_uncertainty")
    fallback = by_field.get("uncertainty")
    if (
        target is None
        or fallback is None
        or target.status != FactStatus.AMBIGUOUS
        or not target.abstention_reason
        or "business effect or audit timing" not in target.abstention_reason
        or fallback.status != FactStatus.SUPPORTED
        or fallback.unit != "TEXT"
        or not fallback.evidence
    ):
        return facts
    evidence_text = "；".join(span.quote.strip() for span in fallback.evidence)
    if not any(marker in evidence_text for marker in ("合同", "中标", "项目")):
        return facts
    if not any(
        marker in evidence_text
        for marker in (
            "风险", "不确定", "不能履行", "无法履行", "可能影响",
            "能否", "执行", "回款", "违约", "不可抗力",
        )
    ):
        return facts
    recovered = StructuredFact(
        field_name="performance_uncertainty",
        status=FactStatus.SUPPORTED,
        raw_value=evidence_text,
        normalized_value=evidence_text,
        unit="TEXT",
        evidence=fallback.evidence,
        abstention_reason=None,
    )
    return [recovered if fact.field_name == "performance_uncertainty" else fact for fact in facts]


def validate_model_response(
    item: StructuredExtractionAuditItem,
    response: dict,
    *,
    abstain_invalid_supported_fields: bool = False,
) -> list[StructuredFact]:
    """Resolve exact quotes to offsets and reject unsupported or extra output."""
    raw = RawExtractionResponse.model_validate(response)
    returned = {fact.field_name for fact in raw.facts}
    requested = set(item.requested_fields)
    if returned != requested:
        missing = sorted(requested - returned)
        extra = sorted(returned - requested)
        raise ValueError(f"model field set mismatch; missing={missing}; extra={extra}")
    facts = []
    for fact in raw.facts:
        fact = _canonicalize_period_scalar(fact)
        try:
            evidence = []
            try:
                for raw_span in fact.evidence:
                    start, end = _nth_occurrence(
                        item.source.content_text,
                        raw_span.quote,
                        raw_span.occurrence,
                    )
                    evidence.append(_span(item, start, end))
            except ValueError:
                recovered = _recover_table_numeric_evidence(
                    item, fact
                ) or _recover_split_entity_evidence(item, fact)
                if recovered is None:
                    raise
                evidence = recovered
            try:
                _validate_evidence_semantics(item, fact, evidence)
            except ValueError:
                recovered = _recover_table_numeric_evidence(item, fact)
                if recovered is None:
                    raise
                evidence = recovered
                _validate_evidence_semantics(item, fact, evidence)
        except ValueError as exc:
            if not abstain_invalid_supported_fields or fact.status != FactStatus.SUPPORTED:
                raise
            facts.append(
                StructuredFact(
                    field_name=fact.field_name,
                    status=FactStatus.AMBIGUOUS,
                    raw_value=None,
                    normalized_value=None,
                    unit=None,
                    evidence=[],
                    abstention_reason=f"local_validator_rejected: {exc}",
                )
            )
            continue
        is_entity = fact.field_name in _ENTITY_FIELDS or fact.field_name.endswith("_name")
        joined_text_evidence = "；".join(
            span.quote.strip() for span in evidence if span.quote.strip()
        )
        facts.append(
            StructuredFact(
                field_name=fact.field_name,
                status=fact.status,
                raw_value=(
                    joined_text_evidence
                    if fact.status == FactStatus.SUPPORTED
                    and fact.unit == "TEXT"
                    and not is_entity
                    else evidence[0].quote
                    if fact.status == FactStatus.SUPPORTED
                    else fact.raw_value
                ),
                normalized_value=(
                    joined_text_evidence
                    if fact.status == FactStatus.SUPPORTED
                    and fact.unit == "TEXT"
                    and not is_entity
                    else fact.normalized_value
                ),
                unit=fact.unit,
                evidence=evidence,
                abstention_reason=fact.abstention_reason,
            )
        )
    facts = _canonicalize_numeric_bounds(item, facts)
    facts = _canonicalize_category_values(item, facts)
    facts = _apply_cross_field_taxonomy(item, facts)
    return _recover_contract_performance_uncertainty(item, facts)

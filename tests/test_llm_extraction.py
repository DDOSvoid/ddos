"""Tests for stateless evidence-bound LLM extraction candidates."""

import hashlib
from datetime import UTC, date, datetime

import pytest

from src.prediction.llm_extraction import (
    build_user_prompt,
    document_role_instruction,
    infer_document_role,
    validate_model_response,
)
from src.prediction.structured_extraction import (
    ExtractionSourceDocument,
    StructuredExtractionAuditItem,
)


def _item() -> StructuredExtractionAuditItem:
    text = "本合同金额为人民币1亿元。"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return StructuredExtractionAuditItem(
        schema_version="announcement-structured-evidence-v1",
        evaluation_role="prompt_development",
        source=ExtractionSourceDocument(
            audit_item_id="a" * 64,
            archive_id="b" * 64,
            source_name="eastmoney",
            source_art_code="AN1",
            announcement_db_id=1,
            published_date=date(2023, 1, 1),
            dataset_role="train",
            major_category="E",
            sub_category="major_contract",
            title="合同公告",
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
        ),
        requested_fields=["contract_amount_cny", "counterparty"],
    )


def _item_for(
    text: str, *, sub_category: str, requested_fields: list[str], title: str | None = None
) -> StructuredExtractionAuditItem:
    item = _item()
    digest = hashlib.sha256(text.encode()).hexdigest()
    source_data = item.source.model_dump()
    source_data.update(
        sub_category=sub_category,
        title=title or source_data["title"],
        content_text=text,
        content_sha256=digest,
        page_manifest=[{"page_index": 1, "chars": len(text), "content_sha256": digest}],
    )
    return item.model_copy(
        update={
            "source": ExtractionSourceDocument(**source_data),
            "requested_fields": requested_fields,
        }
    )


def test_prompt_contains_only_source_and_not_human_review():
    prompt = build_user_prompt(_item())
    assert "DOCUMENT_BEGIN" in prompt
    assert "人民币1亿元" in prompt
    assert "review" not in prompt
    assert "document_role_hint: core_event_announcement" in prompt


def test_model_response_resolves_exact_quote_and_abstention():
    item = _item()
    facts = validate_model_response(
        item,
        {
            "facts": [
                {
                    "field_name": "contract_amount_cny",
                    "status": "supported",
                    "raw_value": "人民币1亿元",
                    "normalized_value": "100000000",
                    "unit": "CNY",
                    "evidence": [{"quote": "人民币1亿元", "occurrence": 1}],
                    "abstention_reason": None,
                },
                {
                    "field_name": "counterparty",
                    "status": "absent",
                    "raw_value": None,
                    "normalized_value": None,
                    "unit": None,
                    "evidence": [],
                    "abstention_reason": "公告未披露交易对手",
                },
            ]
        },
    )
    assert facts[0].evidence[0].start_char == 6
    assert facts[0].evidence[0].page_index == 1
    assert facts[1].status == "absent"


def test_model_response_rejects_missing_field_and_invented_quote():
    item = _item()
    with pytest.raises(ValueError, match="field set mismatch"):
        validate_model_response(item, {"facts": []})
    response = {
        "facts": [
            {
                "field_name": "contract_amount_cny",
                "status": "supported",
                "raw_value": "2亿元",
                "normalized_value": "200000000",
                "unit": "CNY",
                "evidence": [{"quote": "2亿元", "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "counterparty",
                "status": "absent",
                "raw_value": None,
                "normalized_value": None,
                "unit": None,
                "evidence": [],
                "abstention_reason": "未披露",
            },
        ]
    }
    with pytest.raises(ValueError, match="does not occur"):
        validate_model_response(item, response)


def test_supported_raw_value_is_replaced_by_validated_evidence():
    item = _item()
    response = {
        "facts": [
            {
                "field_name": "contract_amount_cny",
                "status": "supported",
                "raw_value": "人民币2亿元",
                "normalized_value": "100000000",
                "unit": "CNY",
                "evidence": [{"quote": "人民币1亿元", "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "counterparty",
                "status": "absent",
                "raw_value": None,
                "normalized_value": None,
                "unit": None,
                "evidence": [],
                "abstention_reason": "未披露",
            },
        ]
    }
    facts = validate_model_response(item, response)
    assert facts[0].raw_value == "人民币1亿元"


def test_pdf_layout_whitespace_is_mapped_back_to_exact_source():
    item = _item()
    text = "本合同金额为人民币\n  1亿元。"
    digest = hashlib.sha256(text.encode()).hexdigest()
    source_data = item.source.model_dump()
    source_data.update(
        content_text=text,
        content_sha256=digest,
        page_manifest=[
            {
                "page_index": 1,
                "chars": len(text),
                "content_sha256": digest,
            }
        ],
    )
    item = item.model_copy(update={"source": ExtractionSourceDocument(**source_data)})
    response = {
        "facts": [
            {
                "field_name": "contract_amount_cny",
                "status": "supported",
                "raw_value": "人民币1亿元",
                "normalized_value": "100000000",
                "unit": "CNY",
                "evidence": [{"quote": "人民币1亿元", "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "counterparty",
                "status": "absent",
                "raw_value": None,
                "normalized_value": None,
                "unit": None,
                "evidence": [],
                "abstention_reason": "未披露",
            },
        ]
    }
    facts = validate_model_response(item, response)
    assert facts[0].raw_value == "人民币\n  1亿元"


def test_numeric_normalized_value_is_canonicalized_to_string():
    item = _item()
    response = {
        "facts": [
            {
                "field_name": "contract_amount_cny",
                "status": "supported",
                "raw_value": "人民币1亿元",
                "normalized_value": 100000000.0,
                "unit": "CNY",
                "evidence": [{"quote": "人民币1亿元", "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "counterparty",
                "status": "absent",
                "raw_value": None,
                "normalized_value": None,
                "unit": None,
                "evidence": [],
                "abstention_reason": "未披露",
            },
        ]
    }
    facts = validate_model_response(item, response)
    assert facts[0].normalized_value == "100000000"


def test_unique_evidence_repairs_wrong_occurrence_index():
    item = _item()
    response = {
        "facts": [
            {
                "field_name": "contract_amount_cny",
                "status": "supported",
                "raw_value": "人民币1亿元",
                "normalized_value": "100000000",
                "unit": "CNY",
                "evidence": [{"quote": "人民币1亿元", "occurrence": 2}],
                "abstention_reason": None,
            },
            {
                "field_name": "counterparty",
                "status": "absent",
                "raw_value": None,
                "normalized_value": None,
                "unit": None,
                "evidence": [],
                "abstention_reason": "未披露",
            },
        ]
    }
    facts = validate_model_response(item, response)
    assert facts[0].evidence[0].start_char == 6


def test_numeric_evidence_requires_value_and_unit_in_same_span():
    item = _item()
    response = {
        "facts": [
            {
                "field_name": "contract_amount_cny",
                "status": "supported",
                "raw_value": "1",
                "normalized_value": "100000000",
                "unit": "CNY",
                "evidence": [{"quote": "1", "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "counterparty",
                "status": "absent",
                "raw_value": None,
                "normalized_value": None,
                "unit": None,
                "evidence": [],
                "abstention_reason": "未披露",
            },
        ]
    }
    with pytest.raises(ValueError, match="numeric evidence is not self-contained"):
        validate_model_response(item, response)


def test_numeric_normalized_value_rejects_non_numeric_text():
    item = _item()
    response = {
        "facts": [
            {
                "field_name": "contract_amount_cny",
                "status": "supported",
                "raw_value": "人民币1亿元",
                "normalized_value": "约1亿元",
                "unit": "CNY",
                "evidence": [{"quote": "人民币1亿元", "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "counterparty",
                "status": "absent",
                "raw_value": None,
                "normalized_value": None,
                "unit": None,
                "evidence": [],
                "abstention_reason": "未披露",
            },
        ]
    }
    with pytest.raises(ValueError, match="normalized numeric value is invalid"):
        validate_model_response(item, response)


def test_text_normalized_value_is_replaced_by_validated_source_evidence():
    item = _item()
    response = {
        "facts": [
            {
                "field_name": "contract_amount_cny",
                "status": "supported",
                "raw_value": "人民币1亿元",
                "normalized_value": "模型自行归纳的摘要",
                "unit": "TEXT",
                "evidence": [{"quote": "人民币1亿元", "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "counterparty",
                "status": "absent",
                "raw_value": None,
                "normalized_value": None,
                "unit": None,
                "evidence": [],
                "abstention_reason": "未披露",
            },
        ]
    }
    facts = validate_model_response(item, response)
    assert facts[0].normalized_value == "人民币1亿元"


def test_invalid_supported_field_can_be_downgraded_without_losing_document():
    item = _item().model_copy(update={"requested_fields": ["share_count", "counterparty"]})
    response = {
        "facts": [
            {
                "field_name": "share_count",
                "status": "supported",
                "raw_value": "1亿元",
                "normalized_value": "100000000",
                "unit": "SHARES",
                "evidence": [{"quote": "1亿元", "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "counterparty",
                "status": "absent",
                "raw_value": None,
                "normalized_value": None,
                "unit": None,
                "evidence": [],
                "abstention_reason": "未披露",
            },
        ]
    }
    facts = validate_model_response(
        item, response, abstain_invalid_supported_fields=True
    )
    assert facts[0].status == "ambiguous"
    assert facts[0].evidence == []
    assert facts[1].status == "absent"


def test_pending_auditor_approval_is_not_uncertainty():
    item = _item()
    text = "本次聘任尚需提交公司股东大会审议。"
    digest = hashlib.sha256(text.encode()).hexdigest()
    source_data = item.source.model_dump()
    source_data.update(
        sub_category="auditor_change",
        content_text=text,
        content_sha256=digest,
        page_manifest=[{"page_index": 1, "chars": len(text), "content_sha256": digest}],
    )
    item = item.model_copy(
        update={
            "source": ExtractionSourceDocument(**source_data),
            "requested_fields": ["uncertainty", "approval_status"],
        }
    )
    supported = {
        "status": "supported",
        "raw_value": text,
        "normalized_value": text,
        "unit": "TEXT",
        "evidence": [{"quote": text, "occurrence": 1}],
        "abstention_reason": None,
    }
    facts = validate_model_response(
        item,
        {
            "facts": [
                {"field_name": "uncertainty", **supported},
                {"field_name": "approval_status", **supported},
            ]
        },
        abstain_invalid_supported_fields=True,
    )
    assert facts[0].status == "ambiguous"
    assert facts[0].abstention_reason.startswith("local_taxonomy_rejected:")
    assert facts[1].status == "supported"


def test_explicit_no_overdue_guarantee_supports_zero_cny():
    item = _item()
    text = "截至公告日，公司无逾期担保。"
    digest = hashlib.sha256(text.encode()).hexdigest()
    source_data = item.source.model_dump()
    source_data.update(
        content_text=text,
        content_sha256=digest,
        page_manifest=[{"page_index": 1, "chars": len(text), "content_sha256": digest}],
    )
    item = item.model_copy(
        update={
            "source": ExtractionSourceDocument(**source_data),
            "requested_fields": ["overdue_guarantee_amount_cny"],
        }
    )
    facts = validate_model_response(
        item,
        {
            "facts": [
                {
                    "field_name": "overdue_guarantee_amount_cny",
                    "status": "supported",
                    "raw_value": "无逾期担保",
                    "normalized_value": "0",
                    "unit": "CNY",
                    "evidence": [{"quote": "无逾期担保", "occurrence": 1}],
                    "abstention_reason": None,
                }
            ]
        },
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "0"


def test_option_grant_price_accepts_cny_per_option_evidence():
    item = _item()
    text = "首次授予部分行权价格：29.77 元/份"
    digest = hashlib.sha256(text.encode()).hexdigest()
    source_data = item.source.model_dump()
    source_data.update(
        content_text=text,
        content_sha256=digest,
        page_manifest=[{"page_index": 1, "chars": len(text), "content_sha256": digest}],
    )
    item = item.model_copy(
        update={
            "source": ExtractionSourceDocument(**source_data),
            "requested_fields": ["grant_price_cny_per_share"],
        }
    )
    facts = validate_model_response(
        item,
        {
            "facts": [
                {
                    "field_name": "grant_price_cny_per_share",
                    "status": "supported",
                    "raw_value": "29.77 元/份",
                    "normalized_value": "29.77",
                    "unit": "CNY_PER_SHARE",
                    "evidence": [{"quote": "29.77 元/份", "occurrence": 1}],
                    "abstention_reason": None,
                }
            ]
        },
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "29.77"


def test_footer_only_date_is_rejected():
    item = _item_for(
        "本员工持股计划所持股票已全部出售完毕。\n公司董事会\n2023年10月28日",
        sub_category="equity_incentive",
        requested_fields=["event_date"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "event_date", "status": "supported", "raw_value": "2023年10月28日", "normalized_value": "2023-10-28", "unit": "DATE", "evidence": [{"quote": "2023年10月28日", "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "ambiguous"
    assert "footer-only" in facts[0].abstention_reason


def test_cash_contribution_method_is_not_funding_source():
    item = _item_for(
        "公司以货币形式认缴新增出资1,275万元。",
        sub_category="external_investment",
        requested_fields=["funding_source"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "funding_source", "status": "supported", "raw_value": "公司以货币形式认缴新增出资1,275万元", "normalized_value": "公司以货币形式认缴新增出资1,275万元", "unit": "TEXT", "evidence": [{"quote": "公司以货币形式认缴新增出资1,275万元", "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "ambiguous"


def test_general_litigation_uncertainty_is_not_enforceability():
    item = _item_for(
        "案件尚未开庭，对公司利润的影响存在不确定性。",
        sub_category="litigation",
        requested_fields=["enforceability_uncertainty"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "enforceability_uncertainty", "status": "supported", "raw_value": "对公司利润的影响存在不确定性", "normalized_value": "对公司利润的影响存在不确定性", "unit": "TEXT", "evidence": [{"quote": "对公司利润的影响存在不确定性", "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "ambiguous"


def test_external_investment_aggregate_is_not_company_contribution():
    item = _item_for(
        "双方拟增资5,000万元，其中公司以自有资金认缴4,500万元。",
        sub_category="external_investment",
        requested_fields=["investment_amount_cny"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "investment_amount_cny", "status": "supported", "raw_value": "拟增资5,000万元", "normalized_value": "50000000", "unit": "CNY", "evidence": [{"quote": "拟增资5,000万元", "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "ambiguous"


def test_private_placement_document_title_is_not_event_stage():
    item = _item_for(
        "向特定对象发行股票上市保荐书",
        sub_category="private_placement",
        requested_fields=["event_stage"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "event_stage", "status": "supported", "raw_value": "上市保荐书", "normalized_value": "上市保荐书", "unit": "TEXT", "evidence": [{"quote": "上市保荐书", "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "ambiguous"


def test_private_placement_range_is_not_final_subscribers():
    text = "本次发行对象不超过35名合格投资者，最终发行对象根据询价结果确定。"
    item = _item_for(
        text,
        sub_category="private_placement",
        requested_fields=["subscribers"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "subscribers", "status": "supported", "raw_value": "不超过35名合格投资者", "normalized_value": "不超过35名合格投资者", "unit": "TEXT", "evidence": [{"quote": "不超过35名合格投资者", "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "ambiguous"


def test_routine_share_registration_is_not_uncertainty():
    text = "公司本次发行新增股份的登记托管手续将尽快在登记结算公司办理完成。"
    item = _item_for(text, sub_category="private_placement", requested_fields=["uncertainty"])
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "uncertainty", "status": "supported", "raw_value": text, "normalized_value": text, "unit": "TEXT", "evidence": [{"quote": text, "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "ambiguous"


def test_generic_business_fit_is_not_private_placement_purpose():
    text = "本次发行募集资金投资项目与公司主营业务密切相关，符合产业政策。"
    item = _item_for(text, sub_category="private_placement", requested_fields=["purpose"])
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "purpose", "status": "supported", "raw_value": text, "normalized_value": text, "unit": "TEXT", "evidence": [{"quote": text, "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "ambiguous"


def test_regulator_is_not_transaction_counterparty():
    text = "合伙企业已在中国证券投资基金业协会完成私募投资基金备案手续。"
    item = _item_for(text, sub_category="external_investment", requested_fields=["counterparty"])
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "counterparty", "status": "supported", "raw_value": "中国证券投资基金业协会", "normalized_value": "中国证券投资基金业协会", "unit": "TEXT", "evidence": [{"quote": "中国证券投资基金业协会", "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "ambiguous"


def test_period_unit_suffix_is_canonicalized_without_value_conversion():
    text = "自新增股份上市之日起六个月内不得转让"
    item = _item_for(text, sub_category="private_placement", requested_fields=["lockup_period"])
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "lockup_period", "status": "supported", "raw_value": "六个月", "normalized_value": "6个月", "unit": "MONTHS", "evidence": [{"quote": "六个月", "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "6"


def test_document_role_router_distinguishes_supporting_documents():
    assert infer_document_role("公司2023年半年度报告摘要") == "financial_report_summary"
    assert infer_document_role("关于限制性股票首次授予的独立财务顾问报告") == "external_adviser_report"
    assert infer_document_role("募集资金管理制度") == "policy_document"
    assert infer_document_role("申请文件财务数据更新的提示性公告") == "supporting_update_notice"
    assert infer_document_role("2023年股票期权激励计划自查表") == "compliance_self_check_supporting_document"
    assert infer_document_role("拟收购股权项目资产评估说明") == "valuation_report_supporting_document"
    assert "current adviser conclusion" in document_role_instruction("external_adviser_report")


def test_private_placement_update_action_is_not_rejected_as_document_title():
    text = "公司对募集说明书等申请文件的财务数据进行内容更新。"
    item = _item_for(
        text,
        sub_category="private_placement",
        requested_fields=["event_stage"],
        title="申请文件财务数据更新的提示性公告",
    )
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "event_stage", "status": "supported", "raw_value": text, "normalized_value": text, "unit": "TEXT", "evidence": [{"quote": text, "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "supported"


def test_pending_shareholder_approval_is_not_uncertainty_for_any_category():
    text = "该事项尚需提交公司股东大会审议。"
    item = _item_for(text, sub_category="related_transaction", requested_fields=["uncertainty"])
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "uncertainty", "status": "supported", "raw_value": text, "normalized_value": text, "unit": "TEXT", "evidence": [{"quote": text, "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "ambiguous"


def test_single_vesting_period_cannot_represent_multiple_schedules():
    text = "第一个归属期为授予后12至24个月。第二个归属期为授予后24至36个月。"
    item = _item_for(text, sub_category="equity_incentive", requested_fields=["vesting_schedule"])
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "vesting_schedule", "status": "supported", "raw_value": "第一个归属期为授予后12至24个月", "normalized_value": "第一个归属期为授予后12至24个月", "unit": "TEXT", "evidence": [{"quote": "第一个归属期为授予后12至24个月", "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "ambiguous"


def test_loss_forecast_bounds_are_signed_and_sorted():
    text = "预计的业绩为亏损，归母净利润亏损1,500万元至2,250万元。"
    item = _item_for(
        text,
        sub_category="forecast_performance",
        requested_fields=["forecast_type", "net_profit_lower_cny", "net_profit_upper_cny"],
    )
    facts = validate_model_response(
        item,
        {"facts": [
            {"field_name": "forecast_type", "status": "supported", "raw_value": "亏损", "normalized_value": "亏损", "unit": "TEXT", "evidence": [{"quote": "亏损", "occurrence": 1}], "abstention_reason": None},
            {"field_name": "net_profit_lower_cny", "status": "supported", "raw_value": "亏损1,500万元至2,250万元", "normalized_value": "15000000", "unit": "CNY", "evidence": [{"quote": "亏损1,500万元至2,250万元", "occurrence": 1}], "abstention_reason": None},
            {"field_name": "net_profit_upper_cny", "status": "supported", "raw_value": "亏损1,500万元至2,250万元", "normalized_value": "22500000", "unit": "CNY", "evidence": [{"quote": "亏损1,500万元至2,250万元", "occurrence": 1}], "abstention_reason": None},
        ]},
    )
    by_name = {fact.field_name: fact for fact in facts}
    assert by_name["net_profit_lower_cny"].normalized_value == "-22500000"
    assert by_name["net_profit_upper_cny"].normalized_value == "-15000000"


def test_policy_future_effective_clause_is_not_current_event_stage():
    text = "第一条制定本制度。本制度自股东大会审议通过之日起生效并施行。"
    item = _item_for(
        text,
        sub_category="fundraising_usage",
        requested_fields=["event_stage"],
        title="募集资金管理制度",
    )
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "event_stage", "status": "supported", "raw_value": "本制度自股东大会审议通过之日起生效并施行", "normalized_value": "本制度自股东大会审议通过之日起生效并施行", "unit": "TEXT", "evidence": [{"quote": "本制度自股东大会审议通过之日起生效并施行", "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "ambiguous"


def test_multi_day_abnormal_period_is_not_single_event_date():
    text = "公司股票于2022年12月29日、2022年12月30日连续两个交易日异常波动。"
    item = _item_for(text, sub_category="abnormal_volatility", requested_fields=["event_date"])
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "event_date", "status": "supported", "raw_value": text, "normalized_value": "2022-12-30", "unit": "DATE", "evidence": [{"quote": text, "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "ambiguous"


def test_adviser_report_rejects_issuer_historical_action_and_date():
    text = "2023年1月3日，公司董事会审议通过了回购议案。综上，本独立财务顾问认为本次回购程序合规。"
    item = _item_for(
        text,
        sub_category="buyback_cancel",
        requested_fields=["event_stage", "event_date"],
        title="关于回购注销事项之独立财务顾问报告",
    )
    facts = validate_model_response(
        item,
        {"facts": [
            {"field_name": "event_stage", "status": "supported", "raw_value": "公司董事会审议通过了回购议案", "normalized_value": "公司董事会审议通过了回购议案", "unit": "TEXT", "evidence": [{"quote": "公司董事会审议通过了回购议案", "occurrence": 1}], "abstention_reason": None},
            {"field_name": "event_date", "status": "supported", "raw_value": "2023年1月3日，公司董事会审议通过了", "normalized_value": "2023-01-03", "unit": "DATE", "evidence": [{"quote": "2023年1月3日，公司董事会审议通过了", "occurrence": 1}], "abstention_reason": None},
        ]},
    )
    assert all(fact.status == "ambiguous" for fact in facts)


def test_policy_category_specific_permission_is_marked_absent():
    text = "本制度经董事会审议通过，公司可以使用自有资金购买理财产品。"
    item = _item_for(
        text,
        sub_category="treasury_management",
        requested_fields=["funding_source"],
        title="委托理财管理制度",
    )
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "funding_source", "status": "supported", "raw_value": "自有资金", "normalized_value": "自有资金", "unit": "TEXT", "evidence": [{"quote": "自有资金", "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "absent"


def test_plain_announcement_title_is_not_event_stage():
    text = "股票交易异常波动公告。公司股票属于股票异常波动情况。"
    item = _item_for(text, sub_category="abnormal_volatility", requested_fields=["event_stage"])
    facts = validate_model_response(
        item,
        {"facts": [{"field_name": "event_stage", "status": "supported", "raw_value": "股票交易异常波动公告", "normalized_value": "股票交易异常波动公告", "unit": "TEXT", "evidence": [{"quote": "股票交易异常波动公告", "occurrence": 1}], "abstention_reason": None}]},
    )
    assert facts[0].status == "ambiguous"


def test_asset_impairment_profit_effect_is_canonicalized_to_absolute_value():
    text = "本次计提资产减值准备将减少公司净利润3,901.81万元。"
    item = _item_for(
        text,
        sub_category="asset_impairment",
        requested_fields=["profit_effect_cny"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "profit_effect_cny",
            "status": "supported",
            "raw_value": "减少公司净利润3,901.81万元",
            "normalized_value": "-39018100",
            "unit": "CNY",
            "evidence": [{"quote": "减少公司净利润3,901.81万元", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "39018100"


def test_same_page_table_header_row_and_cell_jointly_support_numeric_fact():
    text = "单位：人民币元\n项目  2022年末  2021年末\n净资产  581,986,936.64  533,121,780.30"
    item = _item_for(
        text,
        sub_category="asset_sale",
        requested_fields=["book_value_cny"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "book_value_cny",
            "status": "supported",
            "raw_value": "533,121,780.30",
            "normalized_value": "533121780.30",
            "unit": "CNY",
            "evidence": [
                {"quote": "单位：人民币元", "occurrence": 1},
                {"quote": "净资产", "occurrence": 1},
                {"quote": "533,121,780.30", "occurrence": 1},
            ],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "533121780.30"


def test_table_numeric_cell_recovers_row_label_from_unique_same_page_context():
    text = "单位：人民币元\n项目  2022年末\n净资产  533,121,780.30"
    item = _item_for(
        text,
        sub_category="asset_sale",
        requested_fields=["book_value_cny"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "book_value_cny",
            "status": "supported",
            "raw_value": "533,121,780.30",
            "normalized_value": "533121780.30",
            "unit": "CNY",
            "evidence": [
                {"quote": "单位：人民币元", "occurrence": 1},
                {"quote": "533,121,780.30", "occurrence": 1},
            ],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert [span.quote for span in facts[0].evidence] == [
        "533,121,780.30",
        "净资产",
        "单位：人民币元",
    ]


def test_litigation_preservation_asset_value_is_not_claim_amount():
    text = "公司向人民法院申请财产保全：价值4,300万元的设备。"
    item = _item_for(
        text,
        sub_category="litigation",
        requested_fields=["claim_amount_cny"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "claim_amount_cny",
            "status": "supported",
            "raw_value": "价值4,300万元的设备",
            "normalized_value": "43000000",
            "unit": "CNY",
            "evidence": [{"quote": "价值4,300万元的设备", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "absent"
    assert "asset-preservation value" in facts[0].abstention_reason


def test_valuation_report_uses_labelled_base_date():
    text = "本次资产评估基准日为2022年12月31日。"
    item = _item_for(
        text,
        sub_category="asset_sale",
        requested_fields=["event_date"],
        title="拟收购股权项目资产评估说明",
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "event_date",
            "status": "supported",
            "raw_value": "资产评估基准日为2022年12月31日",
            "normalized_value": "2022-12-31",
            "unit": "DATE",
            "evidence": [{"quote": "资产评估基准日为2022年12月31日", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "2022-12-31"


def test_valuation_report_rejects_unlabelled_report_date():
    text = "坤元资产评估有限公司\n二〇二三年六月六日"
    item = _item_for(
        text,
        sub_category="asset_sale",
        requested_fields=["event_date"],
        title="拟收购股权项目资产评估说明",
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "event_date",
            "status": "supported",
            "raw_value": "二〇二三年六月六日",
            "normalized_value": "2023-06-06",
            "unit": "DATE",
            "evidence": [{"quote": "二〇二三年六月六日", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"
    assert "labelled valuation base date" in facts[0].abstention_reason


def test_valuation_report_author_is_not_transaction_counterparty():
    text = "坤元资产评估有限公司接受委托并出具本报告。"
    item = _item_for(
        text,
        sub_category="asset_sale",
        requested_fields=["counterparty"],
        title="拟收购股权项目资产评估说明",
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "counterparty",
            "status": "supported",
            "raw_value": "坤元资产评估有限公司",
            "normalized_value": "坤元资产评估有限公司",
            "unit": "TEXT",
            "evidence": [{"quote": "坤元资产评估有限公司", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "absent"
    assert "report author" in facts[0].abstention_reason


def test_table_numeric_evidence_recovers_unit_row_and_unique_cell():
    text = "（单位：元）\n项目  本报告期  上年同期\n营业总收入  10,468,983,108.89  8,695,335,076.97"
    item = _item_for(
        text,
        sub_category="forecast_express",
        requested_fields=["revenue_cny"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "revenue_cny",
            "status": "supported",
            "raw_value": "10,468,983,108.89",
            "normalized_value": "10468983108.89",
            "unit": "CNY",
            "evidence": [{"quote": "营业总收入 10,468,983,108.89", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "10468983108.89"
    assert [span.quote for span in facts[0].evidence] == [
        "10,468,983,108.89",
        "营业总收入",
        "（单位：元）",
    ]


def test_table_numeric_evidence_recovers_row_label_split_by_other_cells():
    text = "（单位：元）\n归属上市公司股东的股东权  9,377,468,541.27  8,370,468,009.26\n益"
    item = _item_for(
        text,
        sub_category="forecast_express",
        requested_fields=["net_assets_cny"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "net_assets_cny",
            "status": "supported",
            "raw_value": "9,377,468,541.27",
            "normalized_value": "9377468541.27",
            "unit": "CNY",
            "evidence": [{
                "quote": "归属上市公司股东的股东权益 9,377,468,541.27",
                "occurrence": 1,
            }],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert facts[0].evidence[1].quote == "归属上市公司股东的股东权"


def test_table_recovery_rejects_non_unique_numeric_cell():
    text = "（单位：元）\n营业总收入  100.00\n其他项目  100.00"
    item = _item_for(
        text,
        sub_category="forecast_express",
        requested_fields=["revenue_cny"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "revenue_cny",
            "status": "supported",
            "raw_value": "100.00",
            "normalized_value": "100",
            "unit": "CNY",
            "evidence": [{"quote": "100.00", "occurrence": 1}],
            "abstention_reason": None,
        }]},
        abstain_invalid_supported_fields=True,
    )
    assert facts[0].status == "ambiguous"
    assert "numeric evidence is not self-contained" in facts[0].abstention_reason


def test_table_percentage_recovers_column_header_row_and_cell():
    text = "项目  本报告期  上年同期  增减变动幅度（％）\n营业总收入  100.00  89.37  11.90"
    item = _item_for(
        text,
        sub_category="forecast_express",
        requested_fields=["revenue_yoy_pct"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "revenue_yoy_pct",
            "status": "supported",
            "raw_value": "11.90",
            "normalized_value": "11.9",
            "unit": "PCT",
            "evidence": [{"quote": "11.90", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert [span.quote for span in facts[0].evidence] == [
        "11.90",
        "营业总收入",
        "增减变动幅度（％）",
    ]


def test_table_cny_recovers_wanyuan_scale_before_accepting_normalized_value():
    text = "单位：万元\n项目  本报告期  上年同期\n营业总收入  171,465.81  153,238.04"
    item = _item_for(
        text,
        sub_category="forecast_express",
        requested_fields=["revenue_cny"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "revenue_cny",
            "status": "supported",
            "raw_value": "171,465.81",
            "normalized_value": 1714658100,
            "unit": "CNY",
            "evidence": [{"quote": "171,465.81", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert [span.quote for span in facts[0].evidence] == [
        "171,465.81",
        "营业总收入",
        "单位：万元",
    ]


def test_table_split_profit_label_recovers_percentage_cell():
    text = (
        "项目  本报告期  上年同期  增减变动幅度（％）\n"
        "归属于上市公司股  37,639.53  41,694.39  -9.73\n东的净利润"
    )
    item = _item_for(
        text,
        sub_category="forecast_express",
        requested_fields=["net_profit_yoy_pct"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "net_profit_yoy_pct",
            "status": "supported",
            "raw_value": "-9.73",
            "normalized_value": -9.73,
            "unit": "PCT",
            "evidence": [{"quote": "-9.73", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert [span.quote for span in facts[0].evidence] == [
        "-9.73",
        "归属于上市公司股",
        "增减变动幅度（％）",
    ]


def test_split_table_entity_is_recovered_on_one_page():
    text = "质权人\n兴银理财有限\n吴强  336,000  2023.1.5\n责任公司"
    item = _item_for(
        text,
        sub_category="convertible_bond",
        requested_fields=["counterparty"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "counterparty",
            "status": "supported",
            "raw_value": "兴银理财有限责任公司",
            "normalized_value": "兴银理财有限责任公司",
            "unit": "TEXT",
            "evidence": [{"quote": "兴银理财有限责任公司", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert [span.quote for span in facts[0].evidence] == ["兴银理财有限", "责任公司"]


def test_textual_no_remaining_holding_supports_zero_pct():
    text = "本次权益变动后，广东恒锐不再持有易事特股权。"
    item = _item_for(
        text,
        sub_category="equity_change",
        requested_fields=["resulting_holding_ratio_pct"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "resulting_holding_ratio_pct",
            "status": "supported",
            "raw_value": text,
            "normalized_value": "0",
            "unit": "PCT",
            "evidence": [{"quote": text, "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "0"


def test_multiple_text_evidence_spans_are_all_preserved():
    text = "存在汇率波动风险。另存在客户违约风险。"
    item = _item_for(
        text,
        sub_category="hedging_derivatives",
        requested_fields=["uncertainty"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "uncertainty",
            "status": "supported",
            "raw_value": text,
            "normalized_value": "汇率波动风险；客户违约风险",
            "unit": "TEXT",
            "evidence": [
                {"quote": "汇率波动风险", "occurrence": 1},
                {"quote": "客户违约风险", "occurrence": 1},
            ],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].normalized_value == "汇率波动风险；客户违约风险"
    assert facts[0].raw_value == "汇率波动风险；客户违约风险"


def test_creditor_notice_rejects_historical_board_stage_and_date():
    text = "公司于2022年12月26日召开董事会审议通过回购注销议案。公司特此通知债权人。"
    item = _item_for(
        text,
        sub_category="buyback_cancel",
        requested_fields=["event_stage", "event_date"],
        title="关于回购注销部分限制性股票减资暨通知债权人的公告",
    )
    facts = validate_model_response(
        item,
        {"facts": [
            {
                "field_name": "event_stage",
                "status": "supported",
                "raw_value": "董事会审议通过回购注销议案",
                "normalized_value": "董事会审议通过回购注销议案",
                "unit": "TEXT",
                "evidence": [{"quote": "董事会审议通过回购注销议案", "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "event_date",
                "status": "supported",
                "raw_value": "2022年12月26日召开董事会",
                "normalized_value": "2022-12-26",
                "unit": "DATE",
                "evidence": [{"quote": "2022年12月26日召开董事会", "occurrence": 1}],
                "abstention_reason": None,
            },
        ]},
    )
    assert all(fact.status == "ambiguous" for fact in facts)


def test_subsidy_receipt_date_is_not_accounting_period():
    text = "公司于2023年1月19日收到设备补贴。补助将在资产使用寿命内分期计入损益。"
    item = _item_for(
        text,
        sub_category="government_subsidy",
        requested_fields=["accounting_period"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "accounting_period",
            "status": "supported",
            "raw_value": "于2023年1月19日收到",
            "normalized_value": "2023年",
            "unit": "TEXT",
            "evidence": [{"quote": "于2023年1月19日收到", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"
    assert "recognition period" in facts[0].abstention_reason


def test_month_only_evidence_cannot_be_normalized_to_a_day():
    item = _item_for(
        "公司于2022年12月获得政府补助。",
        sub_category="government_subsidy",
        requested_fields=["event_date"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "event_date",
            "status": "supported",
            "raw_value": "2022年12月",
            "normalized_value": "2022-12-31",
            "unit": "DATE",
            "evidence": [{"quote": "2022年12月", "occurrence": 1}],
            "abstention_reason": None,
        }]},
        abstain_invalid_supported_fields=True,
    )
    assert facts[0].status == "ambiguous"
    assert "no explicit day" in facts[0].abstention_reason


def test_policy_existence_alone_is_not_substantive_risk_controls():
    text = "五、采取的风险控制措施：公司已制定了《外汇套期保值管理制度》，并禁止投机交易。"
    item = _item_for(
        text,
        sub_category="hedging_derivatives",
        requested_fields=["risk_controls"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "risk_controls",
            "status": "supported",
            "raw_value": "公司已制定了《外汇套期保值管理制度》",
            "normalized_value": "公司已制定了《外汇套期保值管理制度》",
            "unit": "TEXT",
            "evidence": [{"quote": "公司已制定了《外汇套期保值管理制度》", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"
    assert "substantive risk controls" in facts[0].abstention_reason


def test_event_stage_bare_action_verb_is_rejected():
    item = _item_for(
        "近日，子公司签署了工程总包合同。",
        sub_category="major_contract",
        requested_fields=["event_stage"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "event_stage",
            "status": "supported",
            "raw_value": "签署了",
            "normalized_value": "签署",
            "unit": "TEXT",
            "evidence": [{"quote": "签署了", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"
    assert "no identifiable object" in facts[0].abstention_reason


def test_generic_bank_class_is_not_unique_counterparty():
    text = "公司拟与具有相关业务经营资质的银行等金融机构开展套期保值。"
    item = _item_for(
        text,
        sub_category="hedging_derivatives",
        requested_fields=["counterparty"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "counterparty",
            "status": "supported",
            "raw_value": "具有相关业务经营资质的银行等金融机构",
            "normalized_value": "具有相关业务经营资质的银行等金融机构",
            "unit": "TEXT",
            "evidence": [{"quote": "具有相关业务经营资质的银行等金融机构", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"


def test_hedging_notional_is_not_underlying_exposure():
    text = "外汇套期保值业务规模不超过人民币20,000万元。实际敞口为外币收付款。"
    item = _item_for(
        text,
        sub_category="hedging_derivatives",
        requested_fields=["hedged_exposure"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "hedged_exposure",
            "status": "supported",
            "raw_value": "外汇套期保值业务规模不超过人民币20,000万元",
            "normalized_value": "外汇套期保值业务规模不超过人民币20,000万元",
            "unit": "TEXT",
            "evidence": [{"quote": "外汇套期保值业务规模不超过人民币20,000万元", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"


def test_business_effect_is_not_contract_performance_uncertainty():
    text = "合同预计对未来年度业绩产生积极影响，最终以审计报告为准。合同另存在不可抗力风险。"
    item = _item_for(
        text,
        sub_category="major_contract",
        requested_fields=["performance_uncertainty"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "performance_uncertainty",
            "status": "supported",
            "raw_value": "合同预计对未来年度业绩产生积极影响，最终以审计报告为准",
            "normalized_value": "合同预计对未来年度业绩产生积极影响，最终以审计报告为准",
            "unit": "TEXT",
            "evidence": [{"quote": "合同预计对未来年度业绩产生积极影响，最终以审计报告为准", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"


def test_contract_execution_risk_can_recover_rejected_performance_candidate():
    text = (
        "合同预计对未来年度业绩产生积极影响，最终以审计报告为准。"
        "合同存在不可抗力导致不能履行的风险，客户需求变化可能影响合同执行。"
    )
    item = _item_for(
        text,
        sub_category="major_contract",
        requested_fields=["uncertainty", "performance_uncertainty"],
    )
    facts = validate_model_response(
        item,
        {"facts": [
            {
                "field_name": "uncertainty",
                "status": "supported",
                "raw_value": "合同履约风险",
                "normalized_value": "合同履约风险",
                "unit": "TEXT",
                "evidence": [
                    {"quote": "合同存在不可抗力导致不能履行的风险", "occurrence": 1},
                    {"quote": "客户需求变化可能影响合同执行", "occurrence": 1},
                ],
                "abstention_reason": None,
            },
            {
                "field_name": "performance_uncertainty",
                "status": "supported",
                "raw_value": "合同预计对未来年度业绩产生积极影响，最终以审计报告为准",
                "normalized_value": "合同预计对未来年度业绩产生积极影响，最终以审计报告为准",
                "unit": "TEXT",
                "evidence": [{
                    "quote": "合同预计对未来年度业绩产生积极影响，最终以审计报告为准",
                    "occurrence": 1,
                }],
                "abstention_reason": None,
            },
        ]},
    )
    recovered = next(fact for fact in facts if fact.field_name == "performance_uncertainty")
    assert recovered.status == "supported"
    assert [span.quote for span in recovered.evidence] == [
        "合同存在不可抗力导致不能履行的风险",
        "客户需求变化可能影响合同执行",
    ]


def test_substantive_multi_span_risk_controls_are_preserved():
    text = "采取的风险控制措施：以保值为原则；严格审查合约；业务基于外币收付款预测并匹配期限。"
    item = _item_for(
        text,
        sub_category="hedging_derivatives",
        requested_fields=["risk_controls"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "risk_controls",
            "status": "supported",
            "raw_value": text,
            "normalized_value": text,
            "unit": "TEXT",
            "evidence": [
                {"quote": "以保值为原则", "occurrence": 1},
                {"quote": "严格审查合约", "occurrence": 1},
                {"quote": "业务基于外币收付款预测并匹配期限", "occurrence": 1},
            ],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "以保值为原则；严格审查合约；业务基于外币收付款预测并匹配期限"

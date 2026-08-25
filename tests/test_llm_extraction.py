"""Tests for stateless evidence-bound LLM extraction candidates."""

import hashlib
from datetime import UTC, date, datetime

import pytest

from src.prediction.llm_extraction import (
    build_user_prompt,
    document_role_instruction,
    infer_document_role,
    load_prompt_contract,
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


def test_text_normalized_value_removes_pdf_layout_breaks_only():
    text = "涨幅偏离值累计超过\n\n  20％，属于异常波动。"
    item = _item_for(
        text,
        sub_category="abnormal_volatility",
        requested_fields=["volatility_type"],
    )
    facts = validate_model_response(
        item,
        {
            "facts": [
                {
                    "field_name": "volatility_type",
                    "status": "supported",
                    "raw_value": "涨幅偏离值累计超过 20％",
                    "normalized_value": "涨幅偏离值累计超过\n\n  20％",
                    "unit": "TEXT",
                    "evidence": [
                        {
                            "quote": "涨幅偏离值累计超过\n\n  20％",
                            "occurrence": 1,
                        }
                    ],
                    "abstention_reason": None,
                }
            ]
        },
    )
    assert facts[0].raw_value == "涨幅偏离值累计超过\n\n  20％"
    assert facts[0].normalized_value == "涨幅偏离值累计超过20％"


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
    assert facts[0].status == "absent"
    assert facts[0].abstention_reason.startswith("local_taxonomy_absent:")
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


def test_external_adviser_author_is_not_transaction_counterparty():
    text = "华泰联合证券有限责任公司出具独立财务顾问报告。"
    item = _item_for(
        text,
        sub_category="private_placement",
        requested_fields=["counterparty"],
        title="独立财务顾问报告",
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "counterparty",
            "status": "supported",
            "raw_value": "华泰联合证券有限责任公司",
            "normalized_value": "华泰联合证券有限责任公司",
            "unit": "TEXT",
            "evidence": [{"quote": "华泰联合证券有限责任公司", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "absent"
    assert "external adviser report author" in facts[0].abstention_reason


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


def test_share_release_count_recovers_named_holder_row_not_total_row():
    text = (
        "股东名称  本次解除质押股份数量  质权人\n"
        "林海峰  5,880,000  招商证券股份有限公司\n"
        "合计  5,880,000  -"
    )
    item = _item_for(
        text,
        sub_category="share_pledge",
        requested_fields=["pledged_share_count"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "pledged_share_count",
            "status": "supported",
            "raw_value": "5,880,000",
            "normalized_value": "5880000",
            "unit": "SHARES",
            "evidence": [{"quote": "5,880,000", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert [span.quote for span in facts[0].evidence] == [
        "5,880,000",
        "林海峰  ",
        "本次解除质押股份数量",
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


def test_v22_prompt_freezes_document_role_specificity_and_table_rules():
    prompt = load_prompt_contract()

    assert prompt.version == "structured-extraction-prompt-v22"
    assert "不得自行求和、平均、推算或去重" in prompt.text
    assert "未来上市流通日" in prompt.text
    assert "监管机关不是 counterparty" in prompt.text
    assert "行政复议、行政诉讼、听证或申诉的法定期限不是 reply_deadline" in prompt.text
    assert "财务报表经董事会批准报出属于内部批准信息" in prompt.text
    assert "按照合同规定期限内交货" in prompt.text
    assert "银行、证券公司等金融机构" in prompt.text
    assert "允许分别引用“行标签”“当前期数值”“表头或单位”三个短 quote" in prompt.text
    assert "外部独立财务顾问、保荐机构、评估机构或会计师事务所只是当前文件的出具方" in prompt.text
    assert "质权人是当前质押关系的 counterparty" in prompt.text
    assert "不能截断为“甲公司”" in prompt.text
    assert "不能继承" not in prompt.text


def test_share_increase_plan_end_is_not_current_event_date():
    text = "本次增持计划实施期限自2023年1月1日起至2023年6月30日止。"
    item = _item_for(
        text,
        sub_category="equity_increase",
        requested_fields=["event_date"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "event_date",
            "status": "supported",
            "raw_value": "增持计划实施期限自2023年1月1日起至2023年6月30日止",
            "normalized_value": "2023-06-30",
            "unit": "DATE",
            "evidence": [{
                "quote": "增持计划实施期限自2023年1月1日起至2023年6月30日止",
                "occurrence": 1,
            }],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"
    assert "plan window" in facts[0].abstention_reason


def test_completed_share_increase_date_is_preserved():
    current_action = "2023年6月20日，本次增持计划已实施完毕。"
    text = (
        current_action
        + "公司已按照规定履行信息披露义务，实际增持结果另见本公告后续明细说明。" * 3
    )
    item = _item_for(
        text,
        sub_category="equity_increase",
        requested_fields=["event_date"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "event_date",
            "status": "supported",
            "raw_value": current_action,
            "normalized_value": "2023-06-20",
            "unit": "DATE",
            "evidence": [{"quote": current_action, "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"


def test_lockup_future_listing_time_is_not_current_event_date():
    text = "本次解除限售股份的上市流通时间为2023年7月3日。"
    item = _item_for(
        text,
        sub_category="lockup_expiry",
        requested_fields=["event_date"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "event_date",
            "status": "supported",
            "raw_value": text,
            "normalized_value": "2023-07-03",
            "unit": "DATE",
            "evidence": [{"quote": text, "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"
    assert "future listing date" in facts[0].abstention_reason


@pytest.mark.parametrize("field_name", ["event_stage", "position", "change_type"])
def test_executive_change_must_include_parallel_acting_secretary_action(field_name):
    text = "董事会聘任张三为副总经理，并指定李四代行董事会秘书职责。"
    item = _item_for(
        text,
        sub_category="executive_change",
        requested_fields=[field_name],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": field_name,
            "status": "supported",
            "raw_value": "聘任张三为副总经理",
            "normalized_value": "聘任张三为副总经理",
            "unit": "TEXT",
            "evidence": [{"quote": "聘任张三为副总经理", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"
    assert "acting-board-secretary" in facts[0].abstention_reason


def test_generic_credit_line_omitting_disclosed_types_is_rejected():
    text = "公司申请综合授信额度，品种包括流动资金贷款、银行承兑汇票和信用证。"
    item = _item_for(
        text,
        sub_category="financing_credit",
        requested_fields=["financing_type"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "financing_type",
            "status": "supported",
            "raw_value": "综合授信额度",
            "normalized_value": "综合授信额度",
            "unit": "TEXT",
            "evidence": [{"quote": "综合授信额度", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"
    assert "omits disclosed financing types" in facts[0].abstention_reason


def test_disclosed_financing_types_are_preserved():
    text = "授信品种包括流动资金贷款、银行承兑汇票和信用证。"
    item = _item_for(
        text,
        sub_category="financing_credit",
        requested_fields=["financing_type"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "financing_type",
            "status": "supported",
            "raw_value": "流动资金贷款；银行承兑汇票；信用证",
            "normalized_value": "流动资金贷款；银行承兑汇票；信用证",
            "unit": "TEXT",
            "evidence": [
                {"quote": "流动资金贷款", "occurrence": 1},
                {"quote": "银行承兑汇票", "occurrence": 1},
                {"quote": "信用证", "occurrence": 1},
            ],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "流动资金贷款；银行承兑汇票；信用证"


def test_regulatory_authority_is_absent_as_counterparty():
    text = "深圳证券交易所向公司股东张三出具监管函。"
    item = _item_for(
        text,
        sub_category="regulatory_action",
        requested_fields=["counterparty"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "counterparty",
            "status": "supported",
            "raw_value": "深圳证券交易所",
            "normalized_value": "深圳证券交易所",
            "unit": "TEXT",
            "evidence": [{"quote": "深圳证券交易所", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "absent"
    assert "unilateral regulatory action" in facts[0].abstention_reason


def test_regulated_shareholder_is_not_duplicated_as_common_subject_name():
    text = "公司股东张三因违规减持收到监管函。"
    item = _item_for(
        text,
        sub_category="regulatory_action",
        requested_fields=["subject_name", "subject"],
    )
    response_fact = {
        "status": "supported",
        "raw_value": "张三",
        "normalized_value": "张三",
        "unit": "TEXT",
        "evidence": [{"quote": "股东张三", "occurrence": 1}],
        "abstention_reason": None,
    }
    facts = validate_model_response(
        item,
        {"facts": [
            {"field_name": "subject_name", **response_fact},
            {"field_name": "subject", **response_fact},
        ]},
    )
    by_name = {fact.field_name: fact for fact in facts}
    assert by_name["subject"].status == "supported"
    assert by_name["subject_name"].status == "absent"


@pytest.mark.parametrize("field_name", ["uncertainty", "reply_deadline", "remedy_requirement"])
def test_administrative_appeal_right_is_not_reply_remedy_or_uncertainty(field_name):
    text = "如不服本决定，可在60日内申请行政复议。"
    item = _item_for(
        text,
        sub_category="regulatory_action",
        requested_fields=[field_name],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": field_name,
            "status": "supported",
            "raw_value": text,
            "normalized_value": text,
            "unit": "TEXT",
            "evidence": [{"quote": text, "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "absent"


def test_actual_regulatory_reply_deadline_and_remedy_are_preserved():
    text = "公司应于2023年6月30日前提交书面回复，并按要求限期整改。"
    item = _item_for(
        text,
        sub_category="regulatory_action",
        requested_fields=["reply_deadline", "remedy_requirement"],
    )
    facts = validate_model_response(
        item,
        {"facts": [
            {
                "field_name": "reply_deadline",
                "status": "supported",
                "raw_value": "2023年6月30日前提交书面回复",
                "normalized_value": "2023-06-30",
                "unit": "DATE",
                "evidence": [{"quote": "2023年6月30日前提交书面回复", "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "remedy_requirement",
                "status": "supported",
                "raw_value": "按要求限期整改",
                "normalized_value": "按要求限期整改",
                "unit": "TEXT",
                "evidence": [{"quote": "按要求限期整改", "occurrence": 1}],
                "abstention_reason": None,
            },
        ]},
    )
    assert all(fact.status == "supported" for fact in facts)


def test_intellectual_property_stage_recovers_one_exact_receipt_action():
    text = "公司近日收到国家知识产权局颁发的发明专利证书。该专利类型为发明专利。"
    item = _item_for(
        text,
        sub_category="intellectual_property",
        requested_fields=["event_stage", "property_type"],
    )
    facts = validate_model_response(
        item,
        {"facts": [
            {
                "field_name": "event_stage",
                "status": "supported",
                "raw_value": "公司收到发明专利证书",
                "normalized_value": "公司收到发明专利证书",
                "unit": "TEXT",
                "evidence": [{"quote": "公司收到发明专利证书", "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "property_type",
                "status": "supported",
                "raw_value": "发明专利",
                "normalized_value": "发明专利",
                "unit": "TEXT",
                "evidence": [{"quote": "专利类型为发明专利", "occurrence": 1}],
                "abstention_reason": None,
            },
        ]},
        abstain_invalid_supported_fields=True,
    )
    by_name = {fact.field_name: fact for fact in facts}
    assert by_name["event_stage"].status == "supported"
    assert by_name["event_stage"].evidence[0].quote == (
        "公司近日收到国家知识产权局颁发的发明专利证书"
    )


def test_multiple_bid_projects_cannot_fill_one_scalar_project_name():
    text = (
        "项目名称：甲项目还原炉标段\n项目概况：设备采购。\n"
        "项目名称：乙项目换热器标段\n项目概况：设备采购。\n"
    )
    item = _item_for(
        text,
        sub_category="bid_win",
        requested_fields=["project_name"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "project_name",
            "status": "supported",
            "raw_value": "甲项目还原炉标段",
            "normalized_value": "甲项目还原炉标段",
            "unit": "TEXT",
            "evidence": [{"quote": "甲项目还原炉标段", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"
    assert "multiple independent bid projects" in facts[0].abstention_reason


def test_buyback_cancellation_date_recovers_from_one_exact_action_sentence():
    text = "公司预计上述限制性股票将于 2023 年 1 月 13 日完成注销。后续进展将另行披露。"
    item = _item_for(
        text,
        sub_category="buyback_cancel",
        requested_fields=["registration_date"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "registration_date",
            "status": "supported",
            "raw_value": "2023 年 1 月 13 日",
            "normalized_value": "2023-01-13",
            "unit": "DATE",
            "evidence": [{"quote": "注销日期 2023 年 1 月 13 日", "occurrence": 1}],
            "abstention_reason": None,
        }]},
        abstain_invalid_supported_fields=True,
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "2023-01-13"
    assert "完成注销" in facts[0].evidence[0].quote


def test_synthetic_sum_across_measurement_bases_is_rejected():
    text = "保证金上限为6,000万元，名义本金余额上限为2亿元。"
    item = _item_for(
        text,
        sub_category="hedging_derivatives",
        requested_fields=["maximum_amount_cny"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "maximum_amount_cny",
            "status": "supported",
            "raw_value": text,
            "normalized_value": "260000000",
            "unit": "CNY",
            "evidence": [
                {"quote": "6,000万元", "occurrence": 1},
                {"quote": "2亿元", "occurrence": 1},
            ],
            "abstention_reason": None,
        }]},
        abstain_invalid_supported_fields=True,
    )
    assert facts[0].status == "ambiguous"
    assert "aggregation across rows" in facts[0].abstention_reason


def test_directly_disclosed_sensitive_amount_is_accepted():
    text = "外汇套期保值保证金上限为6,000万元。"
    item = _item_for(
        text,
        sub_category="hedging_derivatives",
        requested_fields=["maximum_amount_cny"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "maximum_amount_cny",
            "status": "supported",
            "raw_value": "6,000万元",
            "normalized_value": "60000000",
            "unit": "CNY",
            "evidence": [{"quote": "6,000万元", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "60000000"


def test_multiple_independent_holders_cannot_fill_scalar_holder_name():
    text = "股东张三、李四拟通过集中竞价减持股份。"
    item = _item_for(
        text,
        sub_category="equity_decrease",
        requested_fields=["holder_name"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "holder_name",
            "status": "supported",
            "raw_value": "张三、李四",
            "normalized_value": "张三、李四",
            "unit": "TEXT",
            "evidence": [{"quote": "张三、李四", "occurrence": 1}],
            "abstention_reason": None,
        }]},
        abstain_invalid_supported_fields=True,
    )
    assert facts[0].status == "ambiguous"
    assert "cannot combine independent holders" in facts[0].abstention_reason


def test_multiple_named_agreement_counterparties_use_separate_evidence():
    text = "公司与北京甲公司签署协议，上海乙公司亦为协议相对方。"
    item = _item_for(
        text,
        sub_category="major_contract",
        requested_fields=["counterparty"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "counterparty",
            "status": "supported",
            "raw_value": "北京甲公司；上海乙公司",
            "normalized_value": "北京甲公司；上海乙公司",
            "unit": "TEXT",
            "evidence": [
                {"quote": "北京甲公司", "occurrence": 1},
                {"quote": "上海乙公司", "occurrence": 1},
            ],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "北京甲公司；上海乙公司"


def test_aggregate_subsidy_with_multiple_receipt_dates_has_no_event_date():
    text = "公司于2023年1月3日收到补助，并于2023年2月6日收到另一笔补助。"
    item = _item_for(
        text,
        sub_category="government_subsidy",
        requested_fields=["event_date"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "event_date",
            "status": "supported",
            "raw_value": "2023年1月3日收到补助",
            "normalized_value": "2023-01-03",
            "unit": "DATE",
            "evidence": [{"quote": "2023年1月3日收到补助", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"
    assert "multiple receipt dates" in facts[0].abstention_reason


def test_contract_performance_start_is_not_signing_event_date():
    text = "合同履行期间自2023年3月1日起至2024年2月29日止，签署日未披露。"
    item = _item_for(
        text,
        sub_category="major_contract",
        requested_fields=["event_date"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "event_date",
            "status": "supported",
            "raw_value": "合同履行期间自2023年3月1日起",
            "normalized_value": "2023-03-01",
            "unit": "DATE",
            "evidence": [{"quote": "合同履行期间自2023年3月1日起", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"
    assert "performance-period date" in facts[0].abstention_reason


def test_unexecuted_market_reduction_plan_has_no_counterparty():
    text = "股东张三拟通过集中竞价或大宗交易实施减持计划。"
    item = _item_for(
        text,
        sub_category="equity_decrease",
        requested_fields=["counterparty"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "counterparty",
            "status": "supported",
            "raw_value": "股东张三",
            "normalized_value": "张三",
            "unit": "TEXT",
            "evidence": [{"quote": "张三", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "absent"
    assert "no identified buyer counterparty" in facts[0].abstention_reason


def test_generic_risk_section_headings_are_not_substantive_uncertainty():
    text = "五、风险提示：（一）项目实施风险；（二）项目运营风险。具体内容另见正文。"
    item = _item_for(
        text,
        sub_category="external_investment",
        requested_fields=["uncertainty"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "uncertainty",
            "status": "supported",
            "raw_value": "项目实施风险；项目运营风险",
            "normalized_value": "项目实施风险；项目运营风险",
            "unit": "TEXT",
            "evidence": [
                {"quote": "项目实施风险", "occurrence": 1},
                {"quote": "项目运营风险", "occurrence": 1},
            ],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"
    assert "not substantive uncertainty" in facts[0].abstention_reason


def test_bare_approval_stage_is_rejected():
    item = _item_for(
        "董事会批准子公司建设锂电项目。",
        sub_category="external_investment",
        requested_fields=["event_stage"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "event_stage",
            "status": "supported",
            "raw_value": "批准",
            "normalized_value": "批准",
            "unit": "TEXT",
            "evidence": [{"quote": "批准", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"
    assert "no identifiable object" in facts[0].abstention_reason


def test_retrospective_adjustment_is_canonicalized_to_boolean():
    text = "本次会计政策变更采用未来适用法，无需对财务报表进行追溯调整。"
    item = _item_for(
        text,
        sub_category="accounting_change",
        requested_fields=["retrospective_adjustment"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "retrospective_adjustment",
            "status": "supported",
            "raw_value": text,
            "normalized_value": text,
            "unit": "TEXT",
            "evidence": [{"quote": text, "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "false"
    assert facts[0].unit == "BOOLEAN"


def test_qualitative_profit_impact_is_not_numeric_zero():
    text = "本次会计政策变更事项对公司利润、总资产和净资产无重大影响。"
    item = _item_for(
        text,
        sub_category="accounting_change",
        requested_fields=["profit_effect_cny"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "profit_effect_cny",
            "status": "supported",
            "raw_value": text,
            "normalized_value": "0",
            "unit": "CNY",
            "evidence": [{"quote": text, "occurrence": 1}],
            "abstention_reason": None,
        }]},
        abstain_invalid_supported_fields=True,
    )
    assert facts[0].status == "absent"
    assert "not a disclosed CNY amount" in facts[0].abstention_reason


def test_accounting_change_scope_is_not_event_stage():
    text = "本次会计政策变更事项涉及计提安全生产费以及选择运用套期会计。"
    item = _item_for(
        text,
        sub_category="accounting_change",
        requested_fields=["event_stage"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "event_stage",
            "status": "supported",
            "raw_value": text,
            "normalized_value": text,
            "unit": "TEXT",
            "evidence": [{"quote": text, "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"
    assert "scope description" in facts[0].abstention_reason


def test_defined_alias_recovers_full_legal_entity_name():
    text = (
        "苏州绿脉电气控股（集团）有限公司（以下简称“苏州绿脉”）签署协议。"
        "本次权益变动后，公司控股股东变更为苏州绿脉。"
    )
    item = _item_for(
        text,
        sub_category="controller_change",
        requested_fields=["new_controller"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "new_controller",
            "status": "supported",
            "raw_value": "苏州绿脉",
            "normalized_value": "苏州绿脉电气控股（集团）有限公司",
            "unit": "TEXT",
            "evidence": [{"quote": "公司控股股东变更为苏州绿脉", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "苏州绿脉电气控股（集团）有限公司"
    assert len(facts[0].evidence) == 2


def test_bankruptcy_stage_reuses_validated_event_date_evidence():
    action = "2023年1月18日，公司获悉申请人向法院申请对公司进行重整，并申请启动预重整程序。"
    uncertainty = "目前公司尚未收到法院受理重整或预重整申请的司法文件。"
    item = _item_for(
        "背景说明：" + "甲" * 100 + "。" + action + uncertainty + "后续说明：" + "乙" * 100,
        sub_category="bankruptcy_restructuring",
        requested_fields=["event_stage", "event_date", "uncertainty", "case_stage"],
    )
    facts = validate_model_response(
        item,
        {"facts": [
            {
                "field_name": "event_stage",
                "status": "supported",
                "raw_value": "债权人申请重整",
                "normalized_value": "债权人申请重整",
                "unit": "TEXT",
                "evidence": [{"quote": "公司获悉债权人申请重整", "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "event_date",
                "status": "supported",
                "raw_value": action,
                "normalized_value": "2023-01-18",
                "unit": "DATE",
                "evidence": [{"quote": action, "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "uncertainty",
                "status": "supported",
                "raw_value": uncertainty,
                "normalized_value": uncertainty,
                "unit": "TEXT",
                "evidence": [{"quote": uncertainty, "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "case_stage",
                "status": "supported",
                "raw_value": "申请重整、法院尚未受理",
                "normalized_value": "申请重整、法院尚未受理",
                "unit": "TEXT",
                "evidence": [{"quote": "债权人申请重整且法院尚未受理", "occurrence": 1}],
                "abstention_reason": None,
            },
        ]},
        abstain_invalid_supported_fields=True,
    )
    by_name = {fact.field_name: fact for fact in facts}
    assert by_name["event_stage"].status == "supported", (
        by_name["event_stage"],
        by_name["event_date"],
    )
    assert by_name["case_stage"].status == "supported", by_name["case_stage"]
    assert by_name["event_stage"].evidence == by_name["event_date"].evidence
    assert len(by_name["case_stage"].evidence) == 2


def test_company_self_check_is_not_regulatory_inquiry():
    text = "公司对相关事项进行了核实，并与控股股东、实际控制人书面或通讯问询。"
    item = _item_for(
        text,
        sub_category="abnormal_volatility",
        requested_fields=["regulatory_inquiry"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "regulatory_inquiry",
            "status": "supported",
            "raw_value": text,
            "normalized_value": text,
            "unit": "TEXT",
            "evidence": [{"quote": text, "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "absent"
    assert "not regulatory inquiries" in facts[0].abstention_reason


def test_buyback_shareholder_record_date_is_not_disclosure_event_date():
    text = "现将前一个交易日(即2023年1月5日)登记在册的前十大股东名单公告如下。"
    item = _item_for(
        text,
        sub_category="buyback",
        requested_fields=["event_date"],
        title="关于回购股份事项前十大股东持股情况的公告",
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "event_date",
            "status": "supported",
            "raw_value": "前一个交易日(即2023年1月5日)登记在册",
            "normalized_value": "2023-01-05",
            "unit": "DATE",
            "evidence": [
                {"quote": "前一个交易日(即2023年1月5日)登记在册", "occurrence": 1}
            ],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"
    assert "shareholder-list record date" in facts[0].abstention_reason


def test_bid_stage_recovers_exact_source_action_after_bad_long_quote():
    exact_action = "公司及控股子公司均已收到上述项目的《中标通知书》"
    text = (
        "背景：" + "甲" * 100
        + "公司于2023年1月18日收到中标通知书。"
        + exact_action + "。后续：" + "乙" * 100
    )
    item = _item_for(
        text,
        sub_category="bid_win",
        requested_fields=["event_stage", "event_date"],
    )
    facts = validate_model_response(
        item,
        {"facts": [
            {
                "field_name": "event_stage",
                "status": "supported",
                "raw_value": "收到中标通知书",
                "normalized_value": "收到中标通知书",
                "unit": "TEXT",
                "evidence": [
                    {"quote": "公司及控股子公司已经收到中标通知书", "occurrence": 1}
                ],
                "abstention_reason": None,
            },
            {
                "field_name": "event_date",
                "status": "supported",
                "raw_value": "2023年1月18日收到中标通知书",
                "normalized_value": "2023-01-18",
                "unit": "DATE",
                "evidence": [
                    {"quote": "2023年1月18日收到中标通知书", "occurrence": 1}
                ],
                "abstention_reason": None,
            },
        ]},
        abstain_invalid_supported_fields=True,
    )
    by_name = {fact.field_name: fact for fact in facts}
    assert by_name["event_stage"].status == "supported"
    assert by_name["event_stage"].normalized_value == "收到中标通知书"
    assert by_name["event_stage"].evidence[0].quote == exact_action


def test_text_reason_recovers_only_unique_long_exact_prefix():
    source_reason = (
        "计提安全生产费主要依据财政部、应急部联合发布的《安全生产办法》"
        "（以下简称“管理办法”）要求执行。"
    )
    item = _item_for(
        source_reason,
        sub_category="accounting_change",
        requested_fields=["reason"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "reason",
            "status": "supported",
            "raw_value": "依据安全生产办法执行",
            "normalized_value": "依据安全生产办法执行",
            "unit": "TEXT",
            "evidence": [{
                "quote": (
                    "计提安全生产费主要依据财政部、应急部联合发布的《安全生产办法》"
                    "要求执行。"
                ),
                "occurrence": 1,
            }],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == (
        "计提安全生产费主要依据财政部、应急部联合发布的《安全生产办法》"
    )
    assert facts[0].evidence[0].quote == facts[0].normalized_value


def test_policy_future_effectiveness_recovers_current_formulation_stage():
    text = (
        "《募集资金管理制度》\n为规范募集资金管理，特制定本制\n度。\n"
        "本制度经股东大会审议通过后生效并实施。"
    )
    item = _item_for(
        text,
        sub_category="fundraising_usage",
        requested_fields=["event_stage"],
        title="《募集资金管理制度》",
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "event_stage",
            "status": "supported",
            "raw_value": "本制度经股东大会审议通过后生效并实施。",
            "normalized_value": "本制度经股东大会审议通过后生效并实施。",
            "unit": "TEXT",
            "evidence": [{
                "quote": "本制度经股东大会审议通过后生效并实施。",
                "occurrence": 1,
            }],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "制定募集资金管理制度"
    assert facts[0].evidence[0].quote == "特制定本制\n度"


def test_lockup_future_listing_details_recover_current_unlock_stage():
    current = "截至本公告披露日，义乌奇光所持剩余限售股份解锁条件已成就，现申请上市流通"
    future = "本次限售股上市流通数量为79,910,991股；上市流通日期为2023年1月13日"
    item = _item_for(
        current + "。" + future + "。",
        sub_category="lockup_expiry",
        requested_fields=["event_stage"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "event_stage",
            "status": "supported",
            "raw_value": future,
            "normalized_value": future,
            "unit": "TEXT",
            "evidence": [{"quote": future, "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert "解锁条件已成就" in facts[0].normalized_value
    assert "申请上市流通" in facts[0].normalized_value
    assert "2023年1月13日" not in facts[0].normalized_value


def test_contract_placeholder_has_no_concrete_period():
    text = "合同履行期限：按照合同规定期限内交货。"
    item = _item_for(
        text,
        sub_category="major_contract",
        requested_fields=["contract_period"],
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "contract_period",
            "status": "supported",
            "raw_value": "按照合同规定期限内交货",
            "normalized_value": "按照合同规定期限内交货",
            "unit": "TEXT",
            "evidence": [{"quote": "按照合同规定期限内交货", "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "ambiguous"
    assert "no concrete period" in facts[0].abstention_reason


def test_financial_report_uses_report_role_not_internal_approval():
    report_title = "积成电子股份有限公司 2023 年半年度报告"
    approval = "本财务报表业经本公司董事会于2023年8月21日决议批准报出。"
    item = _item_for(
        report_title + "全文\n" + approval,
        sub_category="earnings_h1",
        requested_fields=["event_stage", "event_date"],
        title="积成电子:2023年半年度报告",
    )
    assert infer_document_role(item.source.title) == "financial_report"
    facts = validate_model_response(
        item,
        {"facts": [
            {
                "field_name": "event_stage",
                "status": "supported",
                "raw_value": approval,
                "normalized_value": approval,
                "unit": "TEXT",
                "evidence": [{"quote": approval, "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "event_date",
                "status": "supported",
                "raw_value": "2023年8月21日",
                "normalized_value": "2023-08-21",
                "unit": "DATE",
                "evidence": [{"quote": "2023年8月21日", "occurrence": 1}],
                "abstention_reason": None,
            },
        ]},
    )
    by_name = {fact.field_name: fact for fact in facts}
    assert by_name["event_stage"].status == "supported"
    assert by_name["event_stage"].normalized_value == "披露2023年半年度报告"
    assert by_name["event_date"].status == "ambiguous"
    assert "approval date" in by_name["event_date"].abstention_reason


def test_treasury_product_and_institution_must_preserve_specificity():
    generic_product = "安全性较高、流动性较强、风险较低的投资产品"
    generic_institution = "银行、证券公司等金融机构"
    text = (
        generic_product + "。公司拟购买安全性高、流动性好、单项产品期限不超过12个月的"
        "短期理财产品，并可从" + generic_institution + "购买。"
    )
    item = _item_for(
        text,
        sub_category="treasury_management",
        requested_fields=["product_type", "financial_institution"],
    )
    facts = validate_model_response(
        item,
        {"facts": [
            {
                "field_name": "product_type",
                "status": "supported",
                "raw_value": generic_product,
                "normalized_value": generic_product,
                "unit": "TEXT",
                "evidence": [{"quote": generic_product, "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "financial_institution",
                "status": "supported",
                "raw_value": generic_institution,
                "normalized_value": generic_institution,
                "unit": "TEXT",
                "evidence": [{"quote": generic_institution, "occurrence": 1}],
                "abstention_reason": None,
            },
        ]},
    )
    by_name = {fact.field_name: fact for fact in facts}
    assert by_name["product_type"].status == "ambiguous"
    assert by_name["financial_institution"].status == "ambiguous"


def test_pdf_split_text_prefix_maps_back_to_exact_inquiry_evidence():
    source_stage = "申请文\n件进行了审核，并形成如下审核问询问题"
    source_reply = "请逐项落实并在十五个工作日内提交对问询函\n的回复"
    item = _item_for(
        source_stage + "。" + source_reply + "。",
        sub_category="private_placement",
        requested_fields=["event_stage", "approval_status"],
    )
    facts = validate_model_response(
        item,
        {"facts": [
            {
                "field_name": "event_stage",
                "status": "supported",
                "raw_value": "形成审核问询问题",
                "normalized_value": "形成审核问询问题",
                "unit": "TEXT",
                "evidence": [{
                    "quote": "申请文件进行了审核，并形成如下审核问询问题",
                    "occurrence": 1,
                }],
                "abstention_reason": None,
            },
            {
                "field_name": "approval_status",
                "status": "supported",
                "raw_value": "十五个工作日内提交回复",
                "normalized_value": "十五个工作日内提交回复",
                "unit": "TEXT",
                "evidence": [{
                    "quote": "请逐项落实并在十五个工作日内提交对问询函的回复",
                    "occurrence": 1,
                }],
                "abstention_reason": None,
            },
        ]},
    )
    by_name = {fact.field_name: fact for fact in facts}
    assert by_name["event_stage"].status == "supported"
    assert by_name["event_stage"].evidence[0].quote == source_stage
    assert by_name["approval_status"].status == "supported"
    assert by_name["approval_status"].evidence[0].quote == source_reply


def test_financial_table_numeric_evidence_recovers_split_rows():
    text = (
        "本报告期 上年同期 本报告期比上年同期增减\n"
        "营业收入（元） 779,609,020.27 756,304,868.14 3.08%\n"
        "归属于上市公司股东的净利\n润（元） -66,183,051.96 -37,583,867.72 -76.09%\n"
        "归属于上市公司股东的扣除\n非经常性损益的净利润\n（元）\n"
        "-68,844,764.07 -49,763,644.22 -38.34%"
    )
    item = _item_for(
        text,
        sub_category="earnings_h1",
        requested_fields=[
            "revenue_yoy_pct",
            "net_profit_yoy_pct",
            "non_gaap_profit_cny",
        ],
        title="公司:2023年半年度报告",
    )
    response_facts = []
    for field_name, raw_value, normalized_value, unit, quote in (
        (
            "revenue_yoy_pct",
            "3.08%",
            "3.08",
            "PCT",
            "本报告期比上年同期增减 3.08%",
        ),
        (
            "net_profit_yoy_pct",
            "-76.09%",
            "-76.09",
            "PCT",
            "本报告期比上年同期增减 -76.09%",
        ),
        (
            "non_gaap_profit_cny",
            "-68,844,764.07",
            "-68844764.07",
            "CNY",
            "归属于上市公司股东的扣除非经常性损益的净利润（元） -68,844,764.07",
        ),
    ):
        response_facts.append({
            "field_name": field_name,
            "status": "supported",
            "raw_value": raw_value,
            "normalized_value": normalized_value,
            "unit": unit,
            "evidence": [{"quote": quote, "occurrence": 1}],
            "abstention_reason": None,
        })
    facts = validate_model_response(item, {"facts": response_facts})
    assert all(fact.status == "supported" for fact in facts)
    assert {fact.field_name: fact.normalized_value for fact in facts} == {
        "revenue_yoy_pct": "3.08",
        "net_profit_yoy_pct": "-76.09",
        "non_gaap_profit_cny": "-68844764.07",
    }


def test_operating_data_report_title_is_a_valid_disclosure_stage():
    title = "京运通:2022年第四季度及全年新能源电站经营数据公告"
    quote = "2022 年第四季度及全年新能源电站经营数据公告"
    item = _item_for(
        quote,
        sub_category="operating_data",
        requested_fields=["event_stage"],
        title=title,
    )
    assert infer_document_role(title) == "operating_data_report"
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "event_stage",
            "status": "supported",
            "raw_value": "披露经营数据",
            "normalized_value": "披露经营数据",
            "unit": "TEXT",
            "evidence": [{"quote": quote, "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"


def test_st_progress_uses_current_remediation_not_future_monthly_disclosure():
    future = "公司将每月披露一次其他风险警示的进展情况"
    remediation = "公司被冻结银行账户已全部解封"
    item = _item_for(
        future + "。" + remediation + "。",
        sub_category="st_delisting_risk",
        requested_fields=["event_stage", "remediation_status"],
    )
    facts = validate_model_response(
        item,
        {"facts": [
            {
                "field_name": "event_stage",
                "status": "supported",
                "raw_value": future,
                "normalized_value": future,
                "unit": "TEXT",
                "evidence": [{"quote": future, "occurrence": 1}],
                "abstention_reason": None,
            },
            {
                "field_name": "remediation_status",
                "status": "supported",
                "raw_value": remediation,
                "normalized_value": remediation,
                "unit": "TEXT",
                "evidence": [{"quote": remediation, "occurrence": 1}],
                "abstention_reason": None,
            },
        ]},
    )
    by_name = {fact.field_name: fact for fact in facts}
    assert by_name["event_stage"].status == "supported"
    assert by_name["event_stage"].normalized_value == remediation


def test_lockup_notice_title_recovers_current_arrangement_without_condition_sentence():
    title = "公司:关于非公开发行部分限售股份上市流通的提示性公告"
    quote = "关于非公开发行部分限售股份上市流通的提示性公告"
    listing = "本次可解除限售股份上市流通时间为2023年1月20日"
    item = _item_for(
        quote + "。" + listing + "。",
        sub_category="lockup_expiry",
        requested_fields=["event_stage"],
        title=title,
    )
    facts = validate_model_response(
        item,
        {"facts": [{
            "field_name": "event_stage",
            "status": "supported",
            "raw_value": listing,
            "normalized_value": listing,
            "unit": "TEXT",
            "evidence": [{"quote": listing, "occurrence": 1}],
            "abstention_reason": None,
        }]},
    )
    assert facts[0].status == "supported"
    assert facts[0].normalized_value == "披露本次解除限售股份及上市流通安排"
    assert facts[0].evidence[0].quote == quote

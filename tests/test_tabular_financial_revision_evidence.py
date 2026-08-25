from __future__ import annotations

from datetime import date

from src.prediction.tabular.financial_revision_evidence import (
    FINANCIAL_REVISION_CONFIG_CONTRACT,
    _numeric_match,
    _period_terms,
    _title_score,
    _verification_title,
    load_financial_revision_evidence_contract,
)


def test_financial_revision_evidence_contract_is_train_only() -> None:
    contract = load_financial_revision_evidence_contract()
    assert FINANCIAL_REVISION_CONFIG_CONTRACT == "tabular-financial-revision-evidence-v1"
    assert contract.announcement_date_start == date(2023, 1, 1)
    assert contract.announcement_date_end == date(2024, 12, 31)
    assert len(contract.provider_documentation_urls) == 4
    assert contract.derived_metric_fields["cashflow"] == ("free_cashflow",)
    assert contract.excluded_announcement_ids["AN202412121641283020"] == (
        "pdf_unextractable_after_cdp_probe"
    )


def test_financial_pdf_verification_title_removes_revision_only_suffix() -> None:
    assert (
        _verification_title("科士达:2024年第一季度报告(更正后)")
        == "2024年第一季度报告"
    )
    assert (
        _verification_title("安靠智电:2022年年度报告(2024年4月修订)")
        == "2022年年度报告"
    )
    assert _verification_title("关于前期会计差错更正的公告") == "关于前期会计差错更正的公告"


def test_revision_title_scoring_prefers_corrected_full_report() -> None:
    end_date = date(2024, 3, 31)
    original = "科士达:2024年一季度报告"
    corrected = "科士达:2024年第一季度报告(更正后)"
    summary = "科士达:2024年第一季度报告摘要"
    assert _period_terms("1", end_date) == (
        "2024年第一季度报告",
        "2024年一季度报告",
    )
    assert _title_score(
        corrected, end_type="1", end_date=end_date, revision_index=1
    ) > _title_score(original, end_type="1", end_date=end_date, revision_index=1)
    assert _title_score(
        original, end_type="1", end_date=end_date, revision_index=0
    ) > _title_score(summary, end_type="1", end_date=end_date, revision_index=0)
    assert (
        _title_score(
            "泓淋电力:内部控制鉴证报告",
            end_type="4",
            end_date=date(2022, 12, 31),
            revision_index=0,
        )
        < 0
    )


def test_numeric_match_handles_signed_financial_table_values() -> None:
    content = "经营活动现金流量净额 -50,668,634.65 期末现金 1,134,455,000.00"
    assert _numeric_match(-50668634.65, content)
    assert _numeric_match(1134455000.0, content)
    assert _numeric_match(772966378.28, "营业收入 77296.64（万元）")
    assert not _numeric_match(-123.45, content)

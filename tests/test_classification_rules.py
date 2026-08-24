"""分类决策层的真实标题回归测试。"""

import pytest

from src.ml.classifier_wrapper import ClassificationResult
from src.pipeline.classification_rules import ClassificationDecisionEngine


def _model(sub: str, major: str = "F", confidence: float = 0.99, margin: float = 0.8):
    return ClassificationResult(
        major_category=major,
        sub_category=sub,
        confidence=confidence,
        margin=margin,
        model_qualified=True,
    )


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("博力威:广东博力威科技股份有限公司2026年半年度报告", "earnings_h1"),
        ("*ST天宜:2026年半年度业绩预告的自愿性披露公告", "forecast_performance"),
        ("中电电机:股东减持股份计划公告", "equity_decrease"),
        ("国电南瑞:关于以集中竞价交易方式回购股份方案的公告", "buyback"),
        ("晨丰科技:关于使用自有资金支付募投项目款项并以募集资金置换的公告", "fundraising_usage"),
        ("某公司:关于召开2026年第二次临时股东会的通知", "meeting_resolution"),
        ("和展能源:关于公司及子公司担保的进展公告", "guarantee"),
        ("金风科技:关于股东部分股份解除质押的公告", "share_pledge"),
        ("容百科技:关于2026年半年度计提资产减值准备的公告", "asset_impairment"),
        ("某公司:关于收到年度报告信息披露监管问询函的公告", "regulatory_action"),
        ("某公司:关于公司预重整债权申报通知的公告", "bankruptcy_restructuring"),
        ("某公司:2026年年度权益分派实施公告", "dividend"),
        ("某公司:关于完成工商变更登记并换发营业执照的公告", "corporate_admin"),
        ("某公司:H股公告-证券变动月报表", "h_share_routine"),
        ("某公司:关于股票回购专项贷款承诺函的公告", "buyback"),
        ("某公司:关于董事长提议回购公司股份的公告", "buyback"),
        ("某公司:关于对全资孙公司增资的公告", "external_investment"),
        ("某公司:关于子公司增资扩股的公告", "external_investment"),
        ("某公司:关于申请注册发行债务融资工具的公告", "convertible_bond"),
        ("某公司:关于绿色科技创新债券发行完成的公告", "convertible_bond"),
        ("某公司:关于重大算力销售合同的进展公告", "major_contract"),
        ("某公司:关于子公司技改项目的进展公告", "project_progress"),
        ("某公司:2026年1-6月经营数据公告", "operating_data"),
        ("某公司:关于股东股份将被司法拍卖的提示性公告", "judicial_equity_action"),
        ("某公司:关于补缴税款相关事项的公告", "tax_matter"),
        ("某公司:关于更换持续督导保荐代表人的公告", "corporate_admin"),
        ("某公司:关于拟购买董责险的公告", "routine_disclosure"),
        ("某公司:关于终止委托管理协议的公告", "entrusted_management"),
        ("某公司:关于厂房租赁暨关联交易的公告", "related_transaction"),
        ("某公司:关于控股股东股份变动相关的公告", "equity_change"),
        ("某公司:关于转让全资孙公司100%股权的公告", "asset_sale"),
    ],
)
def test_high_precision_rules_override_wrong_model(title, expected):
    engine = ClassificationDecisionEngine(model_only_requires_review=False)
    decision = engine.decide(title, _model("st_delisting_risk"))

    assert decision.sub_category == expected
    assert decision.classification_source == "rule"
    assert decision.model_sub_category == "st_delisting_risk"
    assert decision.review_reason is not None


def test_document_type_is_independent_from_primary_event():
    engine = ClassificationDecisionEngine(model_only_requires_review=False)
    title = "律师事务所关于公司回购注销部分限制性股票的法律意见书"

    decision = engine.decide(title, _model("litigation"))

    assert decision.sub_category == "buyback_cancel"
    assert decision.document_type == "legal_opinion"
    assert "equity_incentive" in decision.secondary_tags


def test_routine_disclosure_has_safe_exit():
    engine = ClassificationDecisionEngine(model_only_requires_review=False)
    decision = engine.decide("某公司:关于变更办公地址的公告", _model("controller_change", "G"))

    assert decision.major_category == "X"
    assert decision.sub_category == "routine_disclosure"
    assert decision.relevance == "routine_disclosure"


def test_low_confidence_model_abstains():
    engine = ClassificationDecisionEngine(
        accept_confidence=0.85,
        min_margin=0.20,
        model_only_requires_review=False,
    )
    decision = engine.decide(
        "某公司:补充说明公告",
        _model("merger_acquisition", "D", confidence=0.51, margin=0.05),
    )

    assert decision.major_category == "X"
    assert decision.sub_category == "other"
    assert decision.classification_source == "abstained"
    assert decision.needs_review is True
    assert decision.model_sub_category == "merger_acquisition"


def test_model_only_result_requires_review_by_default():
    engine = ClassificationDecisionEngine(
        accept_confidence=0.85,
        min_margin=0.20,
        model_only_requires_review=True,
    )
    decision = engine.decide(
        "某公司:一项无法被标题规则覆盖的新事项",
        _model("major_contract", "E", confidence=0.96, margin=0.60),
    )

    assert decision.sub_category == "major_contract"
    assert decision.classification_source == "model"
    assert decision.needs_review is True


def test_unqualified_model_always_abstains():
    engine = ClassificationDecisionEngine(model_only_requires_review=False)
    model = _model("major_contract", "E", confidence=0.999, margin=0.90)
    model.model_qualified = False

    decision = engine.decide("某公司:补充说明公告", model)

    assert decision.major_category == "X"
    assert decision.sub_category == "other"
    assert decision.classification_source == "abstained"
    assert decision.needs_review is True
    assert "未通过真实数据" in decision.review_reason

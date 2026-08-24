"""公告分类决策层。

标题高精度规则负责可确定的公告，BERT 只处理剩余样本。决策层同时输出
文档类型、相关性、辅助标签和复核状态，避免把不同维度继续挤进单一子类。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import config
from src.ml.classifier_wrapper import ClassificationResult


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    major_category: str
    sub_category: str
    pattern: str
    confidence: float = 0.995
    relevance: str = "core_event"
    exclusions: tuple[str, ...] = ()

    def matches(self, title: str) -> bool:
        return bool(re.search(self.pattern, title, re.IGNORECASE)) and not any(
            re.search(exclusion, title, re.IGNORECASE) for exclusion in self.exclusions
        )


@dataclass(frozen=True)
class ClassificationDecision:
    major_category: str
    sub_category: str
    confidence: float
    classification_source: str
    document_type: str
    relevance: str
    secondary_tags: list[str] = field(default_factory=list)
    rule_id: str | None = None
    needs_review: bool = False
    review_reason: str | None = None
    model_sub_category: str | None = None
    model_confidence: float | None = None
    model_margin: float | None = None


# 顺序即优先级：修正先于预告、回购注销先于普通回购、定增先于募集资金用途。
TITLE_RULES: tuple[RuleSpec, ...] = (
    RuleSpec(
        "forecast_revision",
        "B",
        "forecast_revision",
        r"业绩预告.*(?:修正|更正)|(?:修正|更正).*业绩预告",
    ),
    RuleSpec("forecast_express", "B", "forecast_express", r"业绩快报"),
    RuleSpec(
        "forecast_performance", "B", "forecast_performance", r"业绩预告", exclusions=(r"修正|更正",)
    ),
    RuleSpec(
        "earnings_q1", "A", "earnings_q1", r"第一季度报告(?:摘要|全文)?(?:\([^)]*\)|（[^）]*）)?$"
    ),
    RuleSpec(
        "earnings_h1",
        "A",
        "earnings_h1",
        r"半年度报告(?:摘要|全文)?(?:\([^)]*\)|（[^）]*）)?$",
        exclusions=(r"业绩预告|募集资金|非经营性资金|审计报告",),
    ),
    RuleSpec(
        "earnings_q3", "A", "earnings_q3", r"第三季度报告(?:摘要|全文)?(?:\([^)]*\)|（[^）]*）)?$"
    ),
    RuleSpec(
        "earnings_annual",
        "A",
        "earnings_annual",
        r"20\d{2}年年度报告(?:摘要|全文)?(?:\([^)]*\)|（[^）]*）)?$",
    ),
    RuleSpec("operating_data", "A", "operating_data", r"(?:半年度|1-6月|年度|季度).{0,8}经营数据"),
    RuleSpec(
        "no_reduce_commitment",
        "C",
        "no_reduce_commitment",
        r"(?:承诺|自愿).{0,18}不减持|不减持.{0,18}(?:承诺|公告)",
    ),
    RuleSpec(
        "buyback_cancel",
        "C",
        "buyback_cancel",
        r"回购.{0,4}注销.{0,30}限制性股票|限制性股票.{0,30}回购.{0,4}注销",
    ),
    RuleSpec(
        "share_pledge",
        "C",
        "share_pledge",
        r"(?:股东|实际控制人|控股股东).{0,25}(?:股份|股权).{0,12}(?:解除|新增|补充|再)?质押|"
        r"(?:股份|股权).{0,12}(?:解除|新增|补充|再)?质押",
    ),
    RuleSpec(
        "equity_change",
        "C",
        "equity_change",
        r"权益变动报告书|权益变动.{0,18}(?:触及|达到).{0,8}(?:1%|5%|整数倍|刻度)|"
        r"持有股份变动触及|股东询价转让|一致行动人之间内部转让股份|"
        r"控股股东.{0,30}(?:上层)?股权结构.{0,12}(?:变动|变更)|"
        r"控股股东.{0,20}签署.{0,12}增资协议|暨权益变动|控股股东股份变动相关",
    ),
    RuleSpec(
        "dividend",
        "C",
        "dividend",
        r"(?:年度|半年度|季度|中期)?.{0,8}(?:利润分配|权益分派|分红|分红派息)"
        r".{0,8}(?:方案|预案|实施|公告)|"
        r"(?:董事长|控股股东).{0,16}(?:提议|分红提议).{0,12}(?:分红|利润分配)?",
        exclusions=(r"回购价格|调整.{0,12}回购|提议回购|回购公司股份",),
    ),
    RuleSpec(
        "lockup_expiry",
        "C",
        "lockup_expiry",
        r"(?:限售股|限售股份|解除限售股份).{0,30}(?:上市流通|解禁)|"
        r"首次公开发行前已发行股份.{0,30}上市流通",
    ),
    RuleSpec(
        "equity_decrease",
        "C",
        "equity_decrease",
        r"(?:股东|董事|高管|实际控制人).{0,20}减持|减持股份",
    ),
    RuleSpec(
        "equity_increase",
        "C",
        "equity_increase",
        r"(?:股东|董事|高管|实际控制人).{0,20}增持|增持股份",
        exclusions=(r"不增持",),
    ),
    RuleSpec(
        "buyback",
        "C",
        "buyback",
        r"回购(?:公司)?股份|股份回购|回购报告书|回购.{0,8}贷款承诺|"
        r"提议回购|回购事项前十大股东",
        exclusions=(r"回购注销|限制性股票",),
    ),
    RuleSpec(
        "equity_incentive",
        "C",
        "equity_incentive",
        r"股权激励|限制性股票激励|股票期权激励|员工持股计划|"
        r"股票期权.{0,16}(?:行权价格|注销)|注销部分股票期权",
    ),
    RuleSpec(
        "private_placement",
        "D",
        "private_placement",
        r"向特定对象发行|非公开发行|定向增发|特定对象.{0,20}股份认购协议",
    ),
    RuleSpec("fundraising_usage", "D", "fundraising_usage", r"募集资金|募投项目"),
    RuleSpec(
        "convertible_bond",
        "D",
        "convertible_bond",
        r"可转换公司债券|可转债|发行公司债券|转债|转股价格|"
        r"(?:债务融资工具|中期票据|公司债券).{0,20}(?:注册|发行|完成|结果)|"
        r"(?:申请)?注册发行.{0,20}(?:债务融资工具|债券)|"
        r"交易商协会.{0,12}接受注册通知书|债券.{0,12}发行完成",
        exclusions=(r"管理办法|管理制度",),
    ),
    RuleSpec(
        "merger_acquisition",
        "D",
        "merger_acquisition",
        r"重大资产重组|发行股份.{0,12}购买资产|并购|收购.{0,20}股权|"
        r"(?:受让|购买).{0,30}股权|吸收合并|战略性重组",
    ),
    RuleSpec(
        "asset_sale_exact",
        "D",
        "asset_sale",
        r"签署.{0,20}股权转让协议|转让.{0,20}100%股权|处置部分资产",
    ),
    RuleSpec(
        "asset_sale",
        "D",
        "asset_sale",
        r"出售.{0,20}资产|转让.{0,20}(?:资产|股权)|处置.{0,12}资产",
    ),
    RuleSpec(
        "guarantee",
        "D",
        "guarantee",
        r"(?:提供|对外|融资|授信|银行贷款).{0,25}担保|"
        r"(?:公司|子公司).{0,12}担保.{0,12}(?:进展|公告)|"
        r"担保(?:额度)?(?:调剂|预计|进展)|关于担保的进展",
        exclusions=(r"核查意见|独立意见|法律意见|管理制度",),
    ),
    RuleSpec("financial_assistance", "D", "financial_assistance", r"财务资助"),
    RuleSpec(
        "hedging_derivatives",
        "D",
        "hedging_derivatives",
        r"套期保值|金融衍生品交易|外汇衍生品交易",
        exclusions=(r"核查意见|可行性分析报告|管理制度",),
    ),
    RuleSpec(
        "financing_credit",
        "D",
        "financing_credit",
        r"申请.{0,12}(?:综合)?授信额度|向金融机构.{0,12}(?:授信|融资)|"
        r"向银行申请授信|应收账款.{0,12}保理业务",
        exclusions=(r"提供担保|担保额度",),
    ),
    RuleSpec(
        "external_investment",
        "D",
        "external_investment",
        r"对外投资|投资设立.{0,12}(?:子公司|公司|合伙企业|产业基金)|"
        r"设立.{0,8}(?:全资|控股)?(?:子公司|孙公司)|"
        r"与专业投资机构共同投资|共同投资|对.{0,6}(?:子公司|孙公司).{0,40}增资|"
        r"向.{0,6}(?:子公司|孙公司|参股公司).{0,40}(?:增加投资|增资)|"
        r"(?:子公司|孙公司).{0,40}增资扩股|"
        r"(?:参与|放弃).{0,20}(?:子公司|孙公司|参股公司).{0,20}(?:增资|优先认购权)|"
        r"(?:合作设立|认购|参与投资|合作投资).{0,20}(?:投资基金|产业基金)|"
        r"私募基金.{0,12}合作投资|投资基金.{0,12}(?:减资|进展)|"
        r"放弃.{0,20}股权转让优先购买权",
        exclusions=(r"募集资金|重大资产重组|发行股份购买资产|管理制度",),
    ),
    RuleSpec(
        "related_transaction",
        "D",
        "related_transaction",
        r"(?:日常)?关联交易(?:预计|公告)|关于关联交易|形成关联交易|暨关联交易",
        exclusions=(r"担保|财务资助|对外投资|共同投资|购买资产|出售资产|管理制度",),
    ),
    RuleSpec(
        "equity_financing",
        "D",
        "equity_financing",
        r"再融资注册|发行H股股票|H股发行并上市|递交H股.{0,8}上市申请",
        exclusions=(r"管理制度|实施细则|工作细则|政策",),
    ),
    RuleSpec(
        "treasury_management",
        "D",
        "treasury_management",
        r"委托理财|购买金融机构理财产品|闲置.{0,8}资金.{0,8}现金管理|"
        r"开展现金管理|开立现金管理专用结算账户",
    ),
    RuleSpec(
        "entrusted_management",
        "D",
        "entrusted_management",
        r"(?:终止|签订|签署).{0,12}委托管理协议",
    ),
    RuleSpec("bid_win", "E", "bid_win", r"中标|中选(?:候选人)?"),
    RuleSpec(
        "major_contract_exact",
        "E",
        "major_contract",
        r"重大.{0,8}合同.{0,8}进展|获得客户项目定点通知书",
    ),
    RuleSpec(
        "major_contract",
        "E",
        "major_contract",
        r"签(?:订|署).{0,20}(?:重大)?合同|重大合同|重大.{0,8}合同.{0,8}进展|"
        r"获得客户订单|项目定点通知书",
    ),
    RuleSpec(
        "product_approval", "E", "product_approval", r"获得.{0,15}(?:注册证|批准|批件)|产品获批"
    ),
    RuleSpec("project_commissioning", "E", "project_commissioning", r"投产|竣工|产能建设"),
    RuleSpec(
        "project_investment",
        "E",
        "project_investment",
        r"投资建设|投资.{0,16}(?:生产基地|产业园|项目)|"
        r"扩建产能|竞拍土地使用权",
    ),
    RuleSpec("project_progress", "E", "project_progress", r"(?:技改|单晶|建设)项目.{0,12}进展"),
    RuleSpec("strategic_cooperation", "E", "strategic_cooperation", r"战略合作|合作协议"),
    RuleSpec(
        "government_subsidy",
        "E",
        "government_subsidy",
        r"获得政府补助|收到政府补助|收到退税款|退还增值税留抵税额",
    ),
    RuleSpec(
        "intellectual_property",
        "E",
        "intellectual_property",
        r"取得.{0,8}(?:专利证书|软件著作权)|获得.{0,8}专利证书",
    ),
    RuleSpec("litigation", "F", "litigation", r"诉讼|仲裁"),
    RuleSpec("penalty", "F", "penalty", r"行政处罚|处罚决定书|纪律处分|通报批评"),
    RuleSpec("investigation", "F", "investigation", r"立案调查|被立案|接受调查"),
    RuleSpec("debt_default", "F", "debt_default", r"债务逾期|未能清偿|债券违约"),
    RuleSpec(
        "st_delisting_risk",
        "F",
        "st_delisting_risk",
        r"退市风险警示|其他风险警示|撤销.{0,8}风险警示|实施风险警示.{0,12}进展",
    ),
    RuleSpec(
        "bankruptcy_restructuring",
        "F",
        "bankruptcy_restructuring",
        r"(?:破产|预)?重整|破产清算|债权申报|指定临时管理人",
    ),
    RuleSpec(
        "asset_impairment",
        "F",
        "asset_impairment",
        r"计提.{0,12}资产减值|计提减值准备|资产减值准备|资产减值测试",
    ),
    RuleSpec(
        "regulatory_action",
        "F",
        "regulatory_action",
        r"监管问询函|审核问询函|年度报告.{0,12}问询函|监管工作函|监管函|行政监管措施|警示函|"
        r"纪律处分|通报批评",
        exclusions=(r"公司制度|不存在被证券监管",),
    ),
    RuleSpec(
        "trading_risk_warning",
        "F",
        "trading_risk_warning",
        r"股票交易风险提示|公司股价风险提示|交易风险提示公告",
    ),
    RuleSpec(
        "judicial_equity_action",
        "F",
        "judicial_equity_action",
        r"(?:股份|股权).{0,16}(?:司法冻结|司法拍卖)|(?:司法冻结|司法拍卖).{0,16}(?:股份|股权)",
    ),
    RuleSpec("tax_matter", "F", "tax_matter", r"补缴税款|追缴税款|税务处罚"),
    RuleSpec(
        "controller_change",
        "G",
        "controller_change",
        r"实际控制人.{0,12}(?:变更|变化)|控制权.{0,12}(?:变更|变化)",
    ),
    RuleSpec(
        "executive_change",
        "G",
        "executive_change",
        r"(?:董事|监事|高级管理人员|总经理|董事长).{0,15}(?:辞职|离任|聘任|选举)|"
        r"(?:补选|改选|增选|更换).{0,8}(?:董事|监事)|"
        r"变更(?:财务负责人|财务总监|董事会秘书)|"
        r"聘任.{0,8}(?:副总经理|财务总监|证券事务代表)|证券事务代表辞职|"
        r"职工代表董事|选举职工董事|高管变动|董事会秘书.{0,8}变更|董事会延期换届",
    ),
    RuleSpec(
        "trading_halt_resume",
        "G",
        "trading_halt_resume",
        r"(?:股票|证券).{0,8}(?:停牌|复牌)|停牌公告|复牌公告",
    ),
    RuleSpec(
        "abnormal_volatility",
        "G",
        "abnormal_volatility",
        r"交易异常波动|严重异常波动|股价异常波动|公司股价异动",
    ),
    RuleSpec(
        "investor_relations",
        "G",
        "investor_relations",
        r"投资者关系活动记录|投资者关系管理信息|投资者调研|业绩说明会|"
        r"投资者交流活动记录|投资者沟通纪要",
        relevance="routine_disclosure",
    ),
    RuleSpec(
        "accounting_change",
        "G",
        "accounting_change",
        r"会计政策变更|会计估计变更|变更记账本位币|"
        r"采用中国企业会计准则",
    ),
    RuleSpec(
        "auditor_change",
        "G",
        "auditor_change",
        r"变更会计师事务所|续聘会计师事务所|续聘.{0,30}(?:会计师事务所|审计机构)|"
        r"聘请.{0,12}审计机构|选聘会计师事务所",
    ),
    RuleSpec(
        "legal_opinion",
        "D",
        "legal_opinion",
        r"法律意见书|专项核查意见|保荐机构.{0,30}核查意见",
        relevance="supporting_document",
    ),
    RuleSpec(
        "meeting_resolution",
        "G",
        "meeting_resolution",
        r"(?:董事会|监事会|股东会|股东大会|职工代表大会|独立董事.{0,12}会议).{0,25}"
        r"(?:决议(?:的)?公告|(?:会议)?资料|会议决议)|"
        r"召开.{0,25}(?:股东会|股东大会).{0,12}(?:通知|提示性公告)",
        relevance="supporting_document",
    ),
    RuleSpec(
        "routine_policy",
        "X",
        "routine_disclosure",
        r"公司章程|议事规则|管理制度|工作制度|实施细则|工作细则|"
        r"登记备案制度|内部审核制度|独立董事.{0,12}声明|"
        r"管理办法|工作规则|多元化政策|修订公司制度|"
        r"股东回报规划|质量回报双提升|提质增效重回报|"
        r"非经营性资金占用及其他关联资金往来情况汇总表|"
        r"办公地址.{0,8}变更|变更.{0,8}办公地址|"
        r"(?:关联交易|会计师事务所选聘|内部审计|董事会秘书).{0,8}(?:制度|规则)|"
        r"(?:董事会秘书|独立董事).{0,20}(?:培训证明|资格证书|培训并取得)|"
        r"(?:购买|拟购买).{0,12}(?:(?:董事|高级管理人员).{0,12}责任险|董责险)|"
        r"半年度报告披露.{0,8}提示性公告|"
        r"最近五年.{0,25}(?:证券监管部门|交易所).{0,20}(?:监管措施|处罚)",
        relevance="routine_disclosure",
    ),
    RuleSpec(
        "h_share_routine",
        "X",
        "h_share_routine",
        r"H股.{0,4}公告|证券变动月报表|翌日披露报表|暂停过户登记期间",
        relevance="routine_disclosure",
    ),
    RuleSpec(
        "corporate_admin",
        "X",
        "corporate_admin",
        r"完成工商变更登记|换发营业执照|变更营业执照|"
        r"注销.{0,10}(?:全资|控股|参股)?(?:子公司|公司)|设立.{0,8}分公司|"
        r"减少注册资本|"
        r"(?:变更|更换)持续督导保荐代表人|财务顾问主办人",
        relevance="routine_disclosure",
    ),
    RuleSpec(
        "supporting_document",
        "X",
        "supporting_document",
        r"资产评估报告|审计报告|专项审核报告|独立财务顾问报告|"
        r"评估机构.{0,40}(?:公允性|合理性).{0,12}说明|"
        r"董事会.{0,20}本次交易.{0,30}审核条件.{0,12}说明|"
        r"独立意见|审核意见|审查意见|见证意见|法律意见(?:书)?|"
        r"核查意见|核查报告|发行保荐书|上市保荐书|募集说明书|跟踪评级报告|"
        r"可行性分析报告|(?:保荐机构|会计师事务所|证券公司).{0,40}"
        r"(?:核查意见|专项说明|回复)",
        relevance="supporting_document",
    ),
)


DOCUMENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("legal_opinion", r"法律意见书"),
    ("verification_opinion", r"核查意见|鉴证报告"),
    ("audit_report", r"审计报告|审核报告"),
    ("valuation_report", r"资产评估报告"),
    ("regulatory_document", r"监管问询函|审核问询函|监管工作函|监管函|警示函"),
    ("h_share_filing", r"H股公告|证券变动月报表|翌日披露报表"),
    ("periodic_report", r"(?:第一季度|半年度|第三季度|年度)报告"),
    ("performance_forecast", r"业绩预告|业绩快报"),
    ("meeting_resolution", r"(?:董事会|监事会|股东会|股东大会).{0,25}决议"),
    ("meeting_notice", r"召开.{0,20}(?:股东会|股东大会).{0,12}通知"),
    ("meeting_material", r"(?:股东会|股东大会).{0,12}会议资料"),
    ("investor_relations", r"投资者关系活动记录|投资者调研|业绩说明会"),
    ("policy_document", r"公司章程|议事规则|管理制度|声明与承诺"),
)


SECONDARY_TAG_PATTERNS: tuple[tuple[str, str], ...] = (
    ("fundraising", r"募集资金|募投项目"),
    ("related_transaction", r"关联交易"),
    ("equity_incentive", r"股权激励|限制性股票|股票期权"),
    ("dividend", r"利润分配|分红|权益分派"),
    ("guarantee", r"担保"),
    ("financial_assistance", r"财务资助"),
    ("share_change", r"股份变动|股本变动"),
)

# 这些规则的标题语义相对宽，若与模型冲突则保留人工复核；其余精确规则
# 自动接纳，但仍在 review_reason 中记录冲突，便于审计。
REVIEW_ON_CONFLICT_RULES = {
    "asset_sale",
    "major_contract",
    "product_approval",
    "project_commissioning",
    "strategic_cooperation",
    "executive_change",
}


def detect_document_type(title: str) -> str:
    for document_type, pattern in DOCUMENT_PATTERNS:
        if re.search(pattern, title, re.IGNORECASE):
            return document_type
    return "announcement"


def detect_secondary_tags(title: str) -> list[str]:
    return [
        tag for tag, pattern in SECONDARY_TAG_PATTERNS if re.search(pattern, title, re.IGNORECASE)
    ]


def default_relevance(major_category: str, sub_category: str) -> str:
    if major_category == "X":
        if sub_category in {"routine_disclosure", "corporate_admin", "h_share_routine"}:
            return "routine_disclosure"
        if sub_category == "supporting_document":
            return "supporting_document"
        return "uncertain"
    if sub_category in {"legal_opinion", "meeting_resolution"}:
        return "supporting_document"
    if sub_category == "investor_relations":
        return "routine_disclosure"
    return "core_event"


class ClassificationDecisionEngine:
    """合并标题规则与模型结果，并在不确定时主动拒绝分类。"""

    def __init__(
        self,
        accept_confidence: float | None = None,
        min_margin: float | None = None,
        model_only_requires_review: bool | None = None,
    ) -> None:
        self.accept_confidence = (
            accept_confidence
            if accept_confidence is not None
            else config.pipeline.classifier_accept_confidence
        )
        self.min_margin = (
            min_margin if min_margin is not None else config.pipeline.classifier_min_margin
        )
        self.model_only_requires_review = (
            model_only_requires_review
            if model_only_requires_review is not None
            else config.pipeline.model_only_requires_review
        )

    def decide_rule(self, title: str) -> ClassificationDecision | None:
        """只运行可解释标题规则；未命中时返回 None，不触发模型。"""
        normalized = (title or "").strip()
        match = next((rule for rule in TITLE_RULES if rule.matches(normalized)), None)
        if match is None:
            return None

        document_type = detect_document_type(normalized)
        supporting_types = {
            "legal_opinion",
            "verification_opinion",
            "audit_report",
            "valuation_report",
        }
        relevance = (
            "supporting_document" if document_type in supporting_types else match.relevance
        )
        return ClassificationDecision(
            major_category=match.major_category,
            sub_category=match.sub_category,
            confidence=match.confidence,
            classification_source="rule",
            document_type=document_type,
            relevance=relevance,
            secondary_tags=detect_secondary_tags(normalized),
            rule_id=match.rule_id,
        )

    def decide(self, title: str, model: ClassificationResult) -> ClassificationDecision:
        normalized = (title or "").strip()
        document_type = detect_document_type(normalized)
        tags = detect_secondary_tags(normalized)
        match = next((rule for rule in TITLE_RULES if rule.matches(normalized)), None)

        model_margin = model.margin
        if match is not None:
            conflict = model.sub_category != match.sub_category
            supporting_types = {
                "legal_opinion",
                "verification_opinion",
                "audit_report",
                "valuation_report",
            }
            needs_review = (
                conflict
                and match.rule_id in REVIEW_ON_CONFLICT_RULES
                and document_type not in supporting_types
            )
            relevance = (
                "supporting_document"
                if document_type in supporting_types
                else match.relevance
            )
            return ClassificationDecision(
                major_category=match.major_category,
                sub_category=match.sub_category,
                confidence=match.confidence,
                classification_source=("rule+model" if not conflict else "rule"),
                document_type=document_type,
                relevance=relevance,
                secondary_tags=tags,
                rule_id=match.rule_id,
                needs_review=needs_review,
                review_reason=(f"规则与模型冲突：模型={model.sub_category}" if conflict else None),
                model_sub_category=model.sub_category,
                model_confidence=model.confidence,
                model_margin=model_margin,
            )

        if not model.model_qualified:
            return ClassificationDecision(
                major_category="X",
                sub_category="other",
                confidence=0.0,
                classification_source="abstained",
                document_type=document_type,
                relevance="uncertain",
                secondary_tags=tags,
                needs_review=True,
                review_reason="模型未通过真实数据时间外验证，仅保留候选建议",
                model_sub_category=model.sub_category,
                model_confidence=model.confidence,
                model_margin=model_margin,
            )

        insufficient_confidence = model.confidence < self.accept_confidence
        insufficient_margin = model_margin is not None and model_margin < self.min_margin
        if insufficient_confidence or insufficient_margin:
            reasons = []
            if insufficient_confidence:
                reasons.append(
                    f"模型置信度 {model.confidence:.3f} 低于 {self.accept_confidence:.3f}"
                )
            if insufficient_margin:
                reasons.append(f"前两名差值 {model_margin:.3f} 低于 {self.min_margin:.3f}")
            return ClassificationDecision(
                major_category="X",
                sub_category="other",
                confidence=0.0,
                classification_source="abstained",
                document_type=document_type,
                relevance="uncertain",
                secondary_tags=tags,
                needs_review=True,
                review_reason="；".join(reasons),
                model_sub_category=model.sub_category,
                model_confidence=model.confidence,
                model_margin=model_margin,
            )

        return ClassificationDecision(
            major_category=model.major_category,
            sub_category=model.sub_category,
            confidence=model.confidence,
            classification_source="model",
            document_type=document_type,
            relevance=default_relevance(model.major_category, model.sub_category),
            secondary_tags=tags,
            needs_review=self.model_only_requires_review,
            review_reason=(
                "真实公告模型尚未完成独立验证" if self.model_only_requires_review else None
            ),
            model_sub_category=model.sub_category,
            model_confidence=model.confidence,
            model_margin=model_margin,
        )

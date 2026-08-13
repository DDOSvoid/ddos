"""展示用中文标签映射 — 纯函数，不访问数据库。

将分类代码 / 提取字段英文名 / 单位代码翻译为界面上的中文标签。
字段清单覆盖 config/event_types.yaml 全部 subcategories 的 fields。
"""

from src.config import event_registry

# ── 提取字段 → 中文 ────────────────────────────────────────────
# 覆盖 event_types.yaml 中 A-G 各子类定义的 fields + 通用提取字段。

FIELD_LABELS: dict[str, str] = {
    # A 财报
    "revenue": "营业收入",
    "revenue_yoy_pct": "营收同比",
    "net_profit": "净利润",
    "net_profit_yoy_pct": "净利润同比",
    "gross_margin": "毛利率",
    "net_margin": "净利率",
    "eps": "每股收益",
    "roe": "净资产收益率",
    "operating_cashflow": "经营性现金流",
    "dividend_plan": "分红方案",
    "audit_opinion": "审计意见",
    # B 财报前置
    "forecast_type": "预告类型",
    "forecast_net_profit_min": "预告净利润下限",
    "forecast_net_profit_max": "预告净利润上限",
    "forecast_yoy_change_pct": "预告同比变动",
    "previous_net_profit": "上年同期净利润",
    "reason_summary": "变动原因",
    "total_assets": "总资产",
    "net_assets": "净资产",
    "original_forecast": "原预告值",
    "revised_forecast": "修正后预告值",
    "revision_amount": "修正金额",
    "revision_pct": "修正幅度",
    "revision_reason": "修正原因",
    # C 股权变动
    "shareholder_name": "股东名称",
    "shareholder_type": "股东类型",
    "increase_amount": "增持数量",
    "increase_ratio_pct": "增持比例",
    "price_range": "价格区间",
    "increase_reason": "增持原因",
    "planned_continue": "是否计划继续增持",
    "decrease_amount": "减持数量",
    "decrease_ratio_pct": "减持比例",
    "decrease_reason": "减持原因",
    "is_completed": "是否完成",
    "buyback_amount_max": "回购金额上限",
    "buyback_ratio_pct": "回购比例",
    "buyback_price_max": "回购价格上限",
    "buyback_purpose": "回购目的",
    "funding_source": "资金来源",
    "buyback_period": "回购期限",
    "unlock_shares": "解禁股数",
    "unlock_ratio_pct": "解禁比例",
    "unlock_market_value": "解禁市值",
    "unlock_date": "解禁日期",
    "shareholder_list": "解禁股东名单",
    "incentive_shares": "激励股数",
    "incentive_ratio_pct": "激励比例",
    "grant_price": "授予价格",
    "performance_targets": "业绩考核目标",
    "recipient_count": "激励人数",
    "vesting_schedule": "行权安排",
    # D 资本运作
    "placement_amount": "募资金额",
    "placement_shares": "发行股数",
    "placement_price": "发行价格",
    "placement_purpose": "募资用途",
    "subscribers": "认购对象",
    "lockup_period": "锁定期",
    "approval_status": "审批状态",
    "target_name": "标的企业",
    "transaction_amount": "交易金额",
    "payment_method": "支付方式",
    "target_industry": "标的行业",
    "target_revenue": "标的营收",
    "target_net_profit": "标的净利润",
    "deal_status": "交易进展",
    "performance_commitment": "业绩承诺",
    "asset_name": "标的资产",
    "sale_amount": "出售金额",
    "book_value": "账面价值",
    "gain_loss_amount": "转让损益",
    "buyer": "受让方",
    "sale_reason": "出售原因",
    "bond_amount": "债券规模",
    "bond_term": "债券期限",
    "coupon_rate": "票面利率",
    "conversion_price": "转股价",
    "use_of_proceeds": "募资用途",
    "credit_rating": "信用评级",
    # E 经营业务
    "contract_amount": "合同金额",
    "counterparty": "交易对手方",
    "contract_type": "合同类型",
    "contract_period": "合同期限",
    "product_service": "产品/服务",
    "is_framework_agreement": "是否框架协议",
    "project_name": "项目名称",
    "bid_amount": "中标金额",
    "project_owner": "项目业主",
    "project_period": "项目周期",
    "bid_rank": "中标名次",
    "product_name": "产品名称",
    "approval_type": "审批类型",
    "regulatory_body": "监管机构",
    "indication": "适应症/应用领域",
    "market_size_estimate": "市场规模估算",
    "capacity": "产能",
    "investment_amount": "投资金额",
    "expected_revenue_contribution": "预计收入贡献",
    "product_type": "产品类型",
    "partner_name": "合作方",
    "cooperation_scope": "合作范围",
    "cooperation_period": "合作期限",
    "partner_type": "合作方类型",
    "exclusivity": "排他性",
    # F 风险事件
    "case_type": "案件类型",
    "involved_amount": "涉案金额",
    "plaintiff_defendant": "原被告",
    "case_status": "案件进展",
    "potential_loss_estimate": "潜在损失估算",
    "court_level": "法院层级",
    "penalty_type": "处罚类型",
    "penalty_amount": "处罚金额",
    "violation_type": "违规类型",
    "rectification_deadline": "整改期限",
    "business_impact": "经营影响",
    "investigation_body": "调查机构",
    "investigation_reason": "调查原因",
    "subjects_involved": "涉及对象",
    "stage": "调查阶段",
    "default_amount": "逾期金额",
    "default_ratio_pct": "逾期比例",
    "creditor": "债权人",
    "debt_type": "债务类型",
    "default_reason": "逾期原因",
    "negotiation_status": "协商进展",
    "risk_type": "风险类型",
    "trigger_condition": "触发条件",
    "deadline": "期限",
    "remediation_possible": "整改可能性",
    # G 治理与交易
    "executive_name": "高管姓名",
    "position": "职位",
    "change_type": "变动类型",
    "reason": "变动原因",
    "successor": "继任者",
    "background": "背景",
    "new_controller": "新实控人",
    "old_controller": "原实控人",
    "change_method": "变更方式",
    "transfer_ratio": "转让比例",
    "transfer_price": "转让价格",
    "new_controller_bg": "新实控人背景",
    "regulatory_approval_needed": "是否需监管审批",
    "halt_reason": "停牌原因",
    "halt_date": "停牌日期",
    "resume_date": "复牌日期",
    "expected_duration": "预计时长",
    "volatility_period": "波动区间",
    "volatility_type": "波动类型",
    "company_explanation": "公司说明",
    "regulatory_inquiry": "监管问询",
    # 通用提取字段（_generic_extraction_prompt）
    "event_type": "事件类型",
    "amount": "涉及金额",
    "ratio": "涉及比例",
    "direction": "方向",
    "key_entities": "关键主体",
    "summary": "核心摘要",
}

# ── 单位 → 中文 ────────────────────────────────────────────────

UNIT_LABELS: dict[str, str] = {
    "CNY": "元",
    "CNY_100M": "亿元",
    "pct": "%",
    "shares": "股",
    "万元": "万元",
    "万": "万",
}


def field_label(name: str) -> str:
    """字段英文名 → 中文标签；未命中回退原名。"""
    return FIELD_LABELS.get(name, name)


def unit_label(unit: str | None) -> str:
    if not unit:
        return ""
    return UNIT_LABELS.get(unit, unit)


def category_label(major: str) -> str:
    """大类代码 A-G → 中文标签；未命中回退代码。"""
    cat = event_registry.categories.get(major)
    return cat.label if cat else major


def sub_label(major: str, sub: str) -> str:
    """子类代码 → 中文标签；未命中回退代码。"""
    cat = event_registry.categories.get(major)
    if cat and sub in cat.subcategories:
        return cat.subcategories[sub].label
    return sub

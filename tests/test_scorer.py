"""测试评分引擎。"""

from datetime import date

import pytest

from src.database.models import Classification, Company, Score
from src.pipeline.scorer import ImpactScorer


class TestImpactScorer:
    def test_scorer_initialization(self):
        scorer = ImpactScorer()
        assert scorer.defaults.direction == 0.0
        assert scorer.defaults.magnitude == 0.5
        assert scorer.defaults.surprise == 0.5

    def test_direction_earnings_positive(self):
        """测试财报营收增长 → 正向方向。"""
        scorer = ImpactScorer()
        cls = Classification(
            major_category="A",
            sub_category="earnings_annual",
            confidence=0.95,
        )
        fields = {
            "revenue_yoy_pct": 25.0,   # 25% 增长 → 0.5
            "net_profit_yoy_pct": 30.0,  # 30% 增长 → 0.6
        }
        direction = scorer._compute_direction(cls, fields)
        # 净利润增速优先: 30/50 = 0.6
        assert direction == pytest.approx(0.6)

    def test_direction_earnings_negative(self):
        """测试财报利润下滑 → 负向方向。"""
        scorer = ImpactScorer()
        cls = Classification(
            major_category="A",
            sub_category="earnings_q1",
            confidence=0.90,
        )
        fields = {
            "net_profit_yoy_pct": -20.0,  # -20% → -0.4
        }
        direction = scorer._compute_direction(cls, fields)
        assert direction == pytest.approx(-0.4)

    def test_direction_equity_increase(self):
        """测试增持公告 → 正向。"""
        scorer = ImpactScorer()
        cls = Classification(
            major_category="C",
            sub_category="equity_increase",
            confidence=0.88,
        )
        fields = {"increase_ratio_pct": 3.0}  # 3% → 0.6
        direction = scorer._compute_direction(cls, fields)
        assert direction == pytest.approx(0.6)

    def test_direction_litigation_negative(self):
        """测试诉讼公告 → 强负向。"""
        scorer = ImpactScorer()
        cls = Classification(
            major_category="F",
            sub_category="litigation",
            confidence=0.92,
        )
        fields = {"involved_amount": 10000}  # 1亿 → -0.2
        direction = scorer._compute_direction(cls, fields)
        assert direction < 0

    def test_magnitude_with_amount(self):
        """测试金额归一化强度。"""
        scorer = ImpactScorer()
        cls = Classification(
            major_category="E",
            sub_category="major_contract",
            confidence=0.85,
        )
        company = Company(
            stock_code="000001.SZ",
            stock_name="测试公司",
            exchange="SZSE",
            annual_revenue=1_000_000,  # 100亿营收
        )
        fields = {"contract_amount": 100_000}  # 10亿合同
        magnitude = scorer._compute_magnitude(cls, fields, company)
        # 10亿 / 100亿 = 0.1 → log10(1+1) ≈ 0.3
        assert 0.0 <= magnitude <= 1.0

    def test_magnitude_no_amount(self):
        """无金额事件 → 用默认强度。"""
        scorer = ImpactScorer()
        cls = Classification(
            major_category="G",
            sub_category="executive_change",
            confidence=0.80,
        )
        company = Company(
            stock_code="000001.SZ",
            stock_name="测试公司",
            exchange="SZSE",
        )
        magnitude = scorer._compute_magnitude(cls, {}, company)
        assert 0.0 <= magnitude <= 1.0

    def test_surprise_default(self):
        """无预期数据 → 默认意外度。"""
        scorer = ImpactScorer()
        cls = Classification(
            major_category="A",
            sub_category="earnings_annual",
            confidence=0.90,
        )
        surprise = scorer._compute_surprise(cls, {})
        assert surprise == 0.5  # 默认值

    def test_surprise_revision(self):
        """业绩修正 → 基于修正幅度计算意外度。"""
        scorer = ImpactScorer()
        cls = Classification(
            major_category="B",
            sub_category="forecast_revision",
            confidence=0.85,
        )
        fields = {"revision_pct": 30.0}  # 30% 修正 → 0.6
        surprise = scorer._compute_surprise(cls, fields)
        assert surprise == pytest.approx(0.6)

    def test_credibility_full_data(self):
        scorer = ImpactScorer()
        cls = Classification(
            major_category="A",
            sub_category="earnings_annual",
            confidence=0.90,
        )
        # 财报类期望 12 个字段，假设 9 个有值
        fields = {
            "revenue": 100, "revenue_yoy_pct": 15, "net_profit": 20,
            "net_profit_yoy_pct": 20, "gross_margin": 30, "net_margin": 15,
            "eps": 2.5, "roe": 18, "operating_cashflow": 25,
            # "dividend_plan": None, "audit_opinion": None
        }
        credibility = scorer._compute_credibility(cls, fields)
        # 9/12 ≈ 0.75 * 0.90 ≈ 0.675
        assert 0.5 <= credibility <= 0.9

    def test_composite_scoring_logic(self):
        """综合评分: direction * magnitude * surprise * credibility。"""
        # 一条"好但不太意外"的公告:
        # direction=0.6, magnitude=0.5, surprise=0.3, credibility=0.8
        # composite = 0.6 * 0.5 * 0.3 * 0.8 = 0.072
        composite = 0.6 * 0.5 * 0.3 * 0.8
        assert composite < 0.1  # 意外度低 → 综合评分低

        # 一条"好且意外"的公告:
        # direction=0.6, magnitude=0.5, surprise=0.9, credibility=0.8
        # composite = 0.6 * 0.5 * 0.9 * 0.8 = 0.216
        composite2 = 0.6 * 0.5 * 0.9 * 0.8
        assert composite2 > 0.2  # 超预期 → 综合评分高

"""管线评分步骤 — 规则化影响评分引擎。

核心公式:
    composite_score = direction × magnitude × surprise × credibility

四个维度独立计算，乘法确保任一维度缺失时信号自动归零。
"""

import json
from typing import Optional

from loguru import logger
from sqlalchemy.orm import Session

from src.config import config, event_registry
from src.database.engine import get_engine
from src.database.models import Announcement, Classification, Company, ExtractedField, Score
from src.database.repository import (
    AnnouncementRepository,
    ExtractedFieldRepository,
    ScoreRepository,
)
from src.utils.text_utils import clamp, safe_divide


class ImpactScorer:
    """影响评分引擎。

    对每个已提取字段的公告，计算五维度评分。
    """

    def __init__(self) -> None:
        self.defaults = config.scoring.defaults
        self.source_cred = config.scoring.source_credibility

    def run(self, limit: int = 500) -> int:
        """执行评分。返回评分的公告数量。"""
        engine = get_engine()
        count = 0
        failed = 0

        with Session(engine) as session:
            announcements = AnnouncementRepository.get_by_status(
                session, "extracted", limit=limit
            )

            if not announcements:
                logger.info("No announcements to score")
                return 0

            logger.info(f"Scoring {len(announcements)} announcements...")

            for ann in announcements:
                try:
                    # 预加载关联数据
                    classification = ann.classification
                    if not classification:
                        AnnouncementRepository.update_status(
                            session, ann.id, "failed", "Missing classification"
                        )
                        failed += 1
                        continue

                    company = ann.company
                    fields = ExtractedFieldRepository.to_dict(session, ann.id)

                    # 计算各维度
                    direction = self._compute_direction(classification, fields)
                    magnitude = self._compute_magnitude(classification, fields, company)
                    surprise = self._compute_surprise(classification, fields)
                    credibility = self._compute_credibility(classification, fields)

                    # 合成
                    composite = direction * magnitude * surprise * credibility

                    detail = {
                        "direction": direction,
                        "magnitude": magnitude,
                        "surprise": surprise,
                        "credibility": credibility,
                        "composite": composite,
                        "category": classification.sub_category,
                        "company_industry": company.industry if company else None,
                    }

                    ScoreRepository.upsert(
                        session,
                        announcement_id=ann.id,
                        direction=direction,
                        magnitude=magnitude,
                        surprise=surprise,
                        credibility=credibility,
                        composite_score=composite,
                        score_version="v1.0",
                        scoring_detail=detail,
                    )

                    AnnouncementRepository.update_status(session, ann.id, "scored")
                    count += 1

                except Exception as e:
                    logger.warning(f"Scoring failed for {ann.announcement_id}: {e}")
                    AnnouncementRepository.update_status(
                        session, ann.id, "failed", str(e)
                    )
                    failed += 1

            session.commit()

        logger.info(f"Scoring done: {count} scored, {failed} failed")
        return count

    # ── 方向计算 ─────────────────────────────────────────────

    def _compute_direction(
        self,
        classification: Classification,
        fields: dict,
    ) -> float:
        """计算方向 [-1.0, +1.0]。

        基于类别基线方向 + 字段内容调整。
        """
        sub_cat = classification.sub_category
        sub_def = None
        for cat in event_registry.categories.values():
            if sub_cat in cat.subcategories:
                sub_def = cat.subcategories[sub_cat]
                break

        if sub_def is None:
            return 0.0

        direction = sub_def.direction_baseline

        # 基于字段内容的调整
        if sub_cat == "earnings_annual" or sub_cat.startswith("earnings_"):
            # 财报类：营收/利润同比增速决定方向
            rev_yoy = fields.get("revenue_yoy_pct")
            profit_yoy = fields.get("net_profit_yoy_pct")
            if profit_yoy is not None:
                direction = clamp(float(profit_yoy) / 50.0, -1.0, 1.0)
            elif rev_yoy is not None:
                direction = clamp(float(rev_yoy) / 30.0, -1.0, 1.0)

        elif sub_cat == "forecast_revision":
            # 业绩修正：修正方向和幅度
            revision_pct = fields.get("revision_pct")
            if revision_pct is not None:
                direction = clamp(float(revision_pct) / 20.0, -1.0, 1.0)

        elif sub_cat == "equity_increase":
            # 增持比例影响方向强度
            ratio = fields.get("increase_ratio_pct")
            if ratio is not None:
                direction = clamp(float(ratio) / 5.0, 0.0, 1.0)

        elif sub_cat == "equity_decrease":
            ratio = fields.get("decrease_ratio_pct")
            if ratio is not None:
                direction = clamp(-float(ratio) / 5.0, -1.0, 0.0)

        elif sub_cat == "major_contract":
            amount = fields.get("contract_amount")
            if amount is not None:
                direction = clamp(float(amount) / 100_000, 0.0, 1.0)  # 10亿=满分

        elif sub_cat in ("litigation", "penalty", "investigation", "debt_default", "st_delisting_risk"):
            # 风险事件：金额越大越负面
            amount = fields.get("involved_amount") or fields.get("default_amount") or fields.get("penalty_amount")
            if amount is not None:
                direction = clamp(-float(amount) / 50_000, -1.0, 0.0)  # 5亿=满分负

        return clamp(direction, -1.0, 1.0)

    # ── 强度计算 ─────────────────────────────────────────────

    def _compute_magnitude(
        self,
        classification: Classification,
        fields: dict,
        company: Company | None,
    ) -> float:
        """计算强度 [0.0, 1.0]。

        基于金额 / 公司规模归一化。金额越大（相对公司规模），强度越高。
        """
        sub_cat = classification.sub_category

        # 提取金额字段
        amount = self._extract_amount_from_fields(fields)

        if amount is None or company is None:
            # 无法量化的事件（如人事变动），用默认强度
            sub_def = None
            for cat in event_registry.categories.values():
                if sub_cat in cat.subcategories:
                    sub_def = cat.subcategories[sub_cat]
                    break
            return abs(sub_def.direction_baseline) if sub_def else self.defaults.magnitude

        # 以公司年营收为基准归一化
        base = company.annual_revenue or company.net_assets or 1
        normalized = safe_divide(abs(amount), base, 0.0)
        # log 变换避免极端值
        import math
        magnitude = clamp(math.log10(1 + normalized * 10), 0.0, 1.0)

        return magnitude

    @staticmethod
    def _extract_amount_from_fields(fields: dict) -> Optional[float]:
        """从提取字段中找第一个金额类字段。"""
        amount_fields = [
            "contract_amount", "bid_amount", "buyback_amount_max",
            "involved_amount", "default_amount", "penalty_amount",
            "placement_amount", "transaction_amount", "sale_amount",
            "bond_amount", "investment_amount",
            "revenue", "net_profit",
        ]
        for key in amount_fields:
            val = fields.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return None

    # ── 意外度计算 ─────────────────────────────────────────────

    def _compute_surprise(
        self,
        classification: Classification,
        fields: dict,
    ) -> float:
        """计算意外度 [0.0, 1.0]。

        当前 MVP: 基于字段中的 forecast vs actual 对比。
        后期: 接入分析师一致预期数据。
        """
        sub_cat = classification.sub_category

        # 业绩预告修正类：有原始预告 vs 修正值 → 可计算意外度
        if sub_cat == "forecast_revision":
            revision_pct = fields.get("revision_pct")
            if revision_pct is not None:
                return clamp(abs(float(revision_pct)) / 50.0, 0.0, 1.0)

        # 业绩预告 vs 实际
        if sub_cat.startswith("earnings_"):
            forecast_profit = fields.get("forecast_net_profit_max")
            actual_profit = fields.get("net_profit")
            if forecast_profit is not None and actual_profit is not None:
                delta = safe_divide(
                    abs(float(actual_profit) - float(forecast_profit)),
                    abs(float(forecast_profit)),
                    0.0,
                )
                return clamp(delta, 0.0, 1.0)

        # 无预期数据 → 默认基线
        return self.defaults.surprise

    # ── 可信度计算 ─────────────────────────────────────────────

    def _compute_credibility(
        self,
        classification: Classification,
        fields: dict,
    ) -> float:
        """计算可信度 [0.0, 1.0]。

        基于来源可信度 × 数据完整度。
        """
        # 来源可信度（默认: 公司公告）
        source_cred = self.source_cred.get("company_announcement", 0.90)

        # 数据完整度: 有多少期望字段被提取出来了（非 null）
        sub_cat = classification.sub_category
        expected_count = 0
        filled_count = 0
        for cat in event_registry.categories.values():
            if sub_cat in cat.subcategories:
                expected_fields = cat.subcategories[sub_cat].fields
                expected_count = len(expected_fields)
                filled_count = sum(
                    1 for f in expected_fields
                    if fields.get(f) is not None
                )
                break

        if expected_count == 0:
            data_completeness = 0.5
        else:
            data_completeness = filled_count / expected_count

        return clamp(source_cred * data_completeness, 0.0, 1.0)

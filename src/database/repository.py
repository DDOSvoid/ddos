"""数据访问层 — 各表的 CRUD 和批量操作。

使用 Repository 模式封装 SQLAlchemy 查询，业务逻辑只依赖这些接口。
"""

import json
from datetime import date, datetime
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from src.database.models import (
    Announcement,
    Classification,
    Company,
    DailyReport,
    ExtractedField,
    PipelineRun,
    Score,
)


# ── 公告方向阈值 ───────────────────────────────────────────────
# 与 models.Score.direction_label 的 ±0.1 判定保持同步（利好/利空/中性）。
# 若改动须两处一起改。
_ANN_DIRECTION_BOUNDARY = 0.1


def _announcement_filter_query(
    session: Session,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    industry_group: str | None = None,
    major_category: str | None = None,
    direction: str | None = None,  # "利好" | "利空" | "中性"
    status: str | None = None,
    keyword: str | None = None,
):
    """公告列表筛选查询（只追加 filter，不排序不分页，供 search/count_search 复用）。"""
    q = session.query(Announcement)
    if start_date is not None:
        q = q.filter(Announcement.published_date >= start_date)
    if end_date is not None:
        q = q.filter(Announcement.published_date <= end_date)
    if status:
        q = q.filter(Announcement.processing_status == status)
    if keyword:
        q = q.filter(Announcement.title.ilike(f"%{keyword}%"))
    if industry_group or major_category:
        # 1:1 关联（classification 对 announcement 唯一），join 不产生行膨胀
        q = q.join(Classification, Announcement.classification)
        if industry_group:
            q = q.filter(Classification.industry_group == industry_group)
        if major_category:
            q = q.filter(Classification.major_category == major_category)
    if direction:
        # 1:1 关联（score 对 announcement 唯一）
        q = q.join(Score, Announcement.score)
        if direction == "利好":
            q = q.filter(Score.composite_score > _ANN_DIRECTION_BOUNDARY)
        elif direction == "利空":
            q = q.filter(Score.composite_score < -_ANN_DIRECTION_BOUNDARY)
        else:  # 中性
            q = q.filter(func.abs(Score.composite_score) <= _ANN_DIRECTION_BOUNDARY)
    return q


# ── Company Repository ─────────────────────────────────────────


class CompanyRepository:
    """公司信息 CRUD。"""

    @staticmethod
    def upsert(session: Session, stock_code: str, **kwargs) -> Company:
        """插入或更新公司记录（以 stock_code 为唯一键）。"""
        company = session.query(Company).filter_by(stock_code=stock_code).first()
        if company:
            for key, value in kwargs.items():
                if hasattr(company, key) and value is not None:
                    setattr(company, key, value)
            company.updated_at = datetime.utcnow()
        else:
            company = Company(stock_code=stock_code, **kwargs)
            session.add(company)
        session.flush()
        return company

    @staticmethod
    def get_by_code(session: Session, stock_code: str) -> Optional[Company]:
        return session.query(Company).filter_by(stock_code=stock_code).first()

    @staticmethod
    def get_tracked(session: Session, industry: Optional[str] = None) -> list[Company]:
        q = session.query(Company).filter_by(is_tracked=True)
        if industry:
            q = q.filter_by(industry=industry)
        return q.all()

    @staticmethod
    def get_codes(session: Session, tracked_only: bool = True) -> list[str]:
        q = session.query(Company.stock_code)
        if tracked_only:
            q = q.filter_by(is_tracked=True)
        return [row[0] for row in q.all()]

    @staticmethod
    def get_industries(session: Session) -> list[str]:
        rows = (
            session.query(Company.industry)
            .filter(Company.industry.isnot(None))
            .distinct()
            .all()
        )
        return [r[0] for r in rows if r[0]]

    @staticmethod
    def count(session: Session) -> int:
        return session.query(func.count(Company.id)).scalar() or 0


# ── Announcement Repository ────────────────────────────────────


class AnnouncementRepository:
    """公告 CRUD。"""

    @staticmethod
    def bulk_upsert(session: Session, records: list[dict]) -> int:
        """批量 upsert 公告（以 announcement_id 为唯一键）。返回新增/更新数量。"""
        count = 0
        for rec in records:
            ann_id = rec.get("announcement_id")
            if not ann_id:
                continue
            existing = (
                session.query(Announcement).filter_by(announcement_id=ann_id).first()
            )
            if existing:
                # 更新已有记录（但不覆盖已处理的状态）
                for key, value in rec.items():
                    if key != "processing_status" and hasattr(existing, key) and value is not None:
                        setattr(existing, key, value)
            else:
                session.add(Announcement(**rec))
                count += 1
        session.flush()
        return count

    @staticmethod
    def get_by_status(
        session: Session, status: str, limit: int = 1000
    ) -> list[Announcement]:
        return (
            session.query(Announcement)
            .filter_by(processing_status=status)
            .limit(limit)
            .all()
        )

    @staticmethod
    def count_by_status(session: Session, status: str) -> int:
        return (
            session.query(func.count(Announcement.id))
            .filter_by(processing_status=status)
            .scalar()
        ) or 0

    @staticmethod
    def update_status(
        session: Session,
        announcement_id: int,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        session.query(Announcement).filter_by(id=announcement_id).update(
            {"processing_status": status, "error_message": error}
        )
        session.flush()

    @staticmethod
    def get_by_date_range(
        session: Session,
        start_date: date,
        end_date: date,
        status: Optional[str] = None,
    ) -> list[Announcement]:
        q = session.query(Announcement).filter(
            Announcement.published_date >= start_date,
            Announcement.published_date <= end_date,
        )
        if status:
            q = q.filter_by(processing_status=status)
        return q.all()

    @staticmethod
    def get_with_classification(
        session: Session, status: str = "classified", limit: int = 500
    ) -> list[Announcement]:
        """获取已分类的公告（带预加载的关系）。"""
        return (
            session.query(Announcement)
            .filter_by(processing_status=status)
            .limit(limit)
            .all()
        )

    # ── Web 展示用只读查询 ────────────────────────────────────

    @staticmethod
    def count(session: Session) -> int:
        return session.query(func.count(Announcement.id)).scalar() or 0

    @staticmethod
    def count_group_by_status(session: Session) -> dict[str, int]:
        """各处理状态计数，返回 {status: count}。"""
        rows = (
            session.query(Announcement.processing_status, func.count(Announcement.id))
            .group_by(Announcement.processing_status)
            .all()
        )
        return {s: c for s, c in rows}

    @staticmethod
    def get_by_id(session: Session, announcement_id: int) -> Optional[Announcement]:
        """按主键取公告，预加载全部展示用关系（避免 N+1 / DetachedInstanceError）。"""
        return (
            session.query(Announcement)
            .options(
                selectinload(Announcement.company),
                selectinload(Announcement.classification),
                selectinload(Announcement.score),
                selectinload(Announcement.extracted_fields),
            )
            .filter_by(id=announcement_id)
            .first()
        )

    @staticmethod
    def search(
        session: Session,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        industry_group: str | None = None,
        major_category: str | None = None,
        direction: str | None = None,
        status: str | None = None,
        keyword: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> list[Announcement]:
        """分页搜索公告（带筛选），预加载 company/classification/score。"""
        q = _announcement_filter_query(
            session,
            start_date=start_date,
            end_date=end_date,
            industry_group=industry_group,
            major_category=major_category,
            direction=direction,
            status=status,
            keyword=keyword,
        )
        return (
            q.options(
                selectinload(Announcement.company),
                selectinload(Announcement.classification),
                selectinload(Announcement.score),
            )
            .order_by(Announcement.published_date.desc(), Announcement.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

    @staticmethod
    def count_search(
        session: Session,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        industry_group: str | None = None,
        major_category: str | None = None,
        direction: str | None = None,
        status: str | None = None,
        keyword: str | None = None,
    ) -> int:
        """search 对应的总数（用于分页）。"""
        q = _announcement_filter_query(
            session,
            start_date=start_date,
            end_date=end_date,
            industry_group=industry_group,
            major_category=major_category,
            direction=direction,
            status=status,
            keyword=keyword,
        )
        return q.with_entities(func.count(Announcement.id)).scalar() or 0


# ── Classification Repository ──────────────────────────────────


class ClassificationRepository:
    """分类结果 CRUD。"""

    @staticmethod
    def upsert(
        session: Session,
        announcement_id: int,
        major_category: str,
        sub_category: str,
        confidence: float,
        model_version: Optional[str] = None,
        industry: Optional[str] = None,
        industry_group: Optional[str] = None,
    ) -> Classification:
        existing = (
            session.query(Classification)
            .filter_by(announcement_id=announcement_id)
            .first()
        )
        if existing:
            existing.major_category = major_category
            existing.sub_category = sub_category
            existing.confidence = confidence
            existing.model_version = model_version
            existing.industry = industry
            existing.industry_group = industry_group
            existing.classified_at = datetime.utcnow()
        else:
            existing = Classification(
                announcement_id=announcement_id,
                major_category=major_category,
                sub_category=sub_category,
                confidence=confidence,
                model_version=model_version,
                industry=industry,
                industry_group=industry_group,
            )
            session.add(existing)
        session.flush()
        return existing


# ── ExtractedField Repository ──────────────────────────────────


class ExtractedFieldRepository:
    """提取字段 CRUD。"""

    @staticmethod
    def bulk_insert(
        session: Session,
        announcement_id: int,
        fields: list[dict],
    ) -> list[ExtractedField]:
        """批量插入提取字段。先删旧数据再插入（幂等）。"""
        # 删除旧字段
        session.query(ExtractedField).filter_by(announcement_id=announcement_id).delete()
        # 插入新字段
        entities = []
        for f in fields:
            ef = ExtractedField(announcement_id=announcement_id, **f)
            session.add(ef)
            entities.append(ef)
        session.flush()
        return entities

    @staticmethod
    def get_by_announcement(
        session: Session, announcement_id: int
    ) -> list[ExtractedField]:
        return (
            session.query(ExtractedField)
            .filter_by(announcement_id=announcement_id)
            .all()
        )

    @staticmethod
    def to_dict(
        session: Session, announcement_id: int
    ) -> dict[str, str | float | None]:
        """将提取字段转为 {field_name: field_value} 字典。"""
        fields = ExtractedFieldRepository.get_by_announcement(session, announcement_id)
        result = {}
        for f in fields:
            if f.field_type == "numeric":
                result[f.field_name] = f.value_as_float()
            else:
                result[f.field_name] = f.field_value
        return result


# ── Score Repository ───────────────────────────────────────────


class ScoreRepository:
    """评分 CRUD。"""

    @staticmethod
    def upsert(
        session: Session,
        announcement_id: int,
        direction: float,
        magnitude: float,
        surprise: float,
        credibility: float,
        composite_score: float,
        market_reaction: Optional[float] = None,
        score_version: Optional[str] = None,
        scoring_detail: Optional[dict] = None,
    ) -> Score:
        existing = session.query(Score).filter_by(announcement_id=announcement_id).first()
        detail_json = json.dumps(scoring_detail, ensure_ascii=False) if scoring_detail else None
        if existing:
            existing.direction = direction
            existing.magnitude = magnitude
            existing.surprise = surprise
            existing.credibility = credibility
            existing.composite_score = composite_score
            existing.market_reaction = market_reaction
            existing.score_version = score_version
            existing.scoring_detail = detail_json
            existing.scored_at = datetime.utcnow()
        else:
            existing = Score(
                announcement_id=announcement_id,
                direction=direction,
                magnitude=magnitude,
                surprise=surprise,
                credibility=credibility,
                composite_score=composite_score,
                market_reaction=market_reaction,
                score_version=score_version,
                scoring_detail=detail_json,
            )
            session.add(existing)
        session.flush()
        return existing

    @staticmethod
    def get_high_impact(
        session: Session,
        threshold: float = 0.5,
        limit: int = 50,
    ) -> list[Score]:
        return (
            session.query(Score)
            .filter(func.abs(Score.composite_score) >= threshold)
            .order_by(func.abs(Score.composite_score).desc())
            .limit(limit)
            .all()
        )


# ── Stats Repository（仪表盘聚合） ─────────────────────────────


class StatsRepository:
    """只读聚合统计 — 供 Web 仪表盘使用。"""

    @staticmethod
    def category_distribution(session: Session) -> list[tuple[str, int]]:
        """按大类 A-G 统计公告分类数。"""
        return (
            session.query(Classification.major_category, func.count(Classification.id))
            .group_by(Classification.major_category)
            .order_by(func.count(Classification.id).desc())
            .all()
        )

    @staticmethod
    def industry_group_distribution(session: Session) -> list[tuple[str, int]]:
        """按行业域统计公告分类数（空行业域忽略）。"""
        return (
            session.query(Classification.industry_group, func.count(Classification.id))
            .filter(Classification.industry_group.isnot(None))
            .group_by(Classification.industry_group)
            .order_by(func.count(Classification.id).desc())
            .all()
        )

    @staticmethod
    def direction_distribution(session: Session) -> dict[str, int]:
        """按利好/利空/中性统计评分公告数（阈值与 Score.direction_label 一致）。"""
        return {
            "利好": (
                session.query(func.count(Score.id))
                .filter(Score.composite_score > _ANN_DIRECTION_BOUNDARY)
                .scalar()
                or 0
            ),
            "利空": (
                session.query(func.count(Score.id))
                .filter(Score.composite_score < -_ANN_DIRECTION_BOUNDARY)
                .scalar()
                or 0
            ),
            "中性": (
                session.query(func.count(Score.id))
                .filter(func.abs(Score.composite_score) <= _ANN_DIRECTION_BOUNDARY)
                .scalar()
                or 0
            ),
        }

    @staticmethod
    def high_impact_count(session: Session, threshold: float = 0.5) -> int:
        return (
            session.query(func.count(Score.id))
            .filter(func.abs(Score.composite_score) >= threshold)
            .scalar()
            or 0
        )


# ── DailyReport Repository ─────────────────────────────────────


class DailyReportRepository:
    """每日报告 CRUD。"""

    @staticmethod
    def upsert(session: Session, **kwargs) -> DailyReport:
        report_date = kwargs.get("report_date")
        industry_filter = kwargs.get("industry_filter")
        existing = (
            session.query(DailyReport)
            .filter_by(report_date=report_date, industry_filter=industry_filter)
            .first()
        )
        if existing:
            for key, value in kwargs.items():
                if hasattr(existing, key) and value is not None:
                    setattr(existing, key, value)
            existing.generated_at = datetime.utcnow()
        else:
            existing = DailyReport(**kwargs)
            session.add(existing)
        session.flush()
        return existing

    @staticmethod
    def get_by_date(
        session: Session, report_date: date
    ) -> list[DailyReport]:
        return session.query(DailyReport).filter_by(report_date=report_date).all()

    @staticmethod
    def get_all(session: Session, limit: int = 50) -> list[DailyReport]:
        """按日期倒序取最近报告。"""
        return (
            session.query(DailyReport)
            .order_by(DailyReport.report_date.desc())
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_primary(session: Session, report_date: date) -> Optional[DailyReport]:
        """取某日全市场报告（industry_filter 为空/NULL）。"""
        return (
            session.query(DailyReport)
            .filter_by(report_date=report_date)
            .filter(
                (DailyReport.industry_filter.is_(None))
                | (DailyReport.industry_filter == "")
            )
            .first()
        )


# ── PipelineRun Repository ─────────────────────────────────────


class PipelineRunRepository:
    """管线运行记录 CRUD。"""

    @staticmethod
    def create(
        session: Session,
        run_id: str,
        run_type: str,
        metadata: Optional[dict] = None,
    ) -> PipelineRun:
        run = PipelineRun(
            run_id=run_id,
            run_type=run_type,
            status="running",
            run_metadata=json.dumps(metadata, ensure_ascii=False) if metadata else None,
        )
        session.add(run)
        session.flush()
        return run

    @staticmethod
    def complete(
        session: Session,
        run_id: str,
        records_processed: int = 0,
        records_failed: int = 0,
    ) -> None:
        session.query(PipelineRun).filter_by(run_id=run_id).update({
            "status": "completed",
            "completed_at": datetime.utcnow(),
            "records_processed": records_processed,
            "records_failed": records_failed,
        })

    @staticmethod
    def fail(session: Session, run_id: str, error: str) -> None:
        session.query(PipelineRun).filter_by(run_id=run_id).update({
            "status": "failed",
            "completed_at": datetime.utcnow(),
            "error_message": error,
        })

    @staticmethod
    def get_recent(session: Session, limit: int = 10) -> list[PipelineRun]:
        return (
            session.query(PipelineRun)
            .order_by(PipelineRun.started_at.desc())
            .limit(limit)
            .all()
        )

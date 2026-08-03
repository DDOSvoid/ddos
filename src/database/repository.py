"""数据访问层 — 各表的 CRUD 和批量操作。

使用 Repository 模式封装 SQLAlchemy 查询，业务逻辑只依赖这些接口。
"""

import json
from datetime import date, datetime
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.database.models import (
    Announcement,
    Classification,
    Company,
    DailyReport,
    ExtractedField,
    PipelineRun,
    Score,
)


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
            existing.classified_at = datetime.utcnow()
        else:
            existing = Classification(
                announcement_id=announcement_id,
                major_category=major_category,
                sub_category=sub_category,
                confidence=confidence,
                model_version=model_version,
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

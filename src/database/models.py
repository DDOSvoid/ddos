"""SQLAlchemy ORM 模型 — 公告分析核心表。

表结构:
  companies       — 公司基础信息
  announcements   — 公告原文 + 处理状态
  classifications — 分类结果 (A-G + 子类别)
  classification_revisions — 分类人工修订审计历史
  extracted_fields— LLM 提取的结构化字段
  scores          — 影响评分五维度
  daily_reports   — 生成的每日报告
  pipeline_runs   — 管线运行记录
"""

import json
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ── Companies ──────────────────────────────────────────────────


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stock_code: Mapped[str] = mapped_column(String(10), unique=True, nullable=False, comment="股票代码, e.g. 000001.SZ")
    stock_name: Mapped[str] = mapped_column(String(100), nullable=False, comment="股票名称")
    exchange: Mapped[str] = mapped_column(String(10), nullable=False, comment="交易所 SSE/SZSE")
    industry: Mapped[str | None] = mapped_column(String(50), comment="行业分类（申万一级）")
    market_cap: Mapped[int | None] = mapped_column(Integer, comment="总市值（万元）")
    total_shares: Mapped[int | None] = mapped_column(Integer, comment="总股本（万股）")
    annual_revenue: Mapped[int | None] = mapped_column(Integer, comment="最近财年营收（万元）")
    net_assets: Mapped[int | None] = mapped_column(Integer, comment="净资产（万元）")
    is_tracked: Mapped[bool] = mapped_column(default=True, comment="是否跟踪")
    tracked_since: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    announcements: Mapped[list["Announcement"]] = relationship(back_populates="company", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<Company {self.stock_code} {self.stock_name}>"


# ── Announcements ──────────────────────────────────────────────


class Announcement(Base):
    __tablename__ = "announcements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), nullable=False)
    announcement_id: Mapped[str | None] = mapped_column(String(50), unique=True, comment="外部系统公告 ID")
    title: Mapped[str] = mapped_column(Text, nullable=False, comment="公告标题")
    full_text: Mapped[str | None] = mapped_column(Text, comment="公告全文")
    pdf_url: Mapped[str | None] = mapped_column(Text, comment="PDF 链接")
    published_date: Mapped[date] = mapped_column(Date, nullable=False, comment="发布日期")
    source_url: Mapped[str | None] = mapped_column(Text, comment="原始来源链接")
    raw_response: Mapped[str | None] = mapped_column(Text, comment="API 原始响应 JSON")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    processing_status: Mapped[str] = mapped_column(
        String(20),
        default="fetched",
        comment="处理状态: fetched/preprocessed/classified/extracted/scored/reported/failed",
    )
    error_message: Mapped[str | None] = mapped_column(Text)

    # 约束
    __table_args__ = (
        CheckConstraint(
            "processing_status IN ('fetched','preprocessed','classified','extracted','scored','reported','failed')",
            name="ck_announcements_status",
        ),
    )

    # 关系
    company: Mapped["Company"] = relationship(back_populates="announcements")
    classification: Mapped["Classification | None"] = relationship(back_populates="announcement", uselist=False)
    extracted_fields: Mapped[list["ExtractedField"]] = relationship(back_populates="announcement")
    score: Mapped["Score | None"] = relationship(back_populates="announcement", uselist=False)

    def __repr__(self) -> str:
        return f"<Announcement {self.announcement_id} [{self.processing_status}]>"


# ── Classifications ────────────────────────────────────────────


class Classification(Base):
    __tablename__ = "classifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    announcement_id: Mapped[int] = mapped_column(
        ForeignKey("announcements.id"), unique=True, nullable=False
    )
    major_category: Mapped[str] = mapped_column(String(2), nullable=False, comment="大类 A-G/X")
    sub_category: Mapped[str] = mapped_column(String(50), nullable=False, comment="子类别代码")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, comment="分类置信度")
    model_version: Mapped[str | None] = mapped_column(String(50), comment="模型版本")
    industry: Mapped[str | None] = mapped_column(String(50), comment="行业（申万一级，来自公司，非模型产出）")
    industry_group: Mapped[str | None] = mapped_column(String(20), comment="行业域（粗分，用于差异化处理/专家模型路由）")
    classification_source: Mapped[str | None] = mapped_column(
        String(30), default="model", comment="rule/rule+model/model/abstained/manual/legacy"
    )
    rule_id: Mapped[str | None] = mapped_column(String(80), comment="命中的高精度规则")
    document_type: Mapped[str | None] = mapped_column(String(50), comment="文档类型（独立于主事件）")
    relevance: Mapped[str | None] = mapped_column(
        String(30), comment="core_event/supporting_document/routine_disclosure/uncertain"
    )
    secondary_tags: Mapped[str | None] = mapped_column(Text, comment="辅助标签 JSON")
    model_sub_category: Mapped[str | None] = mapped_column(String(50), comment="原始模型 top-1")
    model_confidence: Mapped[float | None] = mapped_column(Float, comment="原始模型 top-1 概率")
    model_margin: Mapped[float | None] = mapped_column(Float, comment="模型 top-1 与 top-2 概率差")
    needs_review: Mapped[bool | None] = mapped_column(Boolean, default=True, comment="是否进入人工复核队列")
    review_status: Mapped[str | None] = mapped_column(String(20), default="pending")
    review_reason: Mapped[str | None] = mapped_column(Text)
    review_note: Mapped[str | None] = mapped_column(Text)
    reviewed_by: Mapped[str | None] = mapped_column(String(100))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime)
    taxonomy_version: Mapped[str | None] = mapped_column(String(20), default="v2")
    classified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # 关系
    announcement: Mapped["Announcement"] = relationship(back_populates="classification")

    def __repr__(self) -> str:
        return f"<Classification {self.major_category}/{self.sub_category} ({self.confidence:.2f})>"

    @property
    def secondary_tag_list(self) -> list[str]:
        if not self.secondary_tags:
            return []
        try:
            value = json.loads(self.secondary_tags)
            return value if isinstance(value, list) else []
        except (TypeError, json.JSONDecodeError):
            return []


class ClassificationRevision(Base):
    """分类变更审计记录；修订不删除原始轨迹。"""

    __tablename__ = "classification_revisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    announcement_id: Mapped[int] = mapped_column(
        ForeignKey("announcements.id"), nullable=False, index=True
    )
    previous_major_category: Mapped[str | None] = mapped_column(String(2))
    previous_sub_category: Mapped[str | None] = mapped_column(String(50))
    previous_confidence: Mapped[float | None] = mapped_column(Float)
    new_major_category: Mapped[str] = mapped_column(String(2), nullable=False)
    new_sub_category: Mapped[str] = mapped_column(String(50), nullable=False)
    new_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    change_source: Mapped[str] = mapped_column(String(30), nullable=False)
    changed_by: Mapped[str | None] = mapped_column(String(100))
    note: Mapped[str | None] = mapped_column(Text)
    changed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ── Extracted Fields ───────────────────────────────────────────


class ExtractedField(Base):
    __tablename__ = "extracted_fields"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    announcement_id: Mapped[int] = mapped_column(ForeignKey("announcements.id"), nullable=False)
    field_name: Mapped[str] = mapped_column(String(50), nullable=False, comment="字段名")
    field_value: Mapped[str | None] = mapped_column(Text, comment="字段值（统一存为文本）")
    field_type: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="值类型: numeric/text/date/json"
    )
    unit: Mapped[str | None] = mapped_column(String(20), comment="单位: CNY/CNY_100M/pct/shares")
    confidence: Mapped[float | None] = mapped_column(Float, comment="提取置信度")
    model_used: Mapped[str | None] = mapped_column(String(50), comment="使用的 LLM 模型")
    tokens_used: Mapped[int | None] = mapped_column(Integer, comment="消耗 token 数")
    extracted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # 约束
    __table_args__ = (
        CheckConstraint(
            "field_type IN ('numeric','text','date','json')",
            name="ck_extracted_field_type",
        ),
    )

    # 关系
    announcement: Mapped["Announcement"] = relationship(back_populates="extracted_fields")

    def value_as_float(self) -> float | None:
        """尝试将 field_value 解析为浮点数。"""
        if self.field_value is None:
            return None
        try:
            return float(self.field_value.replace(",", ""))
        except (ValueError, AttributeError):
            return None

    def __repr__(self) -> str:
        return f"<ExtractedField {self.field_name}={self.field_value}>"


# ── Scores ─────────────────────────────────────────────────────


class Score(Base):
    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    announcement_id: Mapped[int] = mapped_column(
        ForeignKey("announcements.id"), unique=True, nullable=False
    )
    direction: Mapped[float] = mapped_column(Float, nullable=False, comment="方向 [-1.0, +1.0]")
    magnitude: Mapped[float] = mapped_column(Float, nullable=False, comment="强度 [0.0, 1.0]")
    surprise: Mapped[float] = mapped_column(Float, nullable=False, comment="意外度 [0.0, 1.0]")
    credibility: Mapped[float] = mapped_column(Float, nullable=False, comment="可信度 [0.0, 1.0]")
    market_reaction: Mapped[float | None] = mapped_column(Float, comment="市场反应 [-1.0, +1.0]，事后填充")
    composite_score: Mapped[float] = mapped_column(Float, nullable=False, comment="综合得分")
    score_version: Mapped[str | None] = mapped_column(String(20))
    scoring_detail: Mapped[str | None] = mapped_column(Text, comment="评分详情 JSON")
    scored_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # 关系
    announcement: Mapped["Announcement"] = relationship(back_populates="score")

    @property
    def is_high_impact(self, threshold: float = 0.5) -> bool:
        return abs(self.composite_score) >= threshold

    @property
    def direction_label(self) -> str:
        if self.composite_score > 0.1:
            return "利好"
        elif self.composite_score < -0.1:
            return "利空"
        return "中性"

    def __repr__(self) -> str:
        return f"<Score {self.direction_label} composite={self.composite_score:.3f}>"


# ── Market data cache ─────────────────────────────────────────


class DailyPrice(Base):
    """Source-stamped daily bars used to mature prediction outcomes."""

    __tablename__ = "daily_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_code: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    instrument_type: Mapped[str] = mapped_column(String(10), nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    open_price: Mapped[float] = mapped_column(Float, nullable=False)
    high_price: Mapped[float] = mapped_column(Float, nullable=False)
    low_price: Mapped[float] = mapped_column(Float, nullable=False)
    close_price: Mapped[float] = mapped_column(Float, nullable=False)
    pre_close: Mapped[float | None] = mapped_column(Float)
    pct_change: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    amount: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(30), nullable=False, default="tushare")
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint(
            "instrument_code", "trade_date", name="uq_daily_price_instrument_date"
        ),
        CheckConstraint(
            "instrument_type IN ('stock', 'index')", name="ck_daily_price_type"
        ),
        CheckConstraint(
            "open_price > 0 AND high_price > 0 AND low_price > 0 AND close_price > 0",
            name="ck_daily_price_positive",
        ),
    )


class AnnouncementSourceArchive(Base):
    """Immutable source snapshot; new source versions are appended, never overwritten."""

    __tablename__ = "announcement_source_archives"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    archive_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    announcement_id: Mapped[int] = mapped_column(
        ForeignKey("announcements.id"), nullable=False, index=True
    )
    supersedes_id: Mapped[int | None] = mapped_column(
        ForeignKey("announcement_source_archives.id")
    )
    source: Mapped[str] = mapped_column(String(30), nullable=False)
    source_art_code: Mapped[str] = mapped_column(String(50), nullable=False)
    dataset_role: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    notice_date_raw: Mapped[str | None] = mapped_column(String(40))
    source_eitime_raw: Mapped[str | None] = mapped_column(String(40))
    source_time_candidate: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_time_semantics: Mapped[str] = mapped_column(String(50), nullable=False)
    source_time_authoritative: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    content_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_chars: Mapped[int] = mapped_column(Integer, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    pages_expected: Mapped[int] = mapped_column(Integer, nullable=False)
    pages_fetched: Mapped[int] = mapped_column(Integer, nullable=False)
    content_complete: Mapped[bool] = mapped_column(Boolean, nullable=False)
    page_manifest_json: Mapped[str] = mapped_column(Text, nullable=False)
    source_metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    source_payload_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    attachment_url: Mapped[str | None] = mapped_column(Text)
    pdf_fetch_status: Mapped[str] = mapped_column(String(30), nullable=False)
    pdf_content_type: Mapped[str | None] = mapped_column(String(100))
    pdf_bytes: Mapped[int | None] = mapped_column(Integer)
    pdf_sha256: Mapped[str | None] = mapped_column(String(64))
    pdf_storage_path: Mapped[str | None] = mapped_column(Text)
    development_contract: Mapped[str] = mapped_column(String(80), nullable=False)
    development_contract_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        CheckConstraint("content_chars >= 0", name="ck_source_archive_content_chars"),
        CheckConstraint(
            "pages_expected >= 1 AND pages_fetched >= 1",
            name="ck_source_archive_pages_positive",
        ),
        CheckConstraint(
            "source_time_authoritative = 0",
            name="ck_source_archive_candidate_time_not_authoritative",
        ),
    )


@event.listens_for(AnnouncementSourceArchive, "before_update")
def _prevent_source_archive_update(mapper, connection, target) -> None:
    raise ValueError("announcement source archives are immutable; append a new snapshot")


@event.listens_for(AnnouncementSourceArchive, "before_delete")
def _prevent_source_archive_delete(mapper, connection, target) -> None:
    raise ValueError("announcement source archives are append-only")


class AnnouncementMarketTarget(Base):
    """Retrospective market target for training; never a historical prediction."""

    __tablename__ = "announcement_market_targets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    announcement_id: Mapped[int] = mapped_column(
        ForeignKey("announcements.id"), nullable=False, index=True
    )
    horizon_sessions: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_date: Mapped[date] = mapped_column(Date, nullable=False)
    exit_date: Mapped[date] = mapped_column(Date, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=False)
    benchmark_code: Mapped[str] = mapped_column(String(20), nullable=False)
    stock_return: Mapped[float] = mapped_column(Float, nullable=False)
    benchmark_return: Mapped[float] = mapped_column(Float, nullable=False)
    excess_return: Mapped[float] = mapped_column(Float, nullable=False)
    actual_direction: Mapped[int] = mapped_column(Integer, nullable=False)
    outcome_available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    prices_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_role: Mapped[str | None] = mapped_column(String(40), index=True)
    split_contract: Mapped[str | None] = mapped_column(String(80))
    split_source_sha256: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint(
            "announcement_id",
            "horizon_sessions",
            name="uq_announcement_market_target_horizon",
        ),
        CheckConstraint(
            "horizon_sessions IN (1, 3, 5)", name="ck_market_target_horizon"
        ),
        CheckConstraint("exit_date >= entry_date", name="ck_market_target_date_order"),
        CheckConstraint(
            "actual_direction IN (-1, 0, 1)", name="ck_market_target_direction"
        ),
    )


# ── Causal prediction ledger ─────────────────────────────────


class ImpactPrediction(Base):
    """Immutable prediction made with information available at ``as_of`` only."""

    __tablename__ = "impact_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    announcement_id: Mapped[int] = mapped_column(
        ForeignKey("announcements.id"), nullable=False, index=True
    )
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    eligible_entry_date: Mapped[date] = mapped_column(
        Date, nullable=False, comment="保守口径：公告后的下一交易日"
    )
    horizon_sessions: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_direction: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="-1=利空, 0=拒判/中性, 1=利多"
    )
    bullish_probability: Mapped[float | None] = mapped_column(Float)
    predictor_version: Mapped[str] = mapped_column(String(80), nullable=False)
    taxonomy_version: Mapped[str] = mapped_column(String(20), nullable=False)
    classification_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    memory_cutoff: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    memory_key: Mapped[str] = mapped_column(String(160), nullable=False)
    memory_sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    historical_hit_rate: Mapped[float | None] = mapped_column(Float)
    historical_hit_rate_lower: Mapped[float | None] = mapped_column(Float)
    rank_score: Mapped[float | None] = mapped_column(Float)
    abstained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    abstain_reason: Mapped[str | None] = mapped_column(Text)
    payload_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint(
            "announcement_id",
            "predictor_version",
            "horizon_sessions",
            name="uq_impact_prediction_ann_version_horizon",
        ),
        CheckConstraint(
            "horizon_sessions IN (1, 3, 5)", name="ck_prediction_horizon"
        ),
        CheckConstraint(
            "predicted_direction IN (-1, 0, 1)", name="ck_prediction_direction"
        ),
        CheckConstraint(
            "bullish_probability IS NULL OR "
            "(bullish_probability >= 0 AND bullish_probability <= 1)",
            name="ck_prediction_probability",
        ),
        CheckConstraint(
            "historical_hit_rate IS NULL OR "
            "(historical_hit_rate >= 0 AND historical_hit_rate <= 1)",
            name="ck_prediction_hit_rate",
        ),
        CheckConstraint(
            "historical_hit_rate_lower IS NULL OR "
            "(historical_hit_rate_lower >= 0 AND historical_hit_rate_lower <= 1)",
            name="ck_prediction_hit_rate_lower",
        ),
    )


@event.listens_for(ImpactPrediction, "before_update")
def _prevent_prediction_update(mapper, connection, target) -> None:
    raise ValueError("impact predictions are immutable; append a new predictor version")


@event.listens_for(ImpactPrediction, "before_delete")
def _prevent_prediction_delete(mapper, connection, target) -> None:
    raise ValueError("impact predictions are an append-only audit ledger")


class PredictionOutcome(Base):
    """Observed market outcome appended only after its horizon has matured."""

    __tablename__ = "prediction_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("impact_predictions.id"), unique=True, nullable=False
    )
    entry_date: Mapped[date] = mapped_column(Date, nullable=False)
    exit_date: Mapped[date] = mapped_column(Date, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=False)
    benchmark_code: Mapped[str] = mapped_column(String(20), nullable=False)
    stock_return: Mapped[float] = mapped_column(Float, nullable=False)
    benchmark_return: Mapped[float] = mapped_column(Float, nullable=False)
    excess_return: Mapped[float] = mapped_column(Float, nullable=False)
    actual_direction: Mapped[int] = mapped_column(Integer, nullable=False)
    direction_hit: Mapped[bool] = mapped_column(Boolean, nullable=False)
    prices_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        CheckConstraint("exit_date >= entry_date", name="ck_outcome_date_order"),
        CheckConstraint(
            "entry_price > 0 AND exit_price > 0", name="ck_outcome_positive_prices"
        ),
        CheckConstraint(
            "actual_direction IN (-1, 0, 1)", name="ck_outcome_direction"
        ),
    )


class PredictionReflection(Base):
    """Post-outcome error analysis; it can influence only later predictions."""

    __tablename__ = "prediction_reflections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("impact_predictions.id"), nullable=False, index=True
    )
    outcome_id: Mapped[int] = mapped_column(
        ForeignKey("prediction_outcomes.id"), nullable=False
    )
    error_type: Mapped[str] = mapped_column(String(50), nullable=False)
    diagnosis: Mapped[str] = mapped_column(Text, nullable=False)
    lesson: Mapped[str] = mapped_column(Text, nullable=False)
    memory_key: Mapped[str] = mapped_column(String(160), nullable=False)
    reflector_version: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )


class PredictionMemory(Base):
    """Versioned accuracy snapshot built only from outcomes mature by ``as_of``."""

    __tablename__ = "prediction_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memory_key: Mapped[str] = mapped_column(String(160), nullable=False)
    horizon_sessions: Mapped[int] = mapped_column(Integer, nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    predictor_version: Mapped[str] = mapped_column(String(80), nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False)
    hit_rate: Mapped[float] = mapped_column(Float, nullable=False)
    wilson_lower: Mapped[float] = mapped_column(Float, nullable=False)
    wilson_upper: Mapped[float] = mapped_column(Float, nullable=False)
    mean_excess_return: Mapped[float] = mapped_column(Float, nullable=False)
    source_outcomes_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint(
            "memory_key",
            "horizon_sessions",
            "as_of",
            "predictor_version",
            name="uq_prediction_memory_snapshot",
        ),
        CheckConstraint("sample_count >= 0", name="ck_memory_sample_count"),
        CheckConstraint(
            "hit_count >= 0 AND hit_count <= sample_count", name="ck_memory_hit_count"
        ),
        CheckConstraint(
            "hit_rate >= 0 AND hit_rate <= 1", name="ck_memory_hit_rate"
        ),
        CheckConstraint(
            "wilson_lower >= 0 AND wilson_lower <= 1", name="ck_memory_wilson_lower"
        ),
        CheckConstraint(
            "wilson_upper >= 0 AND wilson_upper <= 1", name="ck_memory_wilson_upper"
        ),
    )


# ── Daily Reports ──────────────────────────────────────────────


class DailyReport(Base):
    __tablename__ = "daily_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_date: Mapped[date] = mapped_column(Date, nullable=False, comment="报告日期")
    industry_filter: Mapped[str | None] = mapped_column(String(50), comment="行业过滤（空=全市场）")
    report_title: Mapped[str | None] = mapped_column(String(200), comment="报告标题")
    report_content: Mapped[str | None] = mapped_column(Text, comment="完整 Markdown 报告")
    summary_text: Mapped[str | None] = mapped_column(Text, comment="摘要")
    high_impact_count: Mapped[int] = mapped_column(Integer, default=0)
    total_announcements: Mapped[int] = mapped_column(Integer, default=0)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("report_date", "industry_filter", name="uq_daily_reports_date_industry"),
    )

    def __repr__(self) -> str:
        return f"<DailyReport {self.report_date} [{self.high_impact_count} high]>"


# ── Pipeline Runs ──────────────────────────────────────────────


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, comment="UUID")
    run_type: Mapped[str] = mapped_column(String(20), nullable=False, comment="运行类型: full/fetch/preprocess/classify/extract/score/report")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running", comment="running/completed/failed")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    records_fetched: Mapped[int] = mapped_column(Integer, default=0)
    records_processed: Mapped[int] = mapped_column(Integer, default=0)
    records_failed: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    run_metadata: Mapped[str | None] = mapped_column(Text, comment="JSON: 配置快照/模型版本")

    __table_args__ = (
        CheckConstraint(
            "run_type IN ('full','fetch','preprocess','classify','extract','score','report')",
            name="ck_pipeline_run_type",
        ),
        CheckConstraint(
            "status IN ('running','completed','failed')",
            name="ck_pipeline_run_status",
        ),
    )

    def __repr__(self) -> str:
        return f"<PipelineRun {self.run_id} [{self.status}]>"

"""SQLAlchemy ORM 模型 — 7 张核心表。

表结构:
  companies       — 公司基础信息
  announcements   — 公告原文 + 处理状态
  classifications — 分类结果 (A-G + 子类别)
  extracted_fields— LLM 提取的结构化字段
  scores          — 影响评分五维度
  daily_reports   — 生成的每日报告
  pipeline_runs   — 管线运行记录
"""

from datetime import date, datetime

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
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
    major_category: Mapped[str] = mapped_column(String(2), nullable=False, comment="大类 A-G")
    sub_category: Mapped[str] = mapped_column(String(50), nullable=False, comment="子类别代码")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, comment="分类置信度")
    model_version: Mapped[str | None] = mapped_column(String(50), comment="模型版本")
    classified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # 关系
    announcement: Mapped["Announcement"] = relationship(back_populates="classification")

    def __repr__(self) -> str:
        return f"<Classification {self.major_category}/{self.sub_category} ({self.confidence:.2f})>"


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

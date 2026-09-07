"""SQLAlchemy 引擎和会话工厂。

MVP 使用 SQLite + WAL 模式，后期可切换 PostgreSQL。
"""

from collections.abc import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import config


def get_engine(db_url: str | None = None) -> Engine:
    """创建 SQLAlchemy 引擎。

    SQLite 使用 WAL 模式提升并发读性能。
    """
    url = db_url or config.database.url

    connect_args: dict = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}

    engine = create_engine(
        url,
        echo=config.database.echo,
        connect_args=connect_args,
        pool_pre_ping=True if not url.startswith("sqlite") else False,
    )

    # SQLite WAL 模式（启动时自动设置）
    if url.startswith("sqlite"):
        from sqlalchemy import event

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


# 全局引擎（模块级单例）
_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def _get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=_get_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _SessionLocal


def get_session() -> Generator[Session, None, None]:
    """获取数据库会话（生成器，用于 FastAPI Depends 或 context manager）。

    Usage:
        with next(get_session()) as session:
            ...
    或:
        session = next(get_session())
        try:
            ...
        finally:
            session.close()
    """
    factory = _get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


# ── 轻量迁移 ──────────────────────────────────────────────────
# 开发阶段没有 Alembic，新增列通过幂等 ALTER 补齐，重复运行安全。
# 新列必须是可空类型（SQLite ALTER TABLE ADD COLUMN 对已存在的表）。


def _ensure_column(engine: "Engine", table: str, column: str, ddl: str) -> None:
    """检查 SQLite 表是否缺列，缺则 ALTER 补齐。"""
    from sqlalchemy import text

    with engine.connect() as conn:
        existing = {
            row[1]
            for row in conn.execute(text(f"PRAGMA table_info({table})"))
        }
        if not existing:
            return
        if column not in existing:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))


def _run_lightweight_migrations(engine: "Engine") -> None:
    """新增列的幂等迁移。"""
    if engine.dialect.name != "sqlite":
        return

    columns = {
        "industry": "VARCHAR(50)",
        "industry_group": "VARCHAR(20)",
        "classification_source": "VARCHAR(30)",
        "rule_id": "VARCHAR(80)",
        "document_type": "VARCHAR(50)",
        "relevance": "VARCHAR(30)",
        "secondary_tags": "TEXT",
        "model_sub_category": "VARCHAR(50)",
        "model_confidence": "FLOAT",
        "model_margin": "FLOAT",
        "needs_review": "BOOLEAN",
        "review_status": "VARCHAR(20)",
        "review_reason": "TEXT",
        "review_note": "TEXT",
        "reviewed_by": "VARCHAR(100)",
        "reviewed_at": "DATETIME",
        "taxonomy_version": "VARCHAR(20)",
    }
    for column, ddl in columns.items():
        _ensure_column(engine, "classifications", column, ddl)

    target_columns = {
        "dataset_role": "VARCHAR(40)",
        "split_contract": "VARCHAR(80)",
        "split_source_sha256": "VARCHAR(64)",
    }
    for column, ddl in target_columns.items():
        _ensure_column(engine, "announcement_market_targets", column, ddl)

    company_columns = {
        "listing_date": "DATE",
        "delisting_date": "DATE",
        "list_status": "VARCHAR(2)",
        "universe_source": "VARCHAR(50)",
        "universe_as_of": "DATETIME",
    }
    for column, ddl in company_columns.items():
        _ensure_column(engine, "companies", column, ddl)

    # 历史结果只补充审计元数据，不改动原分类类别和置信度。
    from sqlalchemy import text

    from sqlalchemy import inspect

    if inspect(engine).has_table("classifications"):
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE classifications "
                "SET classification_source = COALESCE(classification_source, 'legacy'), "
                "review_status = COALESCE(review_status, 'pending'), "
                "needs_review = COALESCE(needs_review, 1), "
                "relevance = COALESCE(relevance, 'uncertain'), "
                "taxonomy_version = COALESCE(taxonomy_version, 'v1')"
            ))


def init_db() -> None:
    """创建所有表并补齐迁移（开发阶段使用；生产应通过 Alembic 迁移）。"""
    from src.database.models import Base

    engine = _get_engine()
    Base.metadata.create_all(bind=engine)
    _run_lightweight_migrations(engine)


def dispose_engine() -> None:
    """释放引擎连接池。"""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
        _engine = None
        _SessionLocal = None


def set_database_url(db_url: str) -> None:
    """重设数据库 URL（需在首次建连前调用）。

    供启动脚本 `--db-url` 或测试在 init_db()/get_session() 之前切换
    目标数据库。会先释放已存在的单例引擎。
    """
    dispose_engine()
    config.database.url = db_url

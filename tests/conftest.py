"""Pytest 共享 fixtures。"""

from datetime import date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.config import Config
from src.database.models import Base


@pytest.fixture(scope="function")
def db_session():
    """创建内存 SQLite 数据库会话（每个测试独立）。"""
    engine = create_engine("sqlite:///:memory:", echo=False)

    # 启用外键约束
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def set_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)

    with Session(engine) as session:
        yield session

    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def sample_company_data() -> dict:
    return {
        "stock_code": "000001.SZ",
        "stock_name": "平安银行",
        "exchange": "SZSE",
        "industry": "银行",
        "market_cap": 30_000_000,
        "annual_revenue": 15_000_000,
        "net_assets": 5_000_000,
        "is_tracked": True,
    }


@pytest.fixture
def sample_announcement_data(sample_company_data) -> dict:
    return {
        "announcement_id": "2024-001",
        "title": "2024年第一季度报告",
        "full_text": "公司2024年Q1实现营收100亿元，同比增长15%，净利润20亿元...",
        "published_date": date(2024, 4, 25),
        "source_url": "http://example.com/ann/2024-001",
        "processing_status": "fetched",
    }


@pytest.fixture
def sample_classification_data() -> dict:
    return {
        "major_category": "A",
        "sub_category": "earnings_q1",
        "confidence": 0.92,
        "model_version": "v1.0",
    }


@pytest.fixture
def sample_extracted_fields() -> list[dict]:
    return [
        {"field_name": "revenue", "field_value": "10000000000", "field_type": "numeric", "unit": "CNY", "confidence": 0.95},
        {"field_name": "revenue_yoy_pct", "field_value": "15", "field_type": "numeric", "unit": "pct", "confidence": 0.90},
        {"field_name": "net_profit", "field_value": "2000000000", "field_type": "numeric", "unit": "CNY", "confidence": 0.93},
    ]


@pytest.fixture
def sample_score_data() -> dict:
    return {
        "direction": 0.5,
        "magnitude": 0.6,
        "surprise": 0.4,
        "credibility": 0.85,
        "composite_score": 0.102,
    }

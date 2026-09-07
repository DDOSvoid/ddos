from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine

from src.database.models import Base
from src.prediction.fusion.ledger import (
    record_expert_signal,
    record_fused_stock_prediction,
)


def test_company_day_fusion_tables_are_created():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    assert {
        "expert_signal_records",
        "fused_stock_predictions",
        "fused_prediction_outcomes",
    } <= set(Base.metadata.tables)


def test_expert_signal_and_fused_prediction_are_append_only(db_session):
    cutoff = datetime(2024, 1, 3, 0, 30, tzinfo=UTC)
    signal = record_expert_signal(
        db_session,
        sample_id="sample-1",
        company_day_id="company-day-1",
        stock_code="000001.SZ",
        published_date=date(2024, 1, 2),
        prediction_as_of=cutoff,
        eligible_entry_date=date(2024, 1, 3),
        horizon_sessions=1,
        expert_name="text",
        expert_version="text-v1",
        expert_signal=0.4,
        quality_score=0.9,
        available=True,
        abstain_reason=None,
        data_as_of=datetime(2024, 1, 2, 10, tzinfo=UTC),
        evidence={"announcement_ids": [1]},
    )
    prediction = record_fused_stock_prediction(
        db_session,
        sample_id="sample-1",
        company_day_id="company-day-1",
        stock_code="000001.SZ",
        published_date=date(2024, 1, 2),
        prediction_as_of=cutoff,
        eligible_entry_date=date(2024, 1, 3),
        horizon_sessions=1,
        bullish_probability=0.7,
        expected_excess_return=0.02,
        risk_scale=0.01,
        rank_score=0.013,
        daily_rank=1,
        rank_percentile=1.0,
        eligible=True,
        abstain_reason=None,
        fusion_version="fusion-v1",
        component_contributions={"text": {"available": True, "value": 0.4}},
    )
    assert prediction.available_expert_count == 1
    signal.expert_signal = -0.4
    with pytest.raises(ValueError, match="immutable"):
        db_session.flush()
    db_session.rollback()

"""Causal prediction-memory and ranking tests."""

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import create_engine

from src.database.models import Announcement, Base, Company
from src.prediction.ledger import record_outcome, record_prediction
from src.prediction.memory import (
    MaturedObservation,
    build_accuracy_memory,
    rank_bullish_prediction,
)


def _observation(index: int, *, hit: bool, available_at: datetime) -> MaturedObservation:
    return MaturedObservation(
        prediction_id=f"p-{index:03d}",
        memory_key="subcategory:dividend",
        horizon_sessions=5,
        predicted_direction=1,
        excess_return=0.01 if hit else -0.01,
        outcome_available_at=available_at,
    )


def test_prediction_ledger_tables_are_created():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    assert {
        "impact_predictions",
        "prediction_outcomes",
        "prediction_reflections",
        "prediction_memories",
    } <= set(Base.metadata.tables)


def test_future_outcomes_are_excluded_from_memory():
    cutoff = datetime(2026, 8, 20, 8, tzinfo=UTC)
    observations = [
        _observation(1, hit=True, available_at=cutoff - timedelta(seconds=1)),
        _observation(2, hit=False, available_at=cutoff + timedelta(seconds=1)),
    ]
    memory = build_accuracy_memory(
        observations,
        memory_key="subcategory:dividend",
        horizon_sessions=5,
        as_of=cutoff,
    )
    assert memory.sample_count == 1
    assert memory.hit_count == 1
    assert memory.hit_rate == 1.0


def test_bullish_rank_uses_only_accuracy_lower_bound():
    cutoff = datetime(2026, 8, 20, 8, tzinfo=UTC)
    observations = [
        _observation(i, hit=i < 18, available_at=cutoff - timedelta(days=1))
        for i in range(20)
    ]
    memory = build_accuracy_memory(
        observations,
        memory_key="subcategory:dividend",
        horizon_sessions=5,
        as_of=cutoff,
    )
    rank = rank_bullish_prediction(1, memory)
    assert rank.eligible is True
    assert rank.score == memory.wilson_lower
    assert rank.degree == "强"


def test_bullish_rank_rejects_insufficient_history():
    cutoff = datetime(2026, 8, 20, 8, tzinfo=UTC)
    memory = build_accuracy_memory(
        [_observation(1, hit=True, available_at=cutoff)],
        memory_key="subcategory:dividend",
        horizon_sessions=5,
        as_of=cutoff,
    )
    rank = rank_bullish_prediction(1, memory)
    assert rank.eligible is False
    assert rank.score is None
    assert rank.degree == "样本不足"


def test_memory_requires_timezone_aware_cutoff():
    try:
        build_accuracy_memory(
            [],
            memory_key="subcategory:dividend",
            horizon_sessions=5,
            as_of=datetime(2026, 8, 20, 8),
        )
    except ValueError as exc:
        assert "timezone-aware" in str(exc)
    else:
        raise AssertionError("naive cutoff must be rejected")


def _announcement(db_session) -> Announcement:
    company = Company(
        stock_code="000001.SZ",
        stock_name="测试公司",
        exchange="SZSE",
    )
    db_session.add(company)
    db_session.flush()
    announcement = Announcement(
        company_id=company.id,
        announcement_id="causal-test-001",
        title="测试公告",
        published_date=date(2026, 8, 20),
    )
    db_session.add(announcement)
    db_session.flush()
    return announcement


def _empty_memory(as_of: datetime):
    return build_accuracy_memory(
        [],
        memory_key="subcategory:dividend",
        horizon_sessions=5,
        as_of=as_of,
    )


def test_recorded_prediction_is_immutable_and_abstains_without_history(db_session):
    announcement = _announcement(db_session)
    cutoff = datetime(2026, 8, 20, 9, tzinfo=UTC)
    prediction = record_prediction(
        db_session,
        announcement_id=announcement.id,
        text=announcement.title,
        classification_snapshot={"sub_category": "dividend"},
        as_of=cutoff,
        eligible_entry_date=date(2026, 8, 21),
        horizon_sessions=5,
        predicted_direction=1,
        bullish_probability=0.8,
        predictor_version="shadow-v1",
        taxonomy_version="v2",
        memory=_empty_memory(cutoff),
    )
    assert prediction.abstained is True
    assert prediction.rank_score is None

    prediction.predicted_direction = -1
    with pytest.raises(ValueError, match="immutable"):
        db_session.flush()
    db_session.rollback()


def test_outcome_cannot_use_a_price_before_eligible_entry(db_session):
    announcement = _announcement(db_session)
    cutoff = datetime(2026, 8, 20, 9, tzinfo=UTC)
    prediction = record_prediction(
        db_session,
        announcement_id=announcement.id,
        text=announcement.title,
        classification_snapshot={"sub_category": "dividend"},
        as_of=cutoff,
        eligible_entry_date=date(2026, 8, 21),
        horizon_sessions=5,
        predicted_direction=1,
        bullish_probability=0.8,
        predictor_version="shadow-v1",
        taxonomy_version="v2",
        memory=_empty_memory(cutoff),
    )
    with pytest.raises(ValueError, match="eligible entry"):
        record_outcome(
            db_session,
            prediction_id=prediction.prediction_id,
            entry_date=date(2026, 8, 20),
            exit_date=date(2026, 8, 21),
            entry_price=10.0,
            exit_price=10.2,
            benchmark_code="000300.SH",
            benchmark_return=0.01,
            evaluated_at=datetime(2026, 8, 21, 8, tzinfo=UTC),
            price_evidence={"stock": [10.0, 10.2]},
        )


def test_prediction_rejects_future_memory(db_session):
    announcement = _announcement(db_session)
    cutoff = datetime(2026, 8, 20, 9, tzinfo=UTC)
    with pytest.raises(ValueError, match="memory cutoff"):
        record_prediction(
            db_session,
            announcement_id=announcement.id,
            text=announcement.title,
            classification_snapshot={"sub_category": "dividend"},
            as_of=cutoff,
            eligible_entry_date=date(2026, 8, 21),
            horizon_sessions=5,
            predicted_direction=1,
            bullish_probability=0.8,
            predictor_version="shadow-v1",
            taxonomy_version="v2",
            memory=_empty_memory(cutoff + timedelta(seconds=1)),
        )

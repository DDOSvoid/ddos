"""Append-only persistence helpers for causal impact predictions."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

from src.database.models import (
    ImpactPrediction,
    PredictionMemory,
    PredictionOutcome,
    PredictionReflection,
)
from src.prediction.memory import AccuracyMemory, rank_bullish_prediction


def _utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def record_prediction(
    session: Session,
    *,
    announcement_id: int,
    text: str,
    classification_snapshot: dict,
    as_of: datetime,
    eligible_entry_date: date,
    horizon_sessions: int,
    predicted_direction: int,
    bullish_probability: float | None,
    predictor_version: str,
    taxonomy_version: str,
    memory: AccuracyMemory,
    minimum_samples: int = 20,
) -> ImpactPrediction:
    """Append a frozen prediction and its historical-accuracy snapshot."""
    prediction_time = _utc(as_of, "as_of")
    memory_time = _utc(memory.as_of, "memory.as_of")
    if memory_time > prediction_time:
        raise ValueError("memory cutoff is later than prediction time")
    if eligible_entry_date <= prediction_time.date():
        raise ValueError("eligible entry must be after the prediction date")
    if horizon_sessions not in {1, 3, 5}:
        raise ValueError("horizon_sessions must be 1, 3, or 5")
    if predicted_direction not in {-1, 0, 1}:
        raise ValueError("predicted_direction must be -1, 0, or 1")
    if bullish_probability is not None and not 0 <= bullish_probability <= 1:
        raise ValueError("bullish_probability must be between 0 and 1")

    rank = rank_bullish_prediction(
        predicted_direction,
        memory,
        minimum_samples=minimum_samples,
    )
    classification_json = json.dumps(
        classification_snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    input_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    prediction_id = str(uuid.uuid4())
    payload = {
        "prediction_id": prediction_id,
        "announcement_id": announcement_id,
        "as_of": prediction_time.isoformat(),
        "eligible_entry_date": eligible_entry_date.isoformat(),
        "horizon_sessions": horizon_sessions,
        "predicted_direction": predicted_direction,
        "bullish_probability": bullish_probability,
        "predictor_version": predictor_version,
        "taxonomy_version": taxonomy_version,
        "classification_snapshot": classification_snapshot,
        "input_sha256": input_hash,
        "memory_cutoff": memory_time.isoformat(),
        "memory_key": memory.memory_key,
        "memory_sample_count": memory.sample_count,
        "historical_hit_rate": memory.hit_rate,
        "historical_hit_rate_lower": memory.wilson_lower,
        "rank_score": rank.score,
        "abstained": not rank.eligible,
        "abstain_reason": None if rank.eligible else rank.reason,
    }
    prediction = ImpactPrediction(
        prediction_id=prediction_id,
        announcement_id=announcement_id,
        as_of=prediction_time,
        eligible_entry_date=eligible_entry_date,
        horizon_sessions=horizon_sessions,
        predicted_direction=predicted_direction,
        bullish_probability=bullish_probability,
        predictor_version=predictor_version,
        taxonomy_version=taxonomy_version,
        classification_snapshot=classification_json,
        input_sha256=input_hash,
        memory_cutoff=memory_time,
        memory_key=memory.memory_key,
        memory_sample_count=memory.sample_count,
        historical_hit_rate=memory.hit_rate,
        historical_hit_rate_lower=memory.wilson_lower,
        rank_score=rank.score,
        abstained=not rank.eligible,
        abstain_reason=None if rank.eligible else rank.reason,
        payload_sha256=_canonical_sha256(payload),
        created_at=prediction_time,
    )
    session.add(prediction)
    session.flush()
    return prediction


def record_outcome(
    session: Session,
    *,
    prediction_id: str,
    entry_date: date,
    exit_date: date,
    entry_price: float,
    exit_price: float,
    benchmark_code: str,
    benchmark_return: float,
    evaluated_at: datetime,
    price_evidence: object,
) -> PredictionOutcome:
    """Append a matured outcome; never rewrite the original prediction."""
    prediction = (
        session.query(ImpactPrediction)
        .filter_by(prediction_id=prediction_id)
        .one()
    )
    evaluation_time = _utc(evaluated_at, "evaluated_at")
    if entry_date < prediction.eligible_entry_date:
        raise ValueError("outcome entry predates the prediction's eligible entry")
    if exit_date < entry_date:
        raise ValueError("outcome exit predates entry")
    if evaluation_time.date() < exit_date:
        raise ValueError("outcome was evaluated before the exit session matured")
    if entry_price <= 0 or exit_price <= 0:
        raise ValueError("prices must be positive")

    stock_return = exit_price / entry_price - 1
    excess_return = stock_return - benchmark_return
    actual_direction = 1 if excess_return > 0 else -1 if excess_return < 0 else 0
    outcome = PredictionOutcome(
        prediction_id=prediction.id,
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_price,
        exit_price=exit_price,
        benchmark_code=benchmark_code,
        stock_return=stock_return,
        benchmark_return=benchmark_return,
        excess_return=excess_return,
        actual_direction=actual_direction,
        direction_hit=prediction.predicted_direction == actual_direction,
        prices_sha256=_canonical_sha256(price_evidence),
        evaluated_at=evaluation_time,
    )
    session.add(outcome)
    session.flush()
    return outcome


def record_reflection(
    session: Session,
    *,
    prediction_id: str,
    diagnosis: str,
    lesson: str,
    error_type: str,
    reflector_version: str,
    created_at: datetime,
) -> PredictionReflection:
    """Append reflection only after a wrong prediction has a matured outcome."""
    prediction = (
        session.query(ImpactPrediction)
        .filter_by(prediction_id=prediction_id)
        .one()
    )
    outcome = (
        session.query(PredictionOutcome)
        .filter_by(prediction_id=prediction.id)
        .one()
    )
    reflection_time = _utc(created_at, "created_at")
    outcome_time = _utc(outcome.evaluated_at, "outcome.evaluated_at")
    if outcome.direction_hit:
        raise ValueError("reflection is reserved for prediction errors")
    if reflection_time < outcome_time:
        raise ValueError("reflection predates the matured outcome")
    reflection = PredictionReflection(
        prediction_id=prediction.id,
        outcome_id=outcome.id,
        error_type=error_type,
        diagnosis=diagnosis,
        lesson=lesson,
        memory_key=prediction.memory_key,
        reflector_version=reflector_version,
        created_at=reflection_time,
    )
    session.add(reflection)
    session.flush()
    return reflection


def store_memory_snapshot(
    session: Session,
    memory: AccuracyMemory,
    *,
    predictor_version: str,
) -> PredictionMemory:
    snapshot = PredictionMemory(
        memory_key=memory.memory_key,
        horizon_sessions=memory.horizon_sessions,
        as_of=_utc(memory.as_of, "memory.as_of"),
        predictor_version=predictor_version,
        sample_count=memory.sample_count,
        hit_count=memory.hit_count,
        hit_rate=memory.hit_rate,
        wilson_lower=memory.wilson_lower,
        wilson_upper=memory.wilson_upper,
        mean_excess_return=memory.mean_excess_return,
        source_outcomes_sha256=memory.source_outcomes_sha256,
    )
    session.add(snapshot)
    session.flush()
    return snapshot

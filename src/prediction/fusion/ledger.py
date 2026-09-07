"""Append-only persistence for specialist and fused company-day predictions."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

from src.database.models import ExpertSignalRecord, FusedStockPrediction
from src.prediction.fusion.contract import EXPERT_NAMES, HORIZONS


def _utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def record_expert_signal(
    session: Session,
    *,
    sample_id: str,
    company_day_id: str,
    stock_code: str,
    published_date: date,
    prediction_as_of: datetime,
    eligible_entry_date: date,
    horizon_sessions: int,
    expert_name: str,
    expert_version: str,
    expert_signal: float | None,
    quality_score: float,
    available: bool,
    abstain_reason: str | None,
    data_as_of: datetime,
    evidence: object,
    bullish_probability: float | None = None,
    expected_excess_return: float | None = None,
    risk_scale: float | None = None,
) -> ExpertSignalRecord:
    prediction_time = _utc(prediction_as_of, "prediction_as_of")
    data_time = _utc(data_as_of, "data_as_of")
    if data_time > prediction_time:
        raise ValueError("expert data_as_of is later than prediction_as_of")
    if horizon_sessions not in HORIZONS or expert_name not in EXPERT_NAMES:
        raise ValueError("invalid expert name or horizon")
    if not 0 <= quality_score <= 1:
        raise ValueError("quality_score must be inside [0, 1]")
    if available and expert_signal is None:
        raise ValueError("available expert signal requires expert_signal")
    if not available and not abstain_reason:
        raise ValueError("unavailable expert signal requires abstain_reason")
    if eligible_entry_date <= published_date:
        raise ValueError("eligible entry must be after publication date")
    if bullish_probability is not None and not 0 <= bullish_probability <= 1:
        raise ValueError("bullish_probability must be inside [0, 1]")
    if risk_scale is not None and risk_scale < 0:
        raise ValueError("risk_scale must be non-negative")

    evidence_sha256 = _sha256(evidence)
    payload = {
        "sample_id": sample_id,
        "company_day_id": company_day_id,
        "stock_code": stock_code,
        "published_date": published_date.isoformat(),
        "prediction_as_of": prediction_time.isoformat(),
        "eligible_entry_date": eligible_entry_date.isoformat(),
        "horizon_sessions": horizon_sessions,
        "expert_name": expert_name,
        "expert_version": expert_version,
        "expert_signal": expert_signal,
        "quality_score": quality_score,
        "available": available,
        "abstain_reason": abstain_reason,
        "bullish_probability": bullish_probability,
        "expected_excess_return": expected_excess_return,
        "risk_scale": risk_scale,
        "data_as_of": data_time.isoformat(),
        "evidence_sha256": evidence_sha256,
    }
    row = ExpertSignalRecord(
        signal_id=str(uuid.uuid4()),
        sample_id=sample_id,
        company_day_id=company_day_id,
        stock_code=stock_code,
        published_date=published_date,
        prediction_as_of=prediction_time,
        eligible_entry_date=eligible_entry_date,
        horizon_sessions=horizon_sessions,
        expert_name=expert_name,
        expert_version=expert_version,
        expert_signal=expert_signal,
        quality_score=quality_score,
        available=available,
        abstain_reason=abstain_reason,
        bullish_probability=bullish_probability,
        expected_excess_return=expected_excess_return,
        risk_scale=risk_scale,
        data_as_of=data_time,
        evidence_sha256=evidence_sha256,
        payload_sha256=_sha256(payload),
        created_at=prediction_time,
    )
    session.add(row)
    session.flush()
    return row


def record_fused_stock_prediction(
    session: Session,
    *,
    sample_id: str,
    company_day_id: str,
    stock_code: str,
    published_date: date,
    prediction_as_of: datetime,
    eligible_entry_date: date,
    horizon_sessions: int,
    bullish_probability: float | None,
    expected_excess_return: float | None,
    risk_scale: float | None,
    rank_score: float | None,
    daily_rank: int | None,
    rank_percentile: float | None,
    eligible: bool,
    abstain_reason: str | None,
    fusion_version: str,
    component_contributions: dict[str, object],
) -> FusedStockPrediction:
    prediction_time = _utc(prediction_as_of, "prediction_as_of")
    if horizon_sessions not in HORIZONS:
        raise ValueError("invalid fusion horizon")
    if eligible_entry_date <= published_date:
        raise ValueError("eligible entry must be after publication date")
    if bullish_probability is not None and not 0 <= bullish_probability <= 1:
        raise ValueError("bullish_probability must be inside [0, 1]")
    if risk_scale is not None and risk_scale < 0:
        raise ValueError("risk_scale must be non-negative")
    if rank_percentile is not None and not 0 <= rank_percentile <= 1:
        raise ValueError("rank_percentile must be inside [0, 1]")
    if eligible and (expected_excess_return is None or risk_scale is None or rank_score is None):
        raise ValueError("eligible fusion prediction requires return, risk, and rank score")
    if not eligible and not abstain_reason:
        raise ValueError("ineligible fusion prediction requires abstain_reason")

    components_json = json.dumps(
        component_contributions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    input_sha256 = _sha256(component_contributions)
    payload = {
        "sample_id": sample_id,
        "company_day_id": company_day_id,
        "stock_code": stock_code,
        "published_date": published_date.isoformat(),
        "prediction_as_of": prediction_time.isoformat(),
        "eligible_entry_date": eligible_entry_date.isoformat(),
        "horizon_sessions": horizon_sessions,
        "bullish_probability": bullish_probability,
        "expected_excess_return": expected_excess_return,
        "risk_scale": risk_scale,
        "rank_score": rank_score,
        "daily_rank": daily_rank,
        "rank_percentile": rank_percentile,
        "eligible": eligible,
        "abstain_reason": abstain_reason,
        "fusion_version": fusion_version,
        "input_sha256": input_sha256,
    }
    available_experts = sum(
        bool((component_contributions.get(name) or {}).get("available"))
        for name in EXPERT_NAMES
    )
    row = FusedStockPrediction(
        prediction_id=str(uuid.uuid4()),
        sample_id=sample_id,
        company_day_id=company_day_id,
        stock_code=stock_code,
        published_date=published_date,
        prediction_as_of=prediction_time,
        eligible_entry_date=eligible_entry_date,
        horizon_sessions=horizon_sessions,
        bullish_probability=bullish_probability,
        expected_excess_return=expected_excess_return,
        risk_scale=risk_scale,
        rank_score=rank_score,
        daily_rank=daily_rank,
        rank_percentile=rank_percentile,
        eligible=eligible,
        abstain_reason=abstain_reason,
        available_expert_count=available_experts,
        fusion_version=fusion_version,
        component_contributions=components_json,
        input_sha256=input_sha256,
        payload_sha256=_sha256(payload),
        created_at=prediction_time,
    )
    session.add(row)
    session.flush()
    return row

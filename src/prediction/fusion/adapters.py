"""Adapters from legacy component OOF schemas to the specialist interface."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _identity(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sample_id",
        "company_day_id",
        "stock_code",
        "published_date",
        "prediction_as_of",
        "horizon_sessions",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"component OOF missing identity columns: {sorted(missing)}")
    output = frame.loc[:, sorted(required)].copy()
    prediction_time = pd.to_datetime(frame["prediction_as_of"], errors="raise", utc=True)
    output["prediction_as_of"] = prediction_time
    output["eligible_entry_date"] = prediction_time.dt.tz_convert(SHANGHAI).dt.date
    return output


def adapt_tabular_v2_oof(
    frame: pd.DataFrame,
    *,
    candidate: str,
    feature_set: str,
) -> pd.DataFrame:
    """Select one table candidate and expose it as a specialist signal."""
    required = {
        "candidate",
        "feature_set",
        "predicted_excess_return",
        "calibrated_bullish_probability",
        "predicted_absolute_excess_return",
        "component_available",
        "refusal_reason",
        "model_version",
        "data_as_of",
        "evidence_sha256",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"tabular v2 OOF missing columns: {sorted(missing)}")
    selected = frame.loc[
        (frame["candidate"] == candidate) & (frame["feature_set"] == feature_set)
    ].copy()
    if selected.empty:
        raise ValueError("selected tabular candidate has no rows")
    output = _identity(selected)
    output["expert_signal"] = selected["predicted_excess_return"].to_numpy()
    output["quality_score"] = selected["component_available"].astype(float).to_numpy()
    output["available"] = selected["component_available"].astype(bool).to_numpy()
    output["abstain_reason"] = selected["refusal_reason"].replace("", None).to_numpy()
    output["data_as_of"] = pd.to_datetime(
        selected["data_as_of"], errors="raise", utc=True
    ).to_numpy()
    output["evidence_sha256"] = selected["evidence_sha256"].astype(str).to_numpy()
    output["expert_version"] = selected["model_version"].astype(str).to_numpy()
    output["bullish_probability"] = selected[
        "calibrated_bullish_probability"
    ].to_numpy()
    output["expected_excess_return"] = selected["predicted_excess_return"].to_numpy()
    output["risk_scale"] = selected["predicted_absolute_excess_return"].to_numpy()
    return output


def adapt_timeseries_lstm_oof(
    frame: pd.DataFrame,
    *,
    model_version: str,
) -> pd.DataFrame:
    """Select one LSTM candidate and expose it as a specialist signal."""
    required = {
        "model_version",
        "bullish_probability",
        "expected_excess_return",
        "risk_scale",
        "sequence_available",
        "unavailable_reason",
        "data_as_of",
        "evidence_sha256",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"timeseries LSTM OOF missing columns: {sorted(missing)}")
    selected = frame.loc[frame["model_version"] == model_version].copy()
    if selected.empty:
        raise ValueError("selected timeseries candidate has no rows")
    output = _identity(selected)
    output["expert_signal"] = selected["expected_excess_return"].to_numpy()
    output["quality_score"] = selected["sequence_available"].astype(float).to_numpy()
    output["available"] = selected["sequence_available"].astype(bool).to_numpy()
    output["abstain_reason"] = selected["unavailable_reason"].to_numpy()
    output["data_as_of"] = pd.to_datetime(
        selected["data_as_of"], errors="raise", utc=True
    ).to_numpy()
    output["evidence_sha256"] = selected["evidence_sha256"].astype(str).to_numpy()
    output["expert_version"] = selected["model_version"].astype(str).to_numpy()
    output["bullish_probability"] = selected["bullish_probability"].to_numpy()
    output["expected_excess_return"] = selected["expected_excess_return"].to_numpy()
    output["risk_scale"] = selected["risk_scale"].to_numpy()
    return output

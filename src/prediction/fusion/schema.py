"""Canonical OOF interface shared by the three specialist branches."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src.prediction.fusion.contract import EXPERT_NAMES, ExpertFusionContract

IDENTITY_COLUMNS = [
    "sample_id",
    "company_day_id",
    "stock_code",
    "published_date",
    "prediction_as_of",
    "eligible_entry_date",
    "horizon_sessions",
]
REQUIRED_SIGNAL_COLUMNS = [
    "expert_signal",
    "quality_score",
    "available",
    "abstain_reason",
    "data_as_of",
    "evidence_sha256",
    "expert_version",
]
OPTIONAL_SIGNAL_COLUMNS = [
    "bullish_probability",
    "expected_excess_return",
    "risk_scale",
]


def _aware_utc(series: pd.Series, *, field: str) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="raise", utc=True)
    except Exception as error:
        raise ValueError(f"{field} must contain timezone-aware datetimes") from error


def validate_expert_signal_frame(
    frame: pd.DataFrame,
    *,
    expert_name: str,
    contract: ExpertFusionContract,
) -> pd.DataFrame:
    """Validate and normalize one expert's component OOF rows."""
    if expert_name not in EXPERT_NAMES or expert_name not in contract.expert_names:
        raise ValueError(f"unknown expert_name: {expert_name}")
    required = set(IDENTITY_COLUMNS + REQUIRED_SIGNAL_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{expert_name} signal frame missing columns: {sorted(missing)}")

    output = frame.copy()
    if output["sample_id"].duplicated().any():
        raise ValueError(f"{expert_name} signal frame has duplicate sample_id")
    if not output["horizon_sessions"].isin(contract.horizons_sessions).all():
        raise ValueError(f"{expert_name} signal frame contains an invalid horizon")
    output["prediction_as_of"] = _aware_utc(output["prediction_as_of"], field="prediction_as_of")
    output["data_as_of"] = _aware_utc(output["data_as_of"], field="data_as_of")
    if (output["data_as_of"] > output["prediction_as_of"]).any():
        raise ValueError(f"{expert_name} signal frame uses data after prediction_as_of")

    output["published_date"] = pd.to_datetime(output["published_date"], errors="raise").dt.date
    output["eligible_entry_date"] = pd.to_datetime(
        output["eligible_entry_date"], errors="raise"
    ).dt.date
    if (output["eligible_entry_date"] <= output["published_date"]).any():
        raise ValueError("eligible_entry_date must be strictly after published_date")
    prediction_dates = output["prediction_as_of"].dt.date
    if (
        (output["published_date"] > contract.development_end).any()
        or (output["eligible_entry_date"] > contract.development_end).any()
        or (prediction_dates > contract.development_end).any()
    ):
        raise PermissionError("component OOF contains data after the development cutoff")

    output["available"] = output["available"].astype(bool)
    quality = pd.to_numeric(output["quality_score"], errors="raise")
    if ((quality < 0) | (quality > 1)).any():
        raise ValueError("quality_score must be inside [0, 1]")
    output["quality_score"] = quality
    signal = pd.to_numeric(output["expert_signal"], errors="coerce")
    if signal.loc[output["available"]].isna().any():
        raise ValueError("available expert rows require expert_signal")
    if output.loc[~output["available"], "abstain_reason"].isna().any():
        raise ValueError("unavailable expert rows require abstain_reason")
    if output.loc[output["available"], "evidence_sha256"].isna().any():
        raise ValueError("available expert rows require evidence_sha256")
    output["expert_signal"] = signal
    output["expert_name"] = expert_name
    return output


def validate_development_labels(
    frame: pd.DataFrame,
    *,
    contract: ExpertFusionContract,
) -> pd.DataFrame:
    """Reject sealed roles before any outcome values enter fusion evaluation."""
    required = {
        "sample_id",
        "published_date",
        "dataset_role",
        "future_excess_return",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"development labels missing columns: {sorted(missing)}")
    output = frame.loc[:, sorted(required)].copy()
    if output["sample_id"].duplicated().any():
        raise ValueError("development labels contain duplicate sample_id")
    if not output["dataset_role"].eq(contract.development_dataset_role).all():
        raise PermissionError("development labels contain a sealed dataset role")
    output["published_date"] = pd.to_datetime(output["published_date"], errors="raise").dt.date
    if (output["published_date"] > contract.development_end).any():
        raise PermissionError("development labels cross the development cutoff")
    output["future_excess_return"] = pd.to_numeric(output["future_excess_return"], errors="raise")
    return output


def build_joint_expert_frame(
    expert_frames: Mapping[str, pd.DataFrame],
    *,
    contract: ExpertFusionContract,
) -> pd.DataFrame:
    """Outer-join component OOF rows and retain explicit missingness masks."""
    unknown = set(expert_frames) - set(contract.expert_names)
    if unknown:
        raise ValueError(f"unknown expert frames: {sorted(unknown)}")
    if not expert_frames:
        raise ValueError("at least one expert frame is required")

    joined: pd.DataFrame | None = None
    for expert_name in contract.expert_names:
        source = expert_frames.get(expert_name)
        if source is None:
            continue
        normalized = validate_expert_signal_frame(
            source, expert_name=expert_name, contract=contract
        )
        value_columns = REQUIRED_SIGNAL_COLUMNS + OPTIONAL_SIGNAL_COLUMNS
        present_values = [name for name in value_columns if name in normalized.columns]
        component = normalized.loc[:, IDENTITY_COLUMNS + present_values].rename(
            columns={name: f"{expert_name}__{name}" for name in present_values}
        )
        joined = (
            component
            if joined is None
            else joined.merge(component, on=IDENTITY_COLUMNS, how="outer", validate="one_to_one")
        )

    if joined is None:
        raise ValueError("no configured expert frame was supplied")

    available_columns = []
    for expert_name in contract.expert_names:
        available = f"{expert_name}__available"
        if available not in joined:
            joined[available] = False
        else:
            joined[available] = joined[available].fillna(False).astype(bool)
        available_columns.append(available)
        signal = f"{expert_name}__expert_signal"
        if signal not in joined:
            joined[signal] = 0.0
        else:
            joined[signal] = joined[signal].where(joined[available], 0.0)
        mask = f"{expert_name}__missing_mask"
        joined[mask] = (~joined[available]).astype(int)

    joined["available_expert_count"] = joined[available_columns].sum(axis=1).astype(int)
    joined["fusion_available"] = (
        joined["available_expert_count"] >= contract.minimum_available_experts
    )
    joined["fusion_abstain_reason"] = joined["fusion_available"].map(
        {True: None, False: "insufficient_available_experts"}
    )
    return joined.sort_values(
        ["eligible_entry_date", "horizon_sessions", "stock_code"]
    ).reset_index(drop=True)

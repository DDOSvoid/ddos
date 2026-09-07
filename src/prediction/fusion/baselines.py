"""Simple fusion baselines that precede trainable stacking and ranking models."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.prediction.fusion.contract import ExpertFusionContract
from src.prediction.fusion.ranking import rank_company_day_predictions
from src.prediction.fusion.schema import IDENTITY_COLUMNS


def _available_component_values(
    frame: pd.DataFrame,
    *,
    expert_name: str,
    value_name: str,
) -> pd.Series:
    available_column = f"{expert_name}__available"
    value_column = f"{expert_name}__{value_name}"
    if value_column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[value_column], errors="coerce")
    return values.where(frame[available_column].astype(bool))


def run_equal_weight_baseline(
    joint_frame: pd.DataFrame,
    *,
    contract: ExpertFusionContract,
) -> pd.DataFrame:
    """Average available optional expert heads and produce a daily rank baseline.

    This is only a comparison baseline. Specialist ``expert_signal`` values are
    not assumed to share units; the later trainable meta model owns their weights.
    """
    expected_columns = []
    risk_columns = []
    probability_columns = []
    for expert_name in contract.expert_names:
        expected_columns.append(
            _available_component_values(
                joint_frame, expert_name=expert_name, value_name="expected_excess_return"
            ).rename(expert_name)
        )
        risk_columns.append(
            _available_component_values(
                joint_frame, expert_name=expert_name, value_name="risk_scale"
            ).rename(expert_name)
        )
        probability_columns.append(
            _available_component_values(
                joint_frame, expert_name=expert_name, value_name="bullish_probability"
            ).rename(expert_name)
        )

    expected = pd.concat(expected_columns, axis=1)
    risk = pd.concat(risk_columns, axis=1)
    probability = pd.concat(probability_columns, axis=1)
    complete_component = expected.notna() & risk.notna()

    output = joint_frame.loc[:, IDENTITY_COLUMNS].copy()
    output["available_expert_count"] = complete_component.sum(axis=1).astype(int)
    output["fusion_available"] = output["available_expert_count"] >= max(
        contract.minimum_available_experts, 1
    )
    output["expected_excess_return"] = expected.mean(axis=1, skipna=True).where(
        output["fusion_available"]
    )
    output["risk_scale"] = risk.where(complete_component).mean(axis=1, skipna=True).where(
        output["fusion_available"]
    )
    output["bullish_probability"] = probability.where(complete_component).mean(
        axis=1, skipna=True
    ).where(output["fusion_available"])
    output["fusion_version"] = "equal-weight-available-diagnostic-heads-v1"
    output["abstain_reason"] = output["fusion_available"].map(
        {True: None, False: "no_complete_component_diagnostic_prediction"}
    )

    contributions = []
    for index in joint_frame.index:
        row = {}
        for expert_name in contract.expert_names:
            row[expert_name] = {
                "available": bool(complete_component.loc[index, expert_name]),
                "expected_excess_return": (
                    None
                    if pd.isna(expected.loc[index, expert_name])
                    else float(expected.loc[index, expert_name])
                ),
                "risk_scale": (
                    None
                    if pd.isna(risk.loc[index, expert_name])
                    else float(risk.loc[index, expert_name])
                ),
            }
        contributions.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    output["component_contributions"] = contributions
    return rank_company_day_predictions(
        output,
        risk_aversion=contract.risk_aversion,
        round_trip_cost_bps=contract.round_trip_cost_bps,
    )

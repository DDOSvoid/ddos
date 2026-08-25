"""Absolute and simple-baseline incremental gates for LSTM train OOF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.prediction.development_contract import CausalDevelopmentContract
from src.prediction.timeseries.lstm_contract import LstmTimeseriesContract
from src.prediction.timeseries.sequence_baselines import (
    _gate_report,
    _prediction_metrics,
)


@dataclass(frozen=True)
class LstmCandidateEvaluation:
    oof_predictions: pd.DataFrame
    report: dict[str, object]


def _identity_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    if frame.duplicated(["sample_id", "fold"]).any():
        raise ValueError("duplicate sample/fold identity in OOF comparison")
    return set(
        zip(
            frame["sample_id"].astype(str),
            frame["fold"].astype(str),
            strict=True,
        )
    )


def _simple_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    simple = frame.copy()
    simple["bullish_probability"] = simple["simple_bullish_probability"]
    simple["expected_excess_return"] = simple[
        "simple_expected_excess_return"
    ]
    simple["return_interval_lower"] = simple["simple_interval_lower"]
    simple["return_interval_upper"] = simple["simple_interval_upper"]
    return simple


def evaluate_lstm_candidate(
    lstm_oof: pd.DataFrame,
    simple_baseline_oof: pd.DataFrame,
    *,
    lookback_sessions: int,
    horizon_sessions: int,
    contract: LstmTimeseriesContract,
    development: CausalDevelopmentContract,
) -> LstmCandidateEvaluation:
    candidate = lstm_oof.loc[
        (lstm_oof["lookback_sessions"].astype(int) == lookback_sessions)
        & (lstm_oof["horizon_sessions"].astype(int) == horizon_sessions)
    ].copy()
    baseline = simple_baseline_oof.loc[
        (simple_baseline_oof["lookback_sessions"].astype(int) == lookback_sessions)
        & (simple_baseline_oof["horizon_sessions"].astype(int) == horizon_sessions)
    ].copy()
    if candidate.empty or baseline.empty:
        raise ValueError("LSTM/simple baseline comparison has an empty candidate")
    candidate_keys = _identity_keys(candidate)
    baseline_keys = _identity_keys(baseline)
    if candidate_keys != baseline_keys:
        raise ValueError("LSTM and simple baseline OOF identities/folds differ")

    baseline_columns = baseline.loc[
        :,
        [
            "sample_id",
            "fold",
            "target",
            "future_excess_return",
            "bullish_probability",
            "expected_excess_return",
            "return_interval_lower",
            "return_interval_upper",
            "constant_probability",
            "constant_expected_excess_return",
            "constant_interval_lower",
            "constant_interval_upper",
        ],
    ].rename(
        columns={
            "target": "simple_target",
            "future_excess_return": "simple_future_excess_return",
            "bullish_probability": "simple_bullish_probability",
            "expected_excess_return": "simple_expected_excess_return",
            "return_interval_lower": "simple_interval_lower",
            "return_interval_upper": "simple_interval_upper",
        }
    )
    scored = candidate.merge(
        baseline_columns,
        on=["sample_id", "fold"],
        how="inner",
        validate="one_to_one",
    )
    if not (
        scored["target"].to_numpy(dtype=int)
        == scored["simple_target"].to_numpy(dtype=int)
    ).all():
        raise ValueError("LSTM and simple baseline direction targets differ")
    if not np.allclose(
        scored["future_excess_return"].to_numpy(dtype=float),
        scored["simple_future_excess_return"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("LSTM and simple baseline return targets differ")

    pooled = _prediction_metrics(scored, contract=contract)
    fold_reports = []
    for fold_name, fold_frame in scored.groupby("fold", sort=True):
        fold_reports.append(
            {
                "fold": str(fold_name),
                **_prediction_metrics(fold_frame, contract=contract),
            }
        )
    absolute = _gate_report(
        pooled,
        fold_reports,
        contract=contract,
        development=development,
    )

    simple_frame = _simple_metric_frame(scored)
    simple_pooled = _prediction_metrics(simple_frame, contract=contract)
    simple_folds = {
        str(name): _prediction_metrics(group, contract=contract)
        for name, group in simple_frame.groupby("fold", sort=True)
    }
    lstm_folds = {str(item["fold"]): item for item in fold_reports}
    if set(simple_folds) != set(lstm_folds):
        raise ValueError("LSTM/simple baseline fold names differ")
    brier_improvements = {
        name: float(simple_folds[name]["brier_score"])
        - float(lstm_folds[name]["brier_score"])
        for name in simple_folds
    }
    return_mae_improvements = {
        name: float(simple_folds[name]["return_mae"])
        - float(lstm_folds[name]["return_mae"])
        for name in simple_folds
    }
    interval_score_improvements = {
        name: float(simple_folds[name]["interval_mean_score"])
        - float(lstm_folds[name]["interval_mean_score"])
        for name in simple_folds
    }
    incremental_checks = {
        "pooled_brier_lower": float(pooled["brier_score"])
        < float(simple_pooled["brier_score"]),
        "brier_lower_in_at_least_two_folds": sum(
            value > 0 for value in brier_improvements.values()
        )
        >= 2,
        "pooled_return_mae_lower": float(pooled["return_mae"])
        < float(simple_pooled["return_mae"]),
        "return_mae_lower_in_at_least_two_folds": sum(
            value > 0 for value in return_mae_improvements.values()
        )
        >= 2,
        "pooled_interval_score_lower": float(pooled["interval_mean_score"])
        < float(simple_pooled["interval_mean_score"]),
        "interval_score_lower_in_at_least_two_folds": sum(
            value > 0 for value in interval_score_improvements.values()
        )
        >= 2,
    }
    raw_brier = float(
        np.mean(
            np.square(
                scored["raw_bullish_probability"].to_numpy(dtype=float)
                - scored["target"].to_numpy(dtype=float)
            )
        )
    )
    calibration_check = float(pooled["brier_score"]) <= raw_brier + 1e-15
    train_gate_passed = bool(
        absolute["predictive_general_and_return_passed"]
        and absolute["interval_diagnostic_passed"]
        and calibration_check
        and all(incremental_checks.values())
    )
    return LstmCandidateEvaluation(
        oof_predictions=scored,
        report={
            "lookback_sessions": lookback_sessions,
            "horizon_sessions": horizon_sessions,
            "oof_samples": int(len(scored)),
            "pooled_metrics": pooled,
            "folds": fold_reports,
            "absolute_gates": absolute,
            "probability_calibration": {
                "raw_brier_score": raw_brier,
                "calibrated_brier_score": float(pooled["brier_score"]),
                "calibrated_brier_not_worse": calibration_check,
            },
            "simple_sequence_baseline_metrics": simple_pooled,
            "simple_sequence_increment": {
                "checks": incremental_checks,
                "passed": all(incremental_checks.values()),
                "fold_brier_improvements": brier_improvements,
                "fold_return_mae_improvements": return_mae_improvements,
                "fold_interval_score_improvements": interval_score_improvements,
            },
            "train_gate_passed_before_table_increment": train_gate_passed,
            "table_increment_evaluated": False,
            "decision": (
                "ELIGIBLE_FOR_TABLE_INCREMENT_REVIEW"
                if train_gate_passed
                else "KEEP_RESEARCH_ONLY_DO_NOT_FREEZE"
            ),
        },
    )

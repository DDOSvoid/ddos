"""Fold-local logistic baseline for causal market-state features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.prediction.development_contract import CausalDevelopmentContract
from src.prediction.timeseries.contract import TimeseriesModelContract
from src.prediction.timeseries.metrics import probability_metrics
from src.prediction.timeseries.walk_forward import iter_timeseries_expanding_frames


@dataclass(frozen=True)
class LogisticBaselineResult:
    selected_c_by_horizon: dict[int, float]
    selected_class_weight_by_horizon: dict[int, str]
    report: dict[str, object]
    oof_predictions: pd.DataFrame


def _pipeline(c_value: float, class_weight: str) -> Pipeline:
    sklearn_class_weight = None if class_weight == "none" else class_weight
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight=sklearn_class_weight,
                    max_iter=1_000,
                    solver="liblinear",
                    random_state=0,
                ),
            ),
        ]
    )


def _validate_feature_metadata(frame: pd.DataFrame) -> None:
    required = {
        "sample_id",
        "published_date",
        "prediction_as_of",
        "last_bar_trade_date",
        "last_bar_available_at",
        "dataset_role",
        "outcome_available_at",
        "horizon_sessions",
        "target",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"baseline frame missing columns: {sorted(missing)}")
    prediction = pd.to_datetime(frame["prediction_as_of"], utc=True, errors="raise")
    available = pd.to_datetime(frame["last_bar_available_at"], utc=True, errors="raise")
    if (available > prediction).any():
        raise ValueError("future bar metadata rejected before baseline fitting")
    published = pd.to_datetime(frame["published_date"], errors="raise").dt.date
    last_bar = pd.to_datetime(frame["last_bar_trade_date"], errors="raise").dt.date
    if any(bar >= event for bar, event in zip(last_bar, published, strict=True)):
        raise ValueError("same-day or future daily bar rejected before baseline fitting")


def _candidate_oof(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    c_value: float,
    class_weight: str,
    development: CausalDevelopmentContract,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    predictions = []
    fold_reports = []
    for fold in iter_timeseries_expanding_frames(frame, development=development):
        train_target = fold.train["target"].to_numpy(dtype=int)
        if len(np.unique(train_target)) < 2:
            raise ValueError(f"{fold.name} train target contains only one class")
        model = _pipeline(c_value, class_weight)
        model.fit(fold.train.loc[:, feature_columns], train_target)
        probability = model.predict_proba(
            fold.validation.loc[:, feature_columns]
        )[:, 1]
        constant = np.full(len(fold.validation), train_target.mean(), dtype=float)
        target = fold.validation["target"].to_numpy(dtype=int)
        fold_report = {
            "fold": fold.name,
            "train_samples": int(len(fold.train)),
            "validation_samples": int(len(fold.validation)),
            **probability_metrics(
                target,
                probability,
                constant_probability=constant,
            ),
        }
        fold_reports.append(fold_report)
        prediction_frame = fold.validation.loc[
            :, ["sample_id", "published_date", "horizon_sessions", "target"]
        ].copy()
        prediction_frame["fold"] = fold.name
        prediction_frame["probability"] = probability
        prediction_frame["constant_probability"] = constant
        predictions.append(prediction_frame)
    return pd.concat(predictions, ignore_index=True), fold_reports


def run_logistic_baseline_oof(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str],
    contract: TimeseriesModelContract,
    development: CausalDevelopmentContract,
    c_values: tuple[float, ...] | None = None,
    class_weights: tuple[str, ...] | None = None,
) -> LogisticBaselineResult:
    """Select a baseline using train-period OOF Brier score only."""
    _validate_feature_metadata(frame)
    features = tuple(feature_columns)
    if not features or len(features) != len(set(features)):
        raise ValueError("baseline feature columns must be non-empty and unique")
    for name in features:
        if name not in frame:
            raise ValueError(f"missing baseline feature: {name}")
        if not contract.feature_name_allowed(name):
            raise ValueError(f"target or future feature rejected: {name}")
    candidates_c = c_values or contract.logistic_c_values
    candidates_class_weight = class_weights or contract.logistic_class_weights
    if not candidates_c or any(value <= 0 for value in candidates_c):
        raise ValueError("logistic C candidates must be positive")
    if not candidates_class_weight or not set(candidates_class_weight) <= {
        "none",
        "balanced",
    }:
        raise ValueError("logistic class weights must be none and/or balanced")

    selected_c_by_horizon: dict[int, float] = {}
    selected_class_weight_by_horizon: dict[int, str] = {}
    selected_predictions = []
    horizon_reports: dict[str, object] = {}
    for horizon in contract.horizons_sessions:
        horizon_frame = frame.loc[frame["horizon_sessions"] == horizon].copy()
        if horizon_frame.empty:
            raise ValueError(f"no train rows for horizon {horizon}")
        candidates = []
        for class_weight in candidates_class_weight:
            for c_value in candidates_c:
                oof, folds = _candidate_oof(
                    horizon_frame,
                    feature_columns=features,
                    c_value=float(c_value),
                    class_weight=class_weight,
                    development=development,
                )
                pooled = probability_metrics(
                    oof["target"].to_numpy(dtype=int),
                    oof["probability"].to_numpy(dtype=float),
                    constant_probability=oof[
                        "constant_probability"
                    ].to_numpy(dtype=float),
                )
                candidates.append(
                    {
                        "c": float(c_value),
                        "class_weight": class_weight,
                        "pooled_metrics": pooled,
                        "folds": folds,
                        "oof": oof,
                    }
                )
        selected = min(
            candidates,
            key=lambda item: (
                item["pooled_metrics"]["brier_score"],
                item["class_weight"] != "none",
                item["c"],
            ),
        )
        selected_c = float(selected["c"])
        selected_class_weight = str(selected["class_weight"])
        selected_c_by_horizon[horizon] = selected_c
        selected_class_weight_by_horizon[horizon] = selected_class_weight
        oof = selected["oof"].copy()
        oof["selected_c"] = selected_c
        oof["selected_class_weight"] = selected_class_weight
        selected_predictions.append(oof)
        horizon_reports[str(horizon)] = {
            "selected_c": selected_c,
            "selected_class_weight": selected_class_weight,
            "selection_rule": "minimum pooled train-period OOF Brier score",
            "pooled_metrics": selected["pooled_metrics"],
            "folds": selected["folds"],
            "candidates": [
                {
                    "c": item["c"],
                    "class_weight": item["class_weight"],
                    "pooled_metrics": item["pooled_metrics"],
                    "folds": item["folds"],
                }
                for item in candidates
            ],
        }
    return LogisticBaselineResult(
        selected_c_by_horizon=selected_c_by_horizon,
        selected_class_weight_by_horizon=selected_class_weight_by_horizon,
        report={
            "contract": contract.contract,
            "contract_sha256": contract.source_sha256,
            "dataset_role": "train",
            "split_strategy": contract.split_strategy,
            "feature_columns": list(features),
            "preprocessing": "median imputer and standard scaler fit inside each fold",
            "horizons": horizon_reports,
        },
        oof_predictions=pd.concat(selected_predictions, ignore_index=True).sort_values(
            ["horizon_sessions", "published_date", "sample_id"], kind="mergesort"
        ).reset_index(drop=True),
    )

"""Fixed-candidate expanding-window experiments for causal tabular models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PROJECT_ROOT
from src.prediction.development_contract import CausalDevelopmentContract
from src.prediction.tabular.aggregates import INTRADAY_FEATURE_COLUMNS
from src.prediction.tabular.dataset import (
    CATEGORICAL_FEATURE_COLUMNS,
    DATASET_CONTRACT,
)
from src.prediction.tabular.walk_forward import iter_expanding_window_frames

DEFAULT_EXPERIMENT_PATH = PROJECT_ROOT / "config" / "tabular_experiment_v1.yaml"
EXPERIMENT_RESULT_CONTRACT = "tabular-internal-oof-result-v1"


@dataclass(frozen=True)
class TabularExperimentConfig:
    contract: str
    dataset_contract: str
    dataset_role: str
    seed: int
    classifier_threshold: float
    round_trip_cost_bps: float
    return_target: str
    direction_target: str
    risk_target: str
    return_horizons: tuple[int, ...]
    risk_horizons: tuple[int, ...]
    candidates: dict[str, dict[str, Any]]
    source_path: Path
    source_sha256: str


def load_tabular_experiment_config(
    path: Path = DEFAULT_EXPERIMENT_PATH,
) -> TabularExperimentConfig:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    config = TabularExperimentConfig(
        contract=str(raw["contract"]),
        dataset_contract=str(raw["dataset_contract"]),
        dataset_role=str(raw["dataset_role"]),
        seed=int(raw["seed"]),
        classifier_threshold=float(raw["classifier_threshold"]),
        round_trip_cost_bps=float(raw["round_trip_cost_bps"]),
        return_target=str(raw["tasks"]["return_target"]),
        direction_target=str(raw["tasks"]["direction_target"]),
        risk_target=str(raw["tasks"]["risk_target"]),
        return_horizons=tuple(int(value) for value in raw["tasks"]["return_horizons"]),
        risk_horizons=tuple(int(value) for value in raw["tasks"]["risk_horizons"]),
        candidates={str(name): dict(values) for name, values in raw["candidates"].items()},
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    if config.contract != "tabular-internal-experiment-v1":
        raise ValueError("unexpected tabular experiment contract")
    if config.dataset_contract != DATASET_CONTRACT or config.dataset_role != "train":
        raise ValueError("experiment must use the train-only tabular dataset")
    if raw.get("official_test_allowed") is not False:
        raise PermissionError("internal experiment cannot read official test")
    policy = raw["selection_policy"]
    if policy["random_split_allowed"]:
        raise ValueError("random tabular split is forbidden")
    if not policy["fit_preprocessors_on_fold_train_only"]:
        raise ValueError("fold-local preprocessing is mandatory")
    if not policy["choose_on_internal_oof_only"]:
        raise ValueError("candidate choice must use internal OOF only")
    if policy["official_test_read_before_freeze"]:
        raise PermissionError("official test must remain sealed before candidate freeze")
    if set(config.candidates) != {"linear", "lightgbm", "catboost"}:
        raise ValueError("fixed candidate families changed")
    if set(config.return_horizons) != {1, 3, 5} or set(config.risk_horizons) != {3, 5}:
        raise ValueError("experiment horizons changed")
    if not 0 < config.classifier_threshold < 1:
        raise ValueError("classifier threshold must be inside (0, 1)")
    return config


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return float("nan")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    margin = z * np.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return float((centre - margin) / denominator)


def _mean_ci_lower(values: np.ndarray, z: float = 1.96) -> float:
    if len(values) == 0:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    return float(values.mean() - z * values.std(ddof=1) / np.sqrt(len(values)))


def prediction_metrics(
    frame: pd.DataFrame,
    *,
    threshold: float,
    round_trip_cost_bps: float,
) -> dict:
    actual_return = frame["future_excess_return"].to_numpy(dtype=float)
    predicted_return = frame["predicted_excess_return"].to_numpy(dtype=float)
    actual_bullish = (actual_return > 0).astype(int)
    probability = frame["bullish_probability"].to_numpy(dtype=float)
    signals = probability >= threshold
    signal_returns = actual_return[signals]
    successes = int((signal_returns > 0).sum())
    signal_count = int(signals.sum())
    adjusted = signal_returns - round_trip_cost_bps / 10_000.0
    metrics = {
        "samples": len(frame),
        "return_mae": float(mean_absolute_error(actual_return, predicted_return)),
        "return_rmse": float(np.sqrt(np.mean(np.square(actual_return - predicted_return)))),
        "baseline_return_mae": float(
            mean_absolute_error(actual_return, frame["baseline_excess_return"])
        ),
        "baseline_return_rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        actual_return
                        - frame["baseline_excess_return"].to_numpy(dtype=float)
                    )
                )
            )
        ),
        "brier_score": float(brier_score_loss(actual_bullish, probability)),
        "baseline_brier_score": float(
            brier_score_loss(actual_bullish, frame["baseline_bullish_probability"])
        ),
        "bullish_signals": signal_count,
        "bullish_hits": successes,
        "bullish_hit_rate": successes / signal_count if signal_count else float("nan"),
        "bullish_wilson_lower": wilson_lower(successes, signal_count),
        "signal_mean_excess_return": (
            float(signal_returns.mean()) if signal_count else float("nan")
        ),
        "cost_adjusted_mean_excess_return_ci_lower": _mean_ci_lower(adjusted),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(actual_bullish, probability))
        if len(np.unique(actual_bullish)) == 2
        else float("nan")
    )
    if "future_realized_volatility" in frame and frame[
        "future_realized_volatility"
    ].notna().all():
        actual_risk = frame["future_realized_volatility"].to_numpy(dtype=float)
        predicted_risk = frame["predicted_realized_volatility"].to_numpy(dtype=float)
        metrics["risk_mae"] = float(mean_absolute_error(actual_risk, predicted_risk))
        metrics["risk_rmse"] = float(
            np.sqrt(np.mean(np.square(actual_risk - predicted_risk)))
        )
        metrics["baseline_risk_mae"] = float(
            mean_absolute_error(actual_risk, frame["baseline_realized_volatility"])
        )
        metrics["baseline_risk_rmse"] = float(
            np.sqrt(
                np.mean(
                    np.square(
                        actual_risk
                        - frame["baseline_realized_volatility"].to_numpy(dtype=float)
                    )
                )
            )
        )
    return metrics


def _linear_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    categorical = [name for name in CATEGORICAL_FEATURE_COLUMNS if name in feature_columns]
    numeric = [name for name in feature_columns if name not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )


def _prepare_categories(
    train: pd.DataFrame, validation: pd.DataFrame, *, family: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = train.copy()
    validation = validation.copy()
    categorical = [name for name in CATEGORICAL_FEATURE_COLUMNS if name in train]
    for name in categorical:
        train_values = train[name].fillna("__missing__").astype(str)
        validation_values = validation[name].fillna("__missing__").astype(str)
        if family == "lightgbm":
            categories = sorted(train_values.unique())
            train[name] = pd.Categorical(train_values, categories=categories)
            validation[name] = pd.Categorical(validation_values, categories=categories)
        else:
            train[name] = train_values
            validation[name] = validation_values
    return train, validation, categorical


def _fit_predict_family(
    family: str,
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    config: TabularExperimentConfig,
    fit_risk: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = train.loc[:, feature_columns]
    x_validation = validation.loc[:, feature_columns]
    y_return = train[config.return_target].to_numpy(dtype=float)
    y_direction = (y_return > 0).astype(int)
    y_risk = train[config.risk_target].to_numpy(dtype=float) if fit_risk else None
    parameters = config.candidates[family]
    if family == "linear":
        return_model = Pipeline(
            [
                ("preprocess", _linear_preprocessor(feature_columns)),
                ("model", Ridge(alpha=float(parameters["ridge_alpha"]))),
            ]
        )
        direction_model = Pipeline(
            [
                ("preprocess", _linear_preprocessor(feature_columns)),
                (
                    "model",
                    LogisticRegression(
                        C=float(parameters["logistic_c"]),
                        max_iter=int(parameters["logistic_max_iter"]),
                        random_state=config.seed,
                    ),
                ),
            ]
        )
        risk_model = (
            Pipeline(
                [
                    ("preprocess", _linear_preprocessor(feature_columns)),
                    ("model", Ridge(alpha=float(parameters["ridge_alpha"]))),
                ]
            )
            if fit_risk
            else None
        )
    elif family == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        x_train, x_validation, _ = _prepare_categories(
            x_train, x_validation, family=family
        )
        common = {
            **parameters,
            "random_state": config.seed,
            "n_jobs": 4,
            "verbosity": -1,
        }
        return_model = LGBMRegressor(objective="regression", **common)
        direction_model = LGBMClassifier(objective="binary", **common)
        risk_model = LGBMRegressor(objective="regression", **common) if fit_risk else None
    elif family == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        x_train, x_validation, categorical = _prepare_categories(
            x_train, x_validation, family=family
        )
        common = {
            **parameters,
            "random_seed": config.seed,
            "thread_count": 4,
            "verbose": False,
            "allow_writing_files": False,
            "cat_features": categorical,
        }
        return_model = CatBoostRegressor(loss_function="RMSE", **common)
        direction_model = CatBoostClassifier(loss_function="Logloss", **common)
        risk_model = CatBoostRegressor(loss_function="RMSE", **common) if fit_risk else None
    else:
        raise ValueError(f"unsupported tabular candidate: {family}")

    return_model.fit(x_train, y_return)
    direction_model.fit(x_train, y_direction)
    predicted_return = np.asarray(return_model.predict(x_validation), dtype=float)
    probability = np.asarray(direction_model.predict_proba(x_validation)[:, 1], dtype=float)
    if risk_model is None:
        predicted_risk = np.full(len(validation), np.nan)
    else:
        assert y_risk is not None
        risk_model.fit(x_train, y_risk)
        predicted_risk = np.maximum(
            np.asarray(risk_model.predict(x_validation), dtype=float), 0.0
        )
    return predicted_return, probability, predicted_risk


def run_internal_experiment(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    dataset_manifest: dict,
    development: CausalDevelopmentContract,
    config: TabularExperimentConfig,
) -> tuple[pd.DataFrame, dict]:
    if dataset_manifest.get("contract") != config.dataset_contract:
        raise ValueError("experiment dataset contract mismatch")
    if dataset_manifest.get("official_test_queried") is not False:
        raise PermissionError("experiment dataset does not prove official-test isolation")
    feature_columns = list(dataset_manifest["feature_columns"])
    joined = features.merge(
        labels,
        on=[
            "sample_id",
            "company_day_id",
            "stock_code",
            "published_date",
            "horizon_sessions",
            "dataset_role",
        ],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(features) or len(joined) != len(labels):
        raise ValueError("experiment feature-label join is incomplete")
    predictions = []
    fold_metrics = []
    for horizon in config.return_horizons:
        horizon_frame = joined.loc[joined["horizon_sessions"] == horizon].copy()
        for fold in iter_expanding_window_frames(
            horizon_frame,
            development=development,
        ):
            fit_risk = horizon in config.risk_horizons
            if fit_risk and (
                fold.train[config.risk_target].isna().any()
                or fold.validation[config.risk_target].isna().any()
            ):
                raise ValueError(f"risk target missing inside {fold.name} horizon {horizon}")
            for family in config.candidates:
                predicted_return, probability, predicted_risk = _fit_predict_family(
                    family,
                    train=fold.train,
                    validation=fold.validation,
                    feature_columns=feature_columns,
                    config=config,
                    fit_risk=fit_risk,
                )
                output = fold.validation.loc[
                    :, [
                        "sample_id",
                        "company_day_id",
                        "stock_code",
                        "published_date",
                        "horizon_sessions",
                        "outcome_available_at",
                        config.return_target,
                        config.risk_target,
                    ]
                ].copy()
                output["fold"] = fold.name
                output["candidate"] = family
                output["baseline_excess_return"] = float(
                    fold.train[config.return_target].median()
                )
                output["baseline_bullish_probability"] = float(
                    (fold.train[config.return_target] > 0).mean()
                )
                output["baseline_realized_volatility"] = (
                    float(fold.train[config.risk_target].median())
                    if fit_risk
                    else np.nan
                )
                output["predicted_excess_return"] = predicted_return
                output["bullish_probability"] = probability
                output["predicted_realized_volatility"] = predicted_risk
                output_metrics = prediction_metrics(
                    output,
                    threshold=config.classifier_threshold,
                    round_trip_cost_bps=config.round_trip_cost_bps,
                )
                fold_metrics.append(
                    {
                        "candidate": family,
                        "horizon_sessions": horizon,
                        "fold": fold.name,
                        **output_metrics,
                    }
                )
                predictions.append(output)
    oof = pd.concat(predictions, ignore_index=True)
    aggregate_metrics = []
    for (family, horizon), group in oof.groupby(
        ["candidate", "horizon_sessions"], sort=True
    ):
        aggregate_metrics.append(
            {
                "candidate": family,
                "horizon_sessions": int(horizon),
                **prediction_metrics(
                    group,
                    threshold=config.classifier_threshold,
                    round_trip_cost_bps=config.round_trip_cost_bps,
                ),
            }
        )
    report = {
        "contract": EXPERIMENT_RESULT_CONTRACT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_role": "train",
        "official_test_queried": False,
        "experiment_config_sha256": config.source_sha256,
        "dataset_feature_schema_sha256": dataset_manifest["feature_schema_sha256"],
        "candidates": list(config.candidates),
        "fold_metrics": fold_metrics,
        "aggregate_metrics": aggregate_metrics,
        "selection_status": "internal_oof_only_not_frozen",
    }
    return oof, report


def review_internal_experiment(report: dict) -> dict:
    """Apply the frozen predictive gates without touching any sealed partition."""
    if report.get("official_test_queried") is not False:
        raise PermissionError("cannot review an experiment that touched official test")
    fold_metrics = report["fold_metrics"]
    return_reviews = []
    risk_reviews = []
    for aggregate in report["aggregate_metrics"]:
        family = aggregate["candidate"]
        horizon = int(aggregate["horizon_sessions"])
        folds = [
            item
            for item in fold_metrics
            if item["candidate"] == family
            and int(item["horizon_sessions"]) == horizon
        ]
        gates = {
            "minimum_300_bullish_signals": aggregate["bullish_signals"] >= 300,
            "bullish_wilson_lower_above_50pct": aggregate[
                "bullish_wilson_lower"
            ]
            > 0.50,
            "at_least_two_folds_hit_rate_above_50pct": sum(
                item["bullish_hit_rate"] > 0.50 for item in folds
            )
            >= 2,
            "worst_fold_hit_rate_at_least_48pct": min(
                item["bullish_hit_rate"] for item in folds
            )
            >= 0.48,
            "cost_adjusted_return_ci_lower_above_zero": aggregate[
                "cost_adjusted_mean_excess_return_ci_lower"
            ]
            > 0.0,
            "brier_beats_fold_train_constant": aggregate["brier_score"]
            < aggregate["baseline_brier_score"],
        }
        return_reviews.append(
            {
                "candidate": family,
                "horizon_sessions": horizon,
                "passed": all(gates.values()),
                "gates": gates,
                "return_mae": aggregate["return_mae"],
                "baseline_return_mae": aggregate["baseline_return_mae"],
                "brier_score": aggregate["brier_score"],
                "baseline_brier_score": aggregate["baseline_brier_score"],
                "bullish_wilson_lower": aggregate["bullish_wilson_lower"],
                "cost_adjusted_mean_excess_return_ci_lower": aggregate[
                    "cost_adjusted_mean_excess_return_ci_lower"
                ],
            }
        )
        if "risk_mae" in aggregate:
            risk_fold_improvements = [
                item["risk_mae"] < item["baseline_risk_mae"] for item in folds
            ]
            risk_reviews.append(
                {
                    "candidate": family,
                    "horizon_sessions": horizon,
                    "risk_mae": aggregate["risk_mae"],
                    "baseline_risk_mae": aggregate["baseline_risk_mae"],
                    "risk_rmse": aggregate["risk_rmse"],
                    "baseline_risk_rmse": aggregate["baseline_risk_rmse"],
                    "folds_beating_baseline_mae": int(sum(risk_fold_improvements)),
                    "all_folds_beat_baseline_mae": all(risk_fold_improvements),
                    "research_pass": (
                        aggregate["risk_mae"] < aggregate["baseline_risk_mae"]
                        and all(risk_fold_improvements)
                    ),
                }
            )
    qualified = [
        item
        for item in return_reviews
        if item["passed"]
    ]
    risk_ranked = sorted(
        risk_reviews,
        key=lambda item: (item["horizon_sessions"], item["risk_mae"]),
    )
    return {
        "contract": "tabular-internal-selection-review-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "official_test_queried": False,
        "return_reviews": return_reviews,
        "qualified_return_candidates": qualified,
        "risk_reviews": risk_ranked,
        "decision": (
            "FREEZE_ELIGIBLE"
            if qualified
            else "KEEP_RESEARCH_ONLY_DO_NOT_FREEZE"
        ),
        "official_test_action": "DO_NOT_READ" if not qualified else "AWAIT_MANUAL_FREEZE",
        "interpretation": (
            "Risk regression may remain an internal research candidate, but it does not "
            "override failed return/direction gates or authorize an official-test read."
        ),
    }


def run_intraday_risk_ablation(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    dataset_manifest: dict,
    development: CausalDevelopmentContract,
    config: TabularExperimentConfig,
    full_report: dict,
) -> tuple[pd.DataFrame, dict]:
    """Measure CatBoost risk accuracy after removing every intraday input."""
    if (
        dataset_manifest.get("official_test_queried") is not False
        or full_report.get("official_test_queried") is not False
    ):
        raise PermissionError("intraday ablation is train-only")
    feature_columns = [
        name
        for name in dataset_manifest["feature_columns"]
        if name not in INTRADAY_FEATURE_COLUMNS
    ]
    if len(feature_columns) + len(INTRADAY_FEATURE_COLUMNS) != len(
        dataset_manifest["feature_columns"]
    ):
        raise ValueError("intraday ablation feature schema is incomplete")
    joined = features.merge(
        labels,
        on=[
            "sample_id",
            "company_day_id",
            "stock_code",
            "published_date",
            "horizon_sessions",
            "dataset_role",
        ],
        how="inner",
        validate="one_to_one",
    )
    from catboost import CatBoostRegressor

    outputs = []
    fold_metrics = []
    parameters = config.candidates["catboost"]
    for horizon in config.risk_horizons:
        horizon_frame = joined.loc[joined["horizon_sessions"] == horizon].copy()
        for fold in iter_expanding_window_frames(horizon_frame, development=development):
            x_train, x_validation, categorical = _prepare_categories(
                fold.train.loc[:, feature_columns],
                fold.validation.loc[:, feature_columns],
                family="catboost",
            )
            model = CatBoostRegressor(
                **parameters,
                random_seed=config.seed,
                thread_count=4,
                verbose=False,
                allow_writing_files=False,
                cat_features=categorical,
                loss_function="RMSE",
            )
            model.fit(x_train, fold.train[config.risk_target].to_numpy(dtype=float))
            predicted = np.maximum(
                np.asarray(model.predict(x_validation), dtype=float), 0.0
            )
            output = fold.validation.loc[
                :, [
                    "sample_id",
                    "horizon_sessions",
                    config.risk_target,
                ]
            ].copy()
            output["fold"] = fold.name
            output["candidate"] = "catboost_without_intraday"
            output["predicted_realized_volatility"] = predicted
            output["baseline_realized_volatility"] = float(
                fold.train[config.risk_target].median()
            )
            actual = output[config.risk_target].to_numpy(dtype=float)
            baseline = output["baseline_realized_volatility"].to_numpy(dtype=float)
            fold_metrics.append(
                {
                    "horizon_sessions": horizon,
                    "fold": fold.name,
                    "samples": len(output),
                    "risk_mae": float(mean_absolute_error(actual, predicted)),
                    "risk_rmse": float(
                        np.sqrt(np.mean(np.square(actual - predicted)))
                    ),
                    "baseline_risk_mae": float(mean_absolute_error(actual, baseline)),
                }
            )
            outputs.append(output)
    oof = pd.concat(outputs, ignore_index=True)
    comparisons = []
    for horizon in config.risk_horizons:
        group = oof.loc[oof["horizon_sessions"] == horizon]
        actual = group[config.risk_target].to_numpy(dtype=float)
        predicted = group["predicted_realized_volatility"].to_numpy(dtype=float)
        without_intraday_mae = float(mean_absolute_error(actual, predicted))
        full = next(
            item
            for item in full_report["aggregate_metrics"]
            if item["candidate"] == "catboost"
            and int(item["horizon_sessions"]) == horizon
        )
        full_folds = {
            item["fold"]: item
            for item in full_report["fold_metrics"]
            if item["candidate"] == "catboost"
            and int(item["horizon_sessions"]) == horizon
        }
        ablation_folds = {
            item["fold"]: item
            for item in fold_metrics
            if int(item["horizon_sessions"]) == horizon
        }
        fold_deltas = {
            name: ablation_folds[name]["risk_mae"] - full_folds[name]["risk_mae"]
            for name in sorted(full_folds)
        }
        comparisons.append(
            {
                "horizon_sessions": horizon,
                "full_feature_risk_mae": full["risk_mae"],
                "without_intraday_risk_mae": without_intraday_mae,
                "mae_improvement_from_intraday": without_intraday_mae
                - full["risk_mae"],
                "fold_mae_improvement_from_intraday": fold_deltas,
                "intraday_improves_all_folds": all(
                    value > 0 for value in fold_deltas.values()
                ),
            }
        )
    report = {
        "contract": "tabular-intraday-risk-ablation-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_role": "train",
        "official_test_queried": False,
        "candidate": "catboost",
        "removed_features": list(INTRADAY_FEATURE_COLUMNS),
        "fold_metrics_without_intraday": fold_metrics,
        "comparisons": comparisons,
        "interpretation_rule": (
            "Positive MAE improvement means the otherwise identical full model is better."
        ),
    }
    return oof, report

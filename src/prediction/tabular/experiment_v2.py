"""Calibrated expanding-window tabular-v2 research and group ablation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import PROJECT_ROOT
from src.prediction.development_contract import CausalDevelopmentContract
from src.prediction.tabular.catalog import (
    FEATURE_CATALOG_CONTRACT,
    V2_DATASET_CONTRACT,
    TabularFeatureCatalog,
)
from src.prediction.tabular.experiment import (
    TabularExperimentConfig,
    _fit_predict_family,
    _linear_preprocessor,
    _mean_ci_lower,
    _prepare_categories,
    wilson_lower,
)
from src.prediction.tabular.walk_forward import iter_expanding_window_frames

DEFAULT_EXPERIMENT_V2_PATH = PROJECT_ROOT / "config" / "tabular_experiment_v2.yaml"
EXPERIMENT_V2_RESULT_CONTRACT = "tabular-internal-oof-result-v2"


@dataclass(frozen=True)
class FeatureSetConfig:
    name: str
    include_groups: tuple[str, ...]
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class TabularExperimentV2Config:
    contract: str
    dataset_contract: str
    feature_catalog_contract: str
    output_contract: str
    development_contract: str
    dataset_role: str
    seed: int
    classifier_threshold: float
    round_trip_cost_bps: float
    return_target: str
    direction_target: str
    risk_target: str
    return_horizons: tuple[int, ...]
    risk_horizons: tuple[int, ...]
    calibration_tail_fraction: float
    minimum_inner_train_samples: int
    minimum_calibration_samples: int
    probability_clip_epsilon: float
    candidates: dict[str, dict[str, Any]]
    feature_sets: tuple[FeatureSetConfig, ...]
    source_path: Path
    source_sha256: str

    def as_v1_compatible(self) -> TabularExperimentConfig:
        return TabularExperimentConfig(
            contract=self.contract,
            dataset_contract=self.dataset_contract,
            dataset_role=self.dataset_role,
            seed=self.seed,
            classifier_threshold=self.classifier_threshold,
            round_trip_cost_bps=self.round_trip_cost_bps,
            return_target=self.return_target,
            direction_target=self.direction_target,
            risk_target=self.risk_target,
            return_horizons=self.return_horizons,
            risk_horizons=self.risk_horizons,
            candidates=self.candidates,
            source_path=self.source_path,
            source_sha256=self.source_sha256,
        )


def load_tabular_experiment_v2_config(
    path: Path = DEFAULT_EXPERIMENT_V2_PATH,
) -> TabularExperimentV2Config:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    config = TabularExperimentV2Config(
        contract=str(raw["contract"]),
        dataset_contract=str(raw["dataset_contract"]),
        feature_catalog_contract=str(raw["feature_catalog_contract"]),
        output_contract=str(raw["output_contract"]),
        development_contract=str(raw["development_contract"]),
        dataset_role=str(raw["dataset_role"]),
        seed=int(raw["seed"]),
        classifier_threshold=float(raw["classifier_threshold"]),
        round_trip_cost_bps=float(raw["round_trip_cost_bps"]),
        return_target=str(raw["tasks"]["return_target"]),
        direction_target=str(raw["tasks"]["direction_target"]),
        risk_target=str(raw["tasks"]["risk_target"]),
        return_horizons=tuple(
            int(value) for value in raw["tasks"]["horizons_sessions"]
        ),
        risk_horizons=tuple(
            int(value) for value in raw["tasks"]["horizons_sessions"]
        ),
        calibration_tail_fraction=float(
            raw["calibration"]["chronological_tail_fraction"]
        ),
        minimum_inner_train_samples=int(
            raw["calibration"]["minimum_inner_train_samples"]
        ),
        minimum_calibration_samples=int(
            raw["calibration"]["minimum_calibration_samples"]
        ),
        probability_clip_epsilon=float(
            raw["calibration"]["probability_clip_epsilon"]
        ),
        candidates={
            str(name): dict(values) for name, values in raw["candidates"].items()
        },
        feature_sets=tuple(
            FeatureSetConfig(
                name=str(name),
                include_groups=tuple(str(value) for value in item["include_groups"]),
                candidates=tuple(str(value) for value in item["candidates"]),
            )
            for name, item in raw["feature_sets"].items()
        ),
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    if config.contract != "tabular-internal-experiment-v2":
        raise ValueError("unexpected tabular v2 experiment contract")
    if config.dataset_contract != V2_DATASET_CONTRACT:
        raise ValueError("tabular v2 experiment must use train_v2")
    if config.feature_catalog_contract != FEATURE_CATALOG_CONTRACT:
        raise ValueError("tabular v2 experiment catalog contract mismatch")
    if config.output_contract != "tabular-component-oof-v2":
        raise ValueError("tabular v2 experiment output contract mismatch")
    if config.dataset_role != "train" or raw.get("official_test_allowed") is not False:
        raise PermissionError("tabular v2 experiment must remain train-only")
    if set(config.candidates) != {"linear", "lightgbm", "catboost"}:
        raise ValueError("tabular v2 candidate families changed")
    if set(config.return_horizons) != {1, 3, 5}:
        raise ValueError("tabular v2 horizons changed")
    if config.risk_target != "future_absolute_excess_return":
        raise ValueError("tabular v2 primary risk target changed")
    if not 0 < config.calibration_tail_fraction < 0.5:
        raise ValueError("calibration tail fraction must be inside (0, 0.5)")
    policy = raw["selection_policy"]
    forbidden = (
        policy["random_split_allowed"],
        not policy["fit_preprocessors_on_fold_train_only"],
        not policy["calibrator_uses_outer_train_only"],
        not policy["choose_on_internal_oof_only"],
        policy["official_test_read_before_freeze"],
    )
    if any(forbidden):
        raise PermissionError("tabular v2 selection policy violates causal isolation")
    return config


def chronological_calibration_split(
    frame: pd.DataFrame,
    *,
    tail_fraction: float,
    minimum_inner_train_samples: int,
    minimum_calibration_samples: int,
    timezone,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Split an outer-train frame into earlier fit data and a later calibration tail."""
    working = frame.copy()
    working["__published"] = pd.to_datetime(
        working["published_date"], errors="raise"
    ).dt.date
    dates = np.array(sorted(working["__published"].unique()))
    if len(dates) < 4:
        raise ValueError("too few dates for chronological calibration")
    position = max(1, int(np.floor(len(dates) * (1.0 - tail_fraction))))
    if position >= len(dates):
        raise ValueError("calibration cutoff leaves no tail")
    cutoff = dates[position]
    prior_date = dates[position - 1]
    maturity_cutoff = pd.Timestamp.combine(prior_date, time(23, 59, 59)).tz_localize(
        timezone
    )
    outcome = pd.to_datetime(working["outcome_available_at"], errors="raise", utc=True)
    inner_train = working.loc[
        (working["__published"] < cutoff) & (outcome <= maturity_cutoff.tz_convert("UTC"))
    ].drop(columns="__published")
    calibration = working.loc[working["__published"] >= cutoff].drop(
        columns="__published"
    )
    if len(inner_train) < minimum_inner_train_samples:
        raise ValueError("inner calibration train sample count is below contract")
    if len(calibration) < minimum_calibration_samples:
        raise ValueError("calibration tail sample count is below contract")
    if (inner_train["future_excess_return"] > 0).nunique() != 2:
        raise ValueError("inner calibration train has one direction class")
    if (calibration["future_excess_return"] > 0).nunique() != 2:
        raise ValueError("calibration tail has one direction class")
    return inner_train, calibration, str(cutoff)


def _fit_direction_probability(
    family: str,
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    config: TabularExperimentV2Config,
) -> np.ndarray:
    x_train = train.loc[:, feature_columns]
    x_validation = validation.loc[:, feature_columns]
    y = (train[config.return_target].to_numpy(dtype=float) > 0).astype(int)
    parameters = config.candidates[family]
    if family == "linear":
        model = Pipeline(
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
    elif family == "lightgbm":
        from lightgbm import LGBMClassifier

        x_train, x_validation, _ = _prepare_categories(
            x_train, x_validation, family=family
        )
        model = LGBMClassifier(
            objective="binary",
            **parameters,
            random_state=config.seed,
            n_jobs=4,
            verbosity=-1,
        )
    elif family == "catboost":
        from catboost import CatBoostClassifier

        x_train, x_validation, categorical = _prepare_categories(
            x_train, x_validation, family=family
        )
        model = CatBoostClassifier(
            loss_function="Logloss",
            **parameters,
            random_seed=config.seed,
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
            cat_features=categorical,
        )
    else:
        raise ValueError(f"unsupported tabular v2 candidate: {family}")
    model.fit(x_train, y)
    return np.asarray(model.predict_proba(x_validation)[:, 1], dtype=float)


def fit_platt_calibrator(
    raw_probability: np.ndarray,
    actual_bullish: np.ndarray,
    *,
    epsilon: float,
    seed: int,
) -> LogisticRegression:
    clipped = np.clip(np.asarray(raw_probability, dtype=float), epsilon, 1 - epsilon)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    calibrator.fit(logit, np.asarray(actual_bullish, dtype=int))
    return calibrator


def apply_platt_calibrator(
    calibrator: LogisticRegression,
    raw_probability: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    clipped = np.clip(np.asarray(raw_probability, dtype=float), epsilon, 1 - epsilon)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return np.asarray(calibrator.predict_proba(logit)[:, 1], dtype=float)


def prediction_metrics_v2(
    frame: pd.DataFrame,
    *,
    threshold: float,
    round_trip_cost_bps: float,
) -> dict:
    actual_return = frame["future_excess_return"].to_numpy(dtype=float)
    predicted_return = frame["predicted_excess_return"].to_numpy(dtype=float)
    actual_bullish = (actual_return > 0).astype(int)
    probability = frame["calibrated_bullish_probability"].to_numpy(dtype=float)
    raw_probability = frame["raw_bullish_probability"].to_numpy(dtype=float)
    signals = probability >= threshold
    signal_returns = actual_return[signals]
    successes = int((signal_returns > 0).sum())
    signal_count = int(signals.sum())
    adjusted = signal_returns - round_trip_cost_bps / 10_000.0
    actual_risk = frame["future_absolute_excess_return"].to_numpy(dtype=float)
    predicted_risk = frame["predicted_absolute_excess_return"].to_numpy(dtype=float)
    metrics = {
        "samples": len(frame),
        "return_mae": float(mean_absolute_error(actual_return, predicted_return)),
        "baseline_return_mae": float(
            mean_absolute_error(actual_return, frame["baseline_excess_return"])
        ),
        "risk_mae": float(mean_absolute_error(actual_risk, predicted_risk)),
        "baseline_risk_mae": float(
            mean_absolute_error(actual_risk, frame["baseline_absolute_excess_return"])
        ),
        "brier_score": float(brier_score_loss(actual_bullish, probability)),
        "raw_brier_score": float(brier_score_loss(actual_bullish, raw_probability)),
        "baseline_brier_score": float(
            brier_score_loss(actual_bullish, frame["baseline_bullish_probability"])
        ),
        "bullish_signals": signal_count,
        "bullish_hits": successes,
        "bullish_hit_rate": successes / signal_count if signal_count else float("nan"),
        "bullish_wilson_lower": wilson_lower(successes, signal_count),
        "cost_adjusted_mean_excess_return_ci_lower": _mean_ci_lower(adjusted),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(actual_bullish, probability))
        if len(np.unique(actual_bullish)) == 2
        else float("nan")
    )
    return metrics


def _stable_sha256(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def _feature_columns_for_set(
    feature_set: FeatureSetConfig,
    *,
    dataset_manifest: dict,
) -> list[str]:
    groups = dataset_manifest["feature_groups"]
    unknown = set(feature_set.include_groups) - set(groups)
    if unknown:
        raise ValueError(f"feature set contains unknown groups: {sorted(unknown)}")
    selected = {
        name for group in feature_set.include_groups for name in groups[group]
    }
    return [name for name in dataset_manifest["feature_columns"] if name in selected]


def run_internal_experiment_v2(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    dataset_manifest: dict,
    development: CausalDevelopmentContract,
    catalog: TabularFeatureCatalog,
    config: TabularExperimentV2Config,
) -> tuple[pd.DataFrame, dict]:
    if dataset_manifest.get("contract") != config.dataset_contract:
        raise ValueError("tabular v2 experiment dataset contract mismatch")
    if dataset_manifest.get("official_test_queried") is not False:
        raise PermissionError("tabular v2 dataset does not prove test isolation")
    if dataset_manifest.get("contracts", {}).get("catalog") != catalog.source_sha256:
        raise ValueError("tabular v2 dataset catalog hash mismatch")
    if config.development_contract != development.contract:
        raise ValueError("tabular v2 experiment development contract mismatch")
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
        raise ValueError("tabular v2 feature-label join is incomplete")
    if joined[config.risk_target].isna().any():
        raise ValueError("tabular v2 primary risk target is incomplete")

    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict] = []
    calibration_records: list[dict] = []
    compatible = config.as_v1_compatible()
    for horizon in config.return_horizons:
        horizon_frame = joined.loc[joined["horizon_sessions"] == horizon].copy()
        for fold in iter_expanding_window_frames(
            horizon_frame, development=development
        ):
            inner_train, calibration, calibration_start = chronological_calibration_split(
                fold.train,
                tail_fraction=config.calibration_tail_fraction,
                minimum_inner_train_samples=config.minimum_inner_train_samples,
                minimum_calibration_samples=config.minimum_calibration_samples,
                timezone=development.zone,
            )
            for feature_set in config.feature_sets:
                feature_columns = _feature_columns_for_set(
                    feature_set, dataset_manifest=dataset_manifest
                )
                feature_schema_sha256 = _stable_sha256(*feature_columns)
                for family in feature_set.candidates:
                    calibration_raw = _fit_direction_probability(
                        family,
                        train=inner_train,
                        validation=calibration,
                        feature_columns=feature_columns,
                        config=config,
                    )
                    calibration_actual = (
                        calibration[config.return_target].to_numpy(dtype=float) > 0
                    ).astype(int)
                    calibrator = fit_platt_calibrator(
                        calibration_raw,
                        calibration_actual,
                        epsilon=config.probability_clip_epsilon,
                        seed=config.seed,
                    )
                    predicted_return, raw_probability, predicted_risk = (
                        _fit_predict_family(
                            family,
                            train=fold.train,
                            validation=fold.validation,
                            feature_columns=feature_columns,
                            config=compatible,
                            fit_risk=True,
                        )
                    )
                    calibrated_probability = apply_platt_calibrator(
                        calibrator,
                        raw_probability,
                        epsilon=config.probability_clip_epsilon,
                    )
                    output = fold.validation.loc[
                        :,
                        [
                            "sample_id",
                            "company_day_id",
                            "stock_code",
                            "published_date",
                            "prediction_as_of",
                            "horizon_sessions",
                            "outcome_available_at",
                            "target_evidence_sha256",
                            config.return_target,
                            config.risk_target,
                        ],
                    ].copy()
                    output["fold"] = fold.name
                    output["feature_set"] = feature_set.name
                    output["candidate"] = family
                    output["baseline_excess_return"] = float(
                        fold.train[config.return_target].median()
                    )
                    output["baseline_bullish_probability"] = float(
                        (fold.train[config.return_target] > 0).mean()
                    )
                    output["baseline_absolute_excess_return"] = float(
                        fold.train[config.risk_target].median()
                    )
                    output["predicted_excess_return"] = predicted_return
                    output["raw_bullish_probability"] = raw_probability
                    output["calibrated_bullish_probability"] = calibrated_probability
                    output["predicted_absolute_excess_return"] = predicted_risk
                    output["component_available"] = True
                    output["refusal_reason"] = ""
                    output["model_version"] = (
                        f"{config.contract}:{feature_set.name}:{family}"
                    )
                    output["feature_schema_sha256"] = feature_schema_sha256
                    output["data_as_of"] = output["prediction_as_of"]
                    output["evidence_sha256"] = [
                        _stable_sha256(
                            sample_id,
                            evidence,
                            fold.name,
                            feature_set.name,
                            family,
                            feature_schema_sha256,
                            config.source_sha256,
                            catalog.source_sha256,
                        )
                        for sample_id, evidence in zip(
                            output["sample_id"],
                            output["target_evidence_sha256"],
                            strict=True,
                        )
                    ]
                    metrics = prediction_metrics_v2(
                        output,
                        threshold=config.classifier_threshold,
                        round_trip_cost_bps=config.round_trip_cost_bps,
                    )
                    fold_metrics.append(
                        {
                            "feature_set": feature_set.name,
                            "candidate": family,
                            "horizon_sessions": horizon,
                            "fold": fold.name,
                            **metrics,
                        }
                    )
                    calibration_records.append(
                        {
                            "feature_set": feature_set.name,
                            "candidate": family,
                            "horizon_sessions": horizon,
                            "fold": fold.name,
                            "calibration_start": calibration_start,
                            "inner_train_samples": len(inner_train),
                            "calibration_samples": len(calibration),
                            "calibrator_intercept": float(calibrator.intercept_[0]),
                            "calibrator_coefficient": float(calibrator.coef_[0, 0]),
                            "calibration_raw_brier": float(
                                brier_score_loss(calibration_actual, calibration_raw)
                            ),
                            "calibration_fitted_brier": float(
                                brier_score_loss(
                                    calibration_actual,
                                    apply_platt_calibrator(
                                        calibrator,
                                        calibration_raw,
                                        epsilon=config.probability_clip_epsilon,
                                    ),
                                )
                            ),
                        }
                    )
                    predictions.append(output)
    oof = pd.concat(predictions, ignore_index=True)
    aggregate_metrics = []
    for (feature_set, family, horizon), group in oof.groupby(
        ["feature_set", "candidate", "horizon_sessions"], sort=True
    ):
        aggregate_metrics.append(
            {
                "feature_set": feature_set,
                "candidate": family,
                "horizon_sessions": int(horizon),
                **prediction_metrics_v2(
                    group,
                    threshold=config.classifier_threshold,
                    round_trip_cost_bps=config.round_trip_cost_bps,
                ),
            }
        )
    report = {
        "contract": EXPERIMENT_V2_RESULT_CONTRACT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_role": "train",
        "official_test_queried": False,
        "output_contract": config.output_contract,
        "experiment_config_sha256": config.source_sha256,
        "feature_catalog_sha256": catalog.source_sha256,
        "dataset_feature_schema_sha256": dataset_manifest["feature_schema_sha256"],
        "primary_risk_target": config.risk_target,
        "candidates": list(config.candidates),
        "feature_sets": [item.name for item in config.feature_sets],
        "calibration_records": calibration_records,
        "fold_metrics": fold_metrics,
        "aggregate_metrics": aggregate_metrics,
        "selection_status": "internal_oof_only_not_frozen",
    }
    return oof, report


def _incremental_comparisons(report: dict) -> list[dict]:
    metrics = {
        (
            item["feature_set"],
            item["candidate"],
            int(item["horizon_sessions"]),
        ): item
        for item in report["aggregate_metrics"]
    }
    comparisons = []
    for horizon in (1, 3, 5):
        full = metrics.get(("full_v2", "catboost", horizon))
        base = metrics.get(("without_valuation_liquidity", "catboost", horizon))
        if full is None or base is None:
            continue
        comparisons.append(
            {
                "candidate": "catboost",
                "horizon_sessions": horizon,
                "added_group": "valuation_liquidity",
                "brier_improvement": base["brier_score"] - full["brier_score"],
                "return_mae_improvement": base["return_mae"] - full["return_mae"],
                "risk_mae_improvement": base["risk_mae"] - full["risk_mae"],
                "positive_means_full_v2_is_better": True,
            }
        )
    return comparisons


def review_internal_experiment_v2(report: dict) -> dict:
    if report.get("official_test_queried") is not False:
        raise PermissionError("cannot review a v2 experiment that touched official test")
    fold_metrics = report["fold_metrics"]
    reviews = []
    for aggregate in report["aggregate_metrics"]:
        if aggregate["feature_set"] != "full_v2":
            continue
        family = aggregate["candidate"]
        horizon = int(aggregate["horizon_sessions"])
        folds = [
            item
            for item in fold_metrics
            if item["feature_set"] == "full_v2"
            and item["candidate"] == family
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
            "calibration_does_not_worsen_brier": aggregate["brier_score"]
            <= aggregate["raw_brier_score"],
            "return_mae_beats_fold_train_median": aggregate["return_mae"]
            < aggregate["baseline_return_mae"],
            "risk_mae_beats_fold_train_median": aggregate["risk_mae"]
            < aggregate["baseline_risk_mae"],
            "risk_mae_improves_at_least_two_folds": sum(
                item["risk_mae"] < item["baseline_risk_mae"] for item in folds
            )
            >= 2,
        }
        reviews.append(
            {
                "feature_set": "full_v2",
                "candidate": family,
                "horizon_sessions": horizon,
                "passed": all(gates.values()),
                "gates": gates,
                "metrics": aggregate,
            }
        )
    qualified = [item for item in reviews if item["passed"]]
    return {
        "contract": "tabular-internal-selection-review-v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "official_test_queried": False,
        "candidate_reviews": reviews,
        "qualified_candidates": qualified,
        "incremental_valuation_liquidity": _incremental_comparisons(report),
        "decision": (
            "FREEZE_ELIGIBLE" if qualified else "KEEP_RESEARCH_ONLY_DO_NOT_FREEZE"
        ),
        "official_test_action": (
            "AWAIT_MANUAL_FREEZE" if qualified else "DO_NOT_READ"
        ),
    }

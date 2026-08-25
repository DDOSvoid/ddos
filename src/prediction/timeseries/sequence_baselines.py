"""Train-only expanding-window baselines for causal direct market sequences."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT
from src.prediction.development_contract import (
    CausalDevelopmentContract,
    load_causal_development_contract,
)
from src.prediction.timeseries.lstm_contract import (
    LstmTimeseriesContract,
    load_lstm_timeseries_contract,
)
from src.prediction.timeseries.metrics import (
    cost_adjusted_bullish_returns,
    probability_metrics,
)
from src.prediction.timeseries.sequence_artifacts import (
    _file_sha256,
    _write_json_atomic,
)
from src.prediction.timeseries.sequence_dataset import (
    assemble_direct_sequence_frames,
)
from src.prediction.timeseries.walk_forward import (
    iter_timeseries_expanding_frames,
)


@dataclass(frozen=True)
class DirectSequenceBaselineResult:
    oof_predictions: pd.DataFrame
    report: dict[str, object]


def pool_direct_sequence_features(
    sequences: np.ndarray,
    time_masks: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    statistics: tuple[str, ...],
) -> pd.DataFrame:
    """Pool only within each already-causal sequence; no cross-sample fitting."""
    values = np.asarray(sequences, dtype=np.float64)
    masks = np.asarray(time_masks, dtype=bool)
    if values.ndim != 3 or masks.shape != values.shape[:2]:
        raise ValueError("pooled baseline requires aligned [sample,time,feature] arrays")
    if values.shape[2] != len(feature_names):
        raise ValueError("pooled baseline feature names do not match sequence width")
    expected = ("last", "mean", "std", "minimum", "maximum", "slope")
    if statistics != expected:
        raise ValueError("unregistered pooled sequence statistics")
    if (~masks.any(axis=1)).any():
        raise ValueError("pooled baseline received an empty sequence")

    valid = masks[:, :, None] & np.isfinite(values)
    count = valid.sum(axis=1).astype(np.float64)
    safe_count = np.where(count > 0, count, 1.0)
    observed = np.where(valid, values, 0.0)
    total = observed.sum(axis=1)
    mean = total / safe_count
    centred = np.where(valid, values - mean[:, None, :], 0.0)
    std = np.sqrt(np.square(centred).sum(axis=1) / safe_count)

    minimum = np.where(valid, values, np.inf).min(axis=1)
    maximum = np.where(valid, values, -np.inf).max(axis=1)
    minimum[count == 0] = np.nan
    maximum[count == 0] = np.nan
    mean[count == 0] = np.nan
    std[count == 0] = np.nan

    positions = np.arange(values.shape[1], dtype=np.float64)[None, :, None]
    last_indices = np.where(valid, positions, -1.0).max(axis=1).astype(int)
    safe_last_indices = np.maximum(last_indices, 0)
    last = np.take_along_axis(
        values,
        safe_last_indices[:, None, :],
        axis=1,
    )[:, 0, :]
    last[last_indices < 0] = np.nan

    sum_x = np.where(valid, positions, 0.0).sum(axis=1)
    sum_x2 = np.where(valid, np.square(positions), 0.0).sum(axis=1)
    sum_xy = np.where(valid, positions * values, 0.0).sum(axis=1)
    denominator = count * sum_x2 - np.square(sum_x)
    numerator = count * sum_xy - sum_x * total
    slope = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    slope[count == 0] = np.nan

    arrays = {
        "last": last,
        "mean": mean,
        "std": std,
        "minimum": minimum,
        "maximum": maximum,
        "slope": slope,
    }
    columns = {
        f"{feature_name}__{statistic}": arrays[statistic][:, feature_index]
        for feature_index, feature_name in enumerate(feature_names)
        for statistic in statistics
    }
    return pd.DataFrame(columns, dtype=np.float32)


def _classification_pipeline(contract: LstmTimeseriesContract) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=contract.baseline_logistic_c,
                    class_weight=None,
                    max_iter=1_000,
                    solver="liblinear",
                    random_state=contract.baseline_random_state,
                ),
            ),
        ]
    )


def _regression_pipeline(contract: LstmTimeseriesContract) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=contract.baseline_ridge_alpha)),
        ]
    )


def _mean_ci_lower(values: np.ndarray, z: float = 1.96) -> float:
    observed = np.asarray(values, dtype=float)
    if len(observed) == 0:
        return float("nan")
    if len(observed) == 1:
        return float(observed[0])
    return float(
        observed.mean() - z * observed.std(ddof=1) / np.sqrt(len(observed))
    )


def _prediction_metrics(
    frame: pd.DataFrame,
    *,
    contract: LstmTimeseriesContract,
) -> dict[str, object]:
    target = frame["target"].to_numpy(dtype=int)
    probability = frame["bullish_probability"].to_numpy(dtype=float)
    constant_probability = frame["constant_probability"].to_numpy(dtype=float)
    metrics = probability_metrics(
        target,
        probability,
        bullish_threshold=contract.bullish_threshold,
        constant_probability=constant_probability,
    )
    actual_return = frame["future_excess_return"].to_numpy(dtype=float)
    expected_return = frame["expected_excess_return"].to_numpy(dtype=float)
    constant_return = frame["constant_expected_excess_return"].to_numpy(dtype=float)
    lower = frame["return_interval_lower"].to_numpy(dtype=float)
    upper = frame["return_interval_upper"].to_numpy(dtype=float)
    constant_lower = frame["constant_interval_lower"].to_numpy(dtype=float)
    constant_upper = frame["constant_interval_upper"].to_numpy(dtype=float)
    adjusted = cost_adjusted_bullish_returns(
        actual_return,
        probability,
        bullish_threshold=contract.bullish_threshold,
        round_trip_cost_bps=contract.round_trip_cost_bps,
    )
    interval_coverage = np.mean((actual_return >= lower) & (actual_return <= upper))
    constant_coverage = np.mean(
        (actual_return >= constant_lower) & (actual_return <= constant_upper)
    )
    alpha = 1.0 - (
        contract.interval_quantiles[1] - contract.interval_quantiles[0]
    )
    interval_score = (
        upper
        - lower
        + (2.0 / alpha) * np.maximum(lower - actual_return, 0.0)
        + (2.0 / alpha) * np.maximum(actual_return - upper, 0.0)
    )
    constant_interval_score = (
        constant_upper
        - constant_lower
        + (2.0 / alpha) * np.maximum(constant_lower - actual_return, 0.0)
        + (2.0 / alpha) * np.maximum(actual_return - constant_upper, 0.0)
    )
    metrics.update(
        {
            "return_mae": float(mean_absolute_error(actual_return, expected_return)),
            "constant_median_return_mae": float(
                mean_absolute_error(actual_return, constant_return)
            ),
            "return_mae_improvement_over_constant": float(
                mean_absolute_error(actual_return, constant_return)
                - mean_absolute_error(actual_return, expected_return)
            ),
            "interval_nominal_coverage": float(
                contract.interval_quantiles[1] - contract.interval_quantiles[0]
            ),
            "interval_coverage": float(interval_coverage),
            "interval_mean_width": float(np.mean(upper - lower)),
            "constant_interval_coverage": float(constant_coverage),
            "constant_interval_mean_width": float(
                np.mean(constant_upper - constant_lower)
            ),
            "interval_mean_score": float(np.mean(interval_score)),
            "constant_interval_mean_score": float(
                np.mean(constant_interval_score)
            ),
            "cost_adjusted_mean_excess_return_ci_lower": _mean_ci_lower(adjusted),
        }
    )
    return metrics


def _candidate_oof(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    lookback_sessions: int,
    horizon_sessions: int,
    contract: LstmTimeseriesContract,
    development: CausalDevelopmentContract,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    predictions: list[pd.DataFrame] = []
    fold_reports: list[dict[str, object]] = []
    lower_quantile, upper_quantile = contract.interval_quantiles
    for fold in iter_timeseries_expanding_frames(frame, development=development):
        x_train = fold.train.loc[:, feature_columns]
        x_validation = fold.validation.loc[:, feature_columns]
        target_train = fold.train["target"].to_numpy(dtype=int)
        return_train = fold.train["excess_return"].to_numpy(dtype=float)
        if len(np.unique(target_train)) != 2:
            raise ValueError(f"{fold.name} baseline train target has one class")

        classifier = _classification_pipeline(contract)
        regressor = _regression_pipeline(contract)
        classifier.fit(x_train, target_train)
        regressor.fit(x_train, return_train)
        probability = classifier.predict_proba(x_validation)[:, 1]
        expected_return = regressor.predict(x_validation)
        train_residual = return_train - regressor.predict(x_train)
        residual_lower, residual_upper = np.quantile(
            train_residual,
            [lower_quantile, upper_quantile],
        )

        constant_probability = float(target_train.mean())
        constant_return = float(np.median(return_train))
        constant_lower, constant_upper = np.quantile(
            return_train,
            [lower_quantile, upper_quantile],
        )
        output = fold.validation.loc[
            :,
            [
                "sample_id",
                "company_day_id",
                "stock_code",
                "published_date",
                "prediction_as_of",
                "outcome_available_at",
                "horizon_sessions",
                "target",
                "future_excess_return",
            ],
        ].copy()
        output["fold"] = fold.name
        output["lookback_sessions"] = lookback_sessions
        output["model_family"] = "pooled_sequence_logistic_ridge"
        output["bullish_probability"] = probability
        output["expected_excess_return"] = expected_return
        output["return_interval_lower"] = expected_return + residual_lower
        output["return_interval_upper"] = expected_return + residual_upper
        output["risk_scale"] = (
            output["return_interval_upper"] - output["return_interval_lower"]
        ) / 2.0
        output["constant_probability"] = constant_probability
        output["constant_expected_excess_return"] = constant_return
        output["constant_interval_lower"] = constant_lower
        output["constant_interval_upper"] = constant_upper
        fold_report = {
            "fold": fold.name,
            "train_samples": int(len(fold.train)),
            "validation_samples": int(len(output)),
            **_prediction_metrics(output, contract=contract),
        }
        fold_reports.append(fold_report)
        predictions.append(output)
    return pd.concat(predictions, ignore_index=True), fold_reports


def _gate_report(
    pooled: dict[str, object],
    folds: list[dict[str, object]],
    *,
    contract: LstmTimeseriesContract,
    development: CausalDevelopmentContract,
) -> dict[str, object]:
    general = development.module_gates["predictive_general"]
    fold_hit_rates = [
        float(item["bullish_hit_rate"])
        if item["bullish_hit_rate"] is not None
        else 0.0
        for item in folds
    ]
    checks = {
        "minimum_oof_bullish_signals": int(pooled["bullish_signals"])
        >= int(general["minimum_oof_bullish_signals_per_horizon"]),
        "bullish_wilson_lower_above_half": float(
            pooled["bullish_hit_rate_wilson_95pct"][0]
        )
        > float(general["bullish_wilson_lower_must_exceed"]),
        "at_least_two_folds_above_half": sum(value > 0.5 for value in fold_hit_rates)
        >= int(general["minimum_folds_with_bullish_hit_rate_above_half"]),
        "worst_fold_at_least_48pct": min(fold_hit_rates)
        >= float(general["minimum_worst_fold_bullish_hit_rate"]),
        "brier_beats_constant": float(pooled["brier_score"])
        < float(pooled["constant_base_rate_brier_score"]),
        "cost_adjusted_return_ci_lower_above_zero": float(
            pooled["cost_adjusted_mean_excess_return_ci_lower"]
        )
        > float(
            general["cost_adjusted_mean_excess_return_ci_lower_must_exceed"]
        ),
        "return_mae_beats_fold_train_median": float(pooled["return_mae"])
        < float(pooled["constant_median_return_mae"]),
    }
    nominal = float(pooled["interval_nominal_coverage"])
    interval_diagnostic = {
        "coverage_error_not_worse_than_constant": abs(
            float(pooled["interval_coverage"]) - nominal
        )
        <= abs(float(pooled["constant_interval_coverage"]) - nominal),
        "mean_width_lower_than_constant": float(pooled["interval_mean_width"])
        < float(pooled["constant_interval_mean_width"]),
    }
    return {
        "predictive_general_and_return_passed": all(checks.values()),
        "checks": checks,
        "interval_diagnostic": interval_diagnostic,
        "interval_diagnostic_passed": all(interval_diagnostic.values()),
    }


def run_direct_sequence_baseline_oof(
    sequences: np.ndarray,
    time_masks: np.ndarray,
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    contract: LstmTimeseriesContract,
    development: CausalDevelopmentContract,
) -> DirectSequenceBaselineResult:
    """Report every preregistered lookback without outer-OOF hyperparameter search."""
    predictions: list[pd.DataFrame] = []
    candidates: dict[str, object] = {}
    sample_sets: dict[int, dict[int, set[str]]] = {
        horizon: {} for horizon in contract.horizons_sessions
    }
    for lookback in contract.lookback_candidates_sessions:
        for horizon in contract.horizons_sessions:
            assembled = assemble_direct_sequence_frames(
                sequences,
                time_masks,
                metadata,
                labels,
                horizon_sessions=horizon,
                lookback_sessions=lookback,
                contract=contract,
            )
            pooled_features = pool_direct_sequence_features(
                assembled.sequences,
                assembled.time_masks,
                feature_names=contract.sequence_columns,
                statistics=contract.baseline_statistics,
            )
            feature_columns = tuple(pooled_features.columns)
            frame = pd.concat(
                [
                    assembled.metadata.reset_index(drop=True),
                    pooled_features.reset_index(drop=True),
                ],
                axis=1,
            )
            sample_sets[horizon][lookback] = set(frame["sample_id"].astype(str))
            oof, fold_reports = _candidate_oof(
                frame,
                feature_columns=feature_columns,
                lookback_sessions=lookback,
                horizon_sessions=horizon,
                contract=contract,
                development=development,
            )
            pooled = _prediction_metrics(oof, contract=contract)
            key = f"lookback_{lookback}_horizon_{horizon}"
            candidates[key] = {
                "lookback_sessions": lookback,
                "horizon_sessions": horizon,
                "input_samples": int(len(frame)),
                "oof_samples": int(len(oof)),
                "feature_count": len(feature_columns),
                "pooled_metrics": pooled,
                "folds": fold_reports,
                "gates": _gate_report(
                    pooled,
                    fold_reports,
                    contract=contract,
                    development=development,
                ),
            }
            predictions.append(oof)

    identity_audit: dict[str, object] = {}
    for horizon, by_lookback in sample_sets.items():
        values = list(by_lookback.values())
        identical = bool(values and all(value == values[0] for value in values[1:]))
        identity_audit[str(horizon)] = {
            "identical_across_lookbacks": identical,
            "samples": len(values[0]) if values else 0,
        }
        if not identical:
            raise ValueError(
                f"registered lookbacks do not use identical samples for horizon {horizon}"
            )

    oof = pd.concat(predictions, ignore_index=True).sort_values(
        ["lookback_sessions", "horizon_sessions", "published_date", "sample_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return DirectSequenceBaselineResult(
        oof_predictions=oof,
        report={
            "experiment": "causal-timeseries-lstm-daily-v1-simple-sequence-baselines",
            "status": "KEEP_RESEARCH_ONLY_DO_NOT_FREEZE",
            "dataset_role_read": "train",
            "official_test_read": False,
            "quarantine_read": False,
            "forward_validation_read": False,
            "model_selection": contract.baseline_selection_policy,
            "preprocessing": (
                "median imputation and standardization fitted separately on each "
                "outer-fold train segment"
            ),
            "sequence_statistics": list(contract.baseline_statistics),
            "classification_model": {
                "family": "logistic_regression",
                "c": contract.baseline_logistic_c,
                "class_weight": None,
            },
            "return_model": {
                "family": "ridge",
                "alpha": contract.baseline_ridge_alpha,
            },
            "interval_model": (
                "outer-fold train residual quantiles added to Ridge prediction"
            ),
            "sample_identity_audit": identity_audit,
            "candidates": candidates,
        },
    )


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest must be an object: {path}")
    return value


def run_train_sequence_baselines(
    artifact_directory: Path,
    *,
    contract: LstmTimeseriesContract,
    development: CausalDevelopmentContract,
    force: bool = False,
) -> dict[str, object]:
    root = artifact_directory.resolve()
    if not root.is_relative_to(contract.data_directory):
        raise ValueError("sequence baseline input leaves the registered data boundary")
    sequence_manifest_path = root / "manifests" / "train_manifest.json"
    label_manifest_path = root / "manifests" / "train_labels_manifest.json"
    sequence_manifest = _load_json(sequence_manifest_path)
    label_manifest = _load_json(label_manifest_path)
    for manifest in (sequence_manifest, label_manifest):
        if manifest.get("contract_sha256") != contract.source_sha256:
            raise ValueError("sequence baseline artifact contract hash changed")
        if manifest.get("split_contract_sha256") != contract.split_contract_sha256:
            raise ValueError("sequence baseline artifact split hash changed")
        if manifest.get("official_test_read") is not False:
            raise PermissionError("sealed official-test role reached sequence baseline")

    sequence_path = root / str(sequence_manifest["sequences_file"])
    mask_path = root / str(sequence_manifest["time_masks_file"])
    metadata_path = root / str(sequence_manifest["metadata_file"])
    label_path = root / str(label_manifest["labels_file"])
    for path, expected in (
        (sequence_path, sequence_manifest["sequences_sha256"]),
        (mask_path, sequence_manifest["time_masks_sha256"]),
        (metadata_path, sequence_manifest["metadata_sha256"]),
        (label_path, label_manifest["labels_sha256"]),
    ):
        if _file_sha256(path) != expected:
            raise ValueError(f"sequence baseline input hash changed: {path.name}")

    result = run_direct_sequence_baseline_oof(
        np.load(sequence_path, mmap_mode="r", allow_pickle=False),
        np.load(mask_path, mmap_mode="r", allow_pickle=False),
        pd.read_parquet(metadata_path),
        pd.read_parquet(label_path),
        contract=contract,
        development=development,
    )
    oof_path = root / "oof" / "train_simple_sequence_baselines.parquet"
    report_path = root / "reports" / "train_simple_sequence_baselines.json"
    existing = [path for path in (oof_path, report_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(f"sequence baseline output already exists: {existing[0]}")
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = oof_path.with_suffix(oof_path.suffix + ".tmp")
    result.oof_predictions.to_parquet(temporary, index=False)
    temporary.replace(oof_path)
    report = {
        **result.report,
        "created_at": datetime.now(UTC).isoformat(),
        "contract": contract.contract,
        "contract_sha256": contract.source_sha256,
        "development_contract_sha256": contract.development_contract_sha256,
        "sequence_manifest_sha256": _file_sha256(sequence_manifest_path),
        "label_manifest_sha256": _file_sha256(label_manifest_path),
        "producer_code_sha256": _file_sha256(Path(__file__)),
        "oof_file": oof_path.relative_to(root).as_posix(),
        "oof_sha256": _file_sha256(oof_path),
        "oof_rows": int(len(result.oof_predictions)),
        "limitations": [
            "registered lookbacks are reported separately without outer-OOF selection",
            "probability and interval calibration are reserved for the LSTM stage",
            "no model is release-eligible before LSTM and tabular incremental gates",
        ],
    }
    _write_json_atomic(report, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "timeseries" / "lstm_daily_v1",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    development = load_causal_development_contract()
    contract = load_lstm_timeseries_contract(development=development)
    report = run_train_sequence_baselines(
        args.artifact_dir,
        contract=contract,
        development=development,
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

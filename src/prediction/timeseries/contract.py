"""Machine-enforced contract for causal daily time-series development."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT
from src.prediction.development_contract import (
    CausalDevelopmentContract,
    load_causal_development_contract,
)
from src.prediction.splits import (
    PredictionSplitContract,
    load_prediction_split_contract,
)

DEFAULT_TIMESERIES_CONTRACT_PATH = (
    PROJECT_ROOT / "config" / "timeseries_daily_v1.yaml"
)


@dataclass(frozen=True)
class TimeseriesModelContract:
    contract: str
    split_contract: str
    development_contract: str
    source_directory: Path
    data_directory: Path
    model_directory: Path
    feature_frequency: str
    decision_point: str
    historical_announcement_time_precision: str
    market_daily_rule: str
    daily_bar_available_time: time
    lookbacks_sessions: tuple[int, ...]
    maximum_sequence_sessions: int
    minimum_history_sessions: int
    minimum_valid_stock_sessions: int
    intraday_candidate_frequency: str
    intraday_prediction_status: str
    baseline_families: tuple[str, ...]
    logistic_c_values: tuple[float, ...]
    logistic_class_weights: tuple[str, ...]
    sequence_candidate_families: tuple[str, ...]
    required_availability_metadata: bool
    forbidden_feature_names: frozenset[str]
    forbidden_feature_prefixes: tuple[str, ...]
    training_start: date
    training_end: date
    split_strategy: str
    horizons_sessions: tuple[int, ...]
    official_test_start: date
    official_test_end: date
    official_test_read_limit: int
    quarantine_start: date
    quarantine_end: date
    forward_validation_start: date
    source_path: Path
    source_sha256: str

    def feature_name_allowed(self, name: str) -> bool:
        normalized = name.strip().lower()
        if normalized in self.forbidden_feature_names:
            return False
        return not any(
            normalized.startswith(prefix)
            for prefix in self.forbidden_feature_prefixes
        )


def _project_path(raw: str) -> Path:
    return (PROJECT_ROOT / raw).resolve()


def load_timeseries_model_contract(
    path: Path = DEFAULT_TIMESERIES_CONTRACT_PATH,
    *,
    split: PredictionSplitContract | None = None,
    development: CausalDevelopmentContract | None = None,
) -> TimeseriesModelContract:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    contract = TimeseriesModelContract(
        contract=str(raw["contract"]),
        split_contract=str(raw["split_contract"]),
        development_contract=str(raw["development_contract"]),
        source_directory=_project_path(raw["directories"]["source"]),
        data_directory=_project_path(raw["directories"]["data"]),
        model_directory=_project_path(raw["directories"]["models"]),
        feature_frequency=str(raw["sampling"]["feature_frequency"]),
        decision_point=str(raw["sampling"]["decision_point"]),
        historical_announcement_time_precision=str(
            raw["sampling"]["historical_announcement_time_precision"]
        ),
        market_daily_rule=str(raw["sampling"]["market_daily_rule"]),
        daily_bar_available_time=time.fromisoformat(
            raw["sampling"]["daily_bar_available_time"]
        ),
        lookbacks_sessions=tuple(
            int(item) for item in raw["sampling"]["sequence_lookbacks_sessions"]
        ),
        maximum_sequence_sessions=int(
            raw["sampling"]["maximum_sequence_sessions"]
        ),
        minimum_history_sessions=int(raw["sampling"]["minimum_history_sessions"]),
        minimum_valid_stock_sessions=int(
            raw["sampling"]["minimum_valid_stock_sessions"]
        ),
        intraday_candidate_frequency=str(
            raw["sampling"]["intraday_candidate_frequency"]
        ),
        intraday_prediction_status=str(
            raw["sampling"]["intraday_prediction_status"]
        ),
        baseline_families=tuple(
            str(item) for item in raw["model_families"]["baselines"]
        ),
        logistic_c_values=tuple(
            float(item)
            for item in raw["model_families"]["baseline_candidates"][
                "logistic_regression"
            ]["c_values"]
        ),
        logistic_class_weights=tuple(
            str(item)
            for item in raw["model_families"]["baseline_candidates"][
                "logistic_regression"
            ]["class_weights"]
        ),
        sequence_candidate_families=tuple(
            str(item) for item in raw["model_families"]["sequence_candidates"]
        ),
        required_availability_metadata=bool(
            raw["features"]["required_availability_metadata"]
        ),
        forbidden_feature_names=frozenset(
            str(item).strip().lower()
            for item in raw["features"]["forbidden_exact"]
        ),
        forbidden_feature_prefixes=tuple(
            str(item).strip().lower()
            for item in raw["features"]["forbidden_prefixes"]
        ),
        training_start=date.fromisoformat(raw["training"]["date_start"]),
        training_end=date.fromisoformat(raw["training"]["date_end"]),
        split_strategy=str(raw["training"]["split_strategy"]),
        horizons_sessions=tuple(
            int(item) for item in raw["evaluation"]["horizons_sessions"]
        ),
        official_test_start=date.fromisoformat(raw["official_test"]["date_start"]),
        official_test_end=date.fromisoformat(raw["official_test"]["date_end"]),
        official_test_read_limit=int(raw["official_test"]["read_limit"]),
        quarantine_start=date.fromisoformat(raw["quarantine"]["date_start"]),
        quarantine_end=date.fromisoformat(raw["quarantine"]["date_end"]),
        forward_validation_start=date.fromisoformat(
            raw["forward_validation"]["date_start"]
        ),
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    validate_timeseries_model_contract(
        contract,
        raw=raw,
        split=split or load_prediction_split_contract(),
        development=development or load_causal_development_contract(),
    )
    return contract


def validate_timeseries_model_contract(
    contract: TimeseriesModelContract,
    *,
    raw: dict,
    split: PredictionSplitContract,
    development: CausalDevelopmentContract,
) -> None:
    if contract.split_contract != split.contract:
        raise ValueError("time-series and temporal split contracts do not match")
    if contract.development_contract != development.contract:
        raise ValueError("time-series and causal development contracts do not match")
    if (contract.training_start, contract.training_end) != (
        split.train.start,
        split.train.end,
    ):
        raise ValueError("time-series training dates must equal the train partition")
    if (contract.official_test_start, contract.official_test_end) != (
        split.test.start,
        split.test.end,
    ):
        raise ValueError("time-series official-test dates must equal the test partition")
    if (contract.quarantine_start, contract.quarantine_end) != (
        split.quarantine.start,
        split.quarantine.end,
    ):
        raise ValueError("time-series quarantine dates must equal the quarantine partition")
    if contract.forward_validation_start != split.forward_validation_start:
        raise ValueError("time-series forward-validation start does not match split contract")
    if contract.feature_frequency != "daily":
        raise ValueError("time-series daily-v1 features must remain daily")
    if contract.decision_point != development.decision_name:
        raise ValueError("time-series decision point does not match development contract")
    if contract.historical_announcement_time_precision != "date_only":
        raise ValueError("daily-v1 must retain conservative date-only announcement timing")
    if contract.market_daily_rule != development.historical_market_daily_rule:
        raise ValueError("time-series daily market cutoff differs from development contract")
    if contract.intraday_candidate_frequency != "15min":
        raise ValueError("the first intraday candidate frequency must remain 15min")
    if contract.intraday_prediction_status != (
        "blocked_pending_authoritative_announcement_timestamp"
    ):
        raise ValueError("intraday prediction must remain blocked in daily-v1")
    if contract.split_strategy != "expanding_window":
        raise ValueError("time-series daily-v1 must use expanding-window validation")
    if raw["training"]["random_split_allowed"]:
        raise ValueError("random time-series splits are forbidden")
    if not raw["training"]["fit_preprocessors_on_fold_train_only"]:
        raise ValueError("time-series preprocessors must fit on fold-train only")
    if not raw["training"]["outcome_must_mature_by_fold_end"]:
        raise ValueError("labels must mature by each fold cutoff")
    if tuple(sorted(contract.lookbacks_sessions)) != contract.lookbacks_sessions:
        raise ValueError("time-series lookbacks must be unique and increasing")
    if len(set(contract.lookbacks_sessions)) != len(contract.lookbacks_sessions):
        raise ValueError("time-series lookbacks must be unique and increasing")
    if contract.maximum_sequence_sessions != max(contract.lookbacks_sessions):
        raise ValueError("maximum sequence length must equal the largest lookback")
    if contract.minimum_history_sessions > contract.maximum_sequence_sessions:
        raise ValueError("minimum history cannot exceed maximum sequence length")
    if contract.minimum_valid_stock_sessions > contract.minimum_history_sessions:
        raise ValueError("minimum valid stock sessions cannot exceed minimum history")
    if set(contract.baseline_families) != {"logistic_regression"}:
        raise ValueError("daily-v1 baseline family must remain logistic regression")
    if not contract.logistic_c_values or any(
        value <= 0 for value in contract.logistic_c_values
    ):
        raise ValueError("logistic C candidates must be positive")
    if set(contract.logistic_class_weights) != {"none", "balanced"}:
        raise ValueError("daily-v1 must compare unweighted and balanced logistic models")
    if set(contract.sequence_candidate_families) != {"small_tcn"}:
        raise ValueError("daily-v1 sequence candidate must remain a small TCN")
    if not contract.required_availability_metadata:
        raise ValueError("every time-series feature requires availability metadata")
    normalization = raw["features"]["normalization"]
    if normalization["full_sample_fitted_allowed"]:
        raise ValueError("full-sample normalization is forbidden")
    if not normalization["fold_train_fitted_allowed"]:
        raise ValueError("fold-train-only normalization must be allowed")
    if not raw["labels"]["stored_separately_from_features"]:
        raise ValueError("time-series labels must be physically separate from features")
    if not raw["labels"]["economic_value"]["feature_use_forbidden"]:
        raise ValueError("future economic outcomes can never be model features")
    gate = development.module_gates["time_series_model"]
    if not gate["all_bars_must_precede_prediction_as_of"]:
        raise ValueError("development contract must reject bars after prediction as-of")
    if not gate["incremental_oof_improvement_over_table_required"]:
        raise ValueError("time-series release requires incremental OOF improvement")
    evaluation = raw["evaluation"]
    general_gate = development.module_gates["predictive_general"]
    for key in (
        "minimum_oof_bullish_signals_per_horizon",
        "bullish_wilson_lower_must_exceed",
        "minimum_folds_with_bullish_hit_rate_above_half",
        "minimum_worst_fold_bullish_hit_rate",
        "cost_adjusted_mean_excess_return_ci_lower_must_exceed",
        "brier_score_must_beat_constant_base_rate",
    ):
        if evaluation[key] != general_gate[key]:
            raise ValueError(f"time-series gate differs from general gate: {key}")
    if not evaluation["incremental_oof_improvement_over_table_required"]:
        raise ValueError("time-series evaluation must require table-model increment")
    if not evaluation["incremental_comparison_requires_identical_samples_and_folds"]:
        raise ValueError("incremental comparison requires identical samples and folds")
    if not raw["official_test"][
        "requires_frozen_model_features_preprocessors_thresholds"
    ]:
        raise ValueError("official test requires a completely frozen candidate")
    if raw["official_test"]["tuning_after_read_allowed"]:
        raise ValueError("official-test tuning is forbidden")
    if contract.official_test_read_limit != 1:
        raise ValueError("official test must have a one-read limit")
    if raw["quarantine"]["fitting_allowed"]:
        raise ValueError("quarantine cannot be used for fitting")
    if raw["quarantine"]["model_selection_allowed"]:
        raise ValueError("quarantine cannot be used for model selection")
    if raw["forward_validation"]["training_allowed"]:
        raise ValueError("forward validation cannot be used for training")
    if raw["forward_validation"]["model_selection_allowed"]:
        raise ValueError("forward validation cannot be used for model selection")

    expected_directories = {
        contract.source_directory: PROJECT_ROOT / "src" / "prediction" / "timeseries",
        contract.data_directory: PROJECT_ROOT / "data" / "timeseries",
        contract.model_directory: PROJECT_ROOT / "models" / "timeseries",
    }
    for actual, expected in expected_directories.items():
        if actual != expected.resolve():
            raise ValueError(f"unexpected time-series directory: {actual}")

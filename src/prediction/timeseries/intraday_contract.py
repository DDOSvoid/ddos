"""Contract for train-only complete-day features derived from 15-minute bars."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
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
from src.prediction.tabular.aggregates import (
    AGGREGATION_VERSION,
    intraday_feature_schema_sha256,
)
from src.prediction.timeseries.contract import (
    TimeseriesModelContract,
    load_timeseries_model_contract,
)

DEFAULT_INTRADAY_AGGREGATE_CONTRACT_PATH = (
    PROJECT_ROOT / "config" / "timeseries_intraday_aggregates_v1.yaml"
)


@dataclass(frozen=True)
class IntradayAggregateTimeseriesContract:
    contract: str
    parent_contract: str
    parent_contract_sha256: str
    split_contract: str
    development_contract: str
    source_directory: Path
    data_directory: Path
    model_directory: Path
    intraday_source_directory: Path
    source_audit_path: Path
    aggregate_state_path: Path
    source_frequency: str
    source_bar_semantics: str
    source_aggregation_version: str
    source_feature_schema_sha256: str
    feature_frequency: str
    decision_point: str
    historical_announcement_time_precision: str
    lookbacks_complete_days: tuple[int, ...]
    minimum_complete_intraday_days: int
    feature_set_candidates: tuple[str, ...]
    primary_incremental_comparison: str
    stability_rule: str
    logistic_c_values: tuple[float, ...]
    logistic_class_weights: tuple[str, ...]
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


def load_intraday_aggregate_timeseries_contract(
    path: Path = DEFAULT_INTRADAY_AGGREGATE_CONTRACT_PATH,
    *,
    split: PredictionSplitContract | None = None,
    development: CausalDevelopmentContract | None = None,
    parent: TimeseriesModelContract | None = None,
) -> IntradayAggregateTimeseriesContract:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    fixed = raw["model_families"]["diagnostic_fixed_parameters"]
    contract = IntradayAggregateTimeseriesContract(
        contract=str(raw["contract"]),
        parent_contract=str(raw["parent_contract"]),
        parent_contract_sha256=str(raw["parent_contract_sha256"]),
        split_contract=str(raw["split_contract"]),
        development_contract=str(raw["development_contract"]),
        source_directory=_project_path(raw["directories"]["source"]),
        data_directory=_project_path(raw["directories"]["data"]),
        model_directory=_project_path(raw["directories"]["models"]),
        intraday_source_directory=_project_path(
            raw["directories"]["approved_read_only_intraday_source"]
        ),
        source_audit_path=_project_path(raw["source_snapshot"]["audit_file"]),
        aggregate_state_path=_project_path(
            raw["source_snapshot"]["aggregate_state_file"]
        ),
        source_frequency=str(raw["source_snapshot"]["source_frequency"]),
        source_bar_semantics=str(raw["source_snapshot"]["source_bar_semantics"]),
        source_aggregation_version=str(
            raw["source_snapshot"]["source_aggregation_version"]
        ),
        source_feature_schema_sha256=str(
            raw["source_snapshot"]["source_feature_schema_sha256"]
        ),
        feature_frequency=str(raw["sampling"]["feature_frequency"]),
        decision_point=str(raw["sampling"]["decision_point"]),
        historical_announcement_time_precision=str(
            raw["sampling"]["historical_announcement_time_precision"]
        ),
        lookbacks_complete_days=tuple(
            int(item) for item in raw["sampling"]["lookbacks_complete_days"]
        ),
        minimum_complete_intraday_days=int(
            raw["sampling"]["minimum_complete_intraday_days"]
        ),
        feature_set_candidates=tuple(
            str(item) for item in raw["feature_sets"]["candidates"]
        ),
        primary_incremental_comparison=str(
            raw["feature_sets"]["primary_incremental_comparison"]
        ),
        stability_rule=str(raw["feature_sets"]["stability_rule"]),
        logistic_c_values=(float(fixed["c"]),),
        logistic_class_weights=(str(fixed["class_weight"]),),
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
    validate_intraday_aggregate_timeseries_contract(
        contract,
        raw=raw,
        split=split or load_prediction_split_contract(),
        development=development or load_causal_development_contract(),
        parent=parent or load_timeseries_model_contract(),
    )
    return contract


def validate_intraday_aggregate_timeseries_contract(
    contract: IntradayAggregateTimeseriesContract,
    *,
    raw: dict,
    split: PredictionSplitContract,
    development: CausalDevelopmentContract,
    parent: TimeseriesModelContract,
) -> None:
    if contract.parent_contract != parent.contract:
        raise ValueError("intraday aggregate and parent time-series contracts differ")
    if contract.parent_contract_sha256 != parent.source_sha256:
        raise ValueError("parent daily-v1 contract changed; version the intraday contract")
    if contract.split_contract != split.contract:
        raise ValueError("intraday aggregate and temporal split contracts differ")
    if contract.development_contract != development.contract:
        raise ValueError("intraday aggregate and development contracts differ")
    if (contract.training_start, contract.training_end) != (
        split.train.start,
        split.train.end,
    ):
        raise ValueError("intraday aggregate training dates must equal train partition")
    if (contract.official_test_start, contract.official_test_end) != (
        split.test.start,
        split.test.end,
    ):
        raise ValueError("intraday official-test dates must equal test partition")
    if (contract.quarantine_start, contract.quarantine_end) != (
        split.quarantine.start,
        split.quarantine.end,
    ):
        raise ValueError("intraday quarantine dates must equal quarantine partition")
    if contract.forward_validation_start != split.forward_validation_start:
        raise ValueError("intraday forward-validation boundary differs")
    if contract.source_directory != (
        PROJECT_ROOT / "src" / "prediction" / "timeseries"
    ).resolve():
        raise ValueError("unexpected intraday time-series source directory")
    if contract.data_directory != (PROJECT_ROOT / "data" / "timeseries").resolve():
        raise ValueError("unexpected intraday time-series data directory")
    if contract.model_directory != (
        PROJECT_ROOT / "models" / "timeseries"
    ).resolve():
        raise ValueError("unexpected intraday time-series model directory")
    if contract.intraday_source_directory != (
        PROJECT_ROOT / "data" / "tabular" / "intraday_15m"
    ).resolve():
        raise ValueError("unexpected read-only intraday source directory")
    if contract.source_frequency != "15min":
        raise ValueError("intraday aggregate source frequency must remain 15min")
    if contract.source_bar_semantics != "bar_start":
        raise ValueError("AmazingData bars must retain bar-start semantics")
    if contract.source_aggregation_version != AGGREGATION_VERSION:
        raise ValueError("intraday aggregation code version changed")
    if contract.source_feature_schema_sha256 != intraday_feature_schema_sha256():
        raise ValueError("intraday aggregate feature schema changed")
    if contract.feature_frequency != "daily_from_complete_15min_days":
        raise ValueError("intraday-v1 may only emit complete-day features")
    if contract.decision_point != development.decision_name:
        raise ValueError("intraday aggregate decision point differs")
    if contract.historical_announcement_time_precision != "date_only":
        raise ValueError("intraday aggregate v1 must retain date-only timing")
    if tuple(sorted(contract.lookbacks_complete_days)) != (
        contract.lookbacks_complete_days
    ):
        raise ValueError("intraday complete-day lookbacks must be increasing")
    if contract.minimum_complete_intraday_days != max(
        contract.lookbacks_complete_days
    ):
        raise ValueError("minimum intraday history must equal maximum lookback")
    if set(contract.feature_set_candidates) != {
        "daily_only_matched",
        "intraday_only",
        "daily_plus_intraday",
    }:
        raise ValueError("intraday comparison feature sets changed")
    if contract.primary_incremental_comparison != (
        "daily_plus_intraday_vs_daily_only_matched"
    ):
        raise ValueError("intraday primary incremental comparison changed")
    if contract.stability_rule != (
        "pooled_brier_lower_and_at_least_two_of_three_folds_lower"
    ):
        raise ValueError("intraday stability rule changed")
    if contract.logistic_c_values != (0.1,):
        raise ValueError("intraday diagnostic must use fixed C=0.1")
    if contract.logistic_class_weights != ("none",):
        raise ValueError("intraday diagnostic must use unweighted logistic")
    if raw["sampling"]["publication_day_intraday_allowed"]:
        raise ValueError("publication-day intraday bars are forbidden")
    if raw["sampling"]["minute_level_decision_status"] != (
        "blocked_pending_authoritative_announcement_timestamp"
    ):
        raise ValueError("minute-level decisions must remain blocked")
    if raw["training"]["random_split_allowed"]:
        raise ValueError("random intraday aggregate splits are forbidden")
    if not raw["training"]["fit_preprocessors_on_fold_train_only"]:
        raise ValueError("intraday preprocessing must fit on fold-train only")
    if raw["features"]["normalization"]["full_sample_fitted_allowed"]:
        raise ValueError("full-sample intraday normalization is forbidden")
    if not raw["labels"]["stored_separately_from_features"]:
        raise ValueError("intraday labels must be physically separate")
    if not raw["labels"]["reuse_parent_train_labels_by_sample_id"]:
        raise ValueError("intraday labels must reuse parent sample identities")
    if not contract.required_availability_metadata:
        raise ValueError("intraday features require availability metadata")
    if contract.split_strategy != "expanding_window":
        raise ValueError("intraday aggregate v1 must use expanding windows")
    if raw["official_test"]["tuning_after_read_allowed"]:
        raise ValueError("official-test tuning is forbidden")
    if contract.official_test_read_limit != 1:
        raise ValueError("official test must have a one-read limit")
    if raw["quarantine"]["fitting_allowed"]:
        raise ValueError("quarantine fitting is forbidden")
    if raw["quarantine"]["model_selection_allowed"]:
        raise ValueError("quarantine model selection is forbidden")
    if raw["forward_validation"]["training_allowed"]:
        raise ValueError("forward-validation training is forbidden")
    if raw["forward_validation"]["model_selection_allowed"]:
        raise ValueError("forward-validation model selection is forbidden")

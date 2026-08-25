"""Machine-enforced contract for the causal daily LSTM research branch."""

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
from src.prediction.timeseries.contract import (
    TimeseriesModelContract,
    load_timeseries_model_contract,
)

DEFAULT_LSTM_CONTRACT_PATH = (
    PROJECT_ROOT / "config" / "timeseries_lstm_daily_v1.yaml"
)

EXPECTED_SEQUENCE_COLUMNS = (
    "stock_return_1d",
    "benchmark_return_1d",
    "excess_return_1d",
    "stock_range_1d",
    "stock_gap_1d",
    "stock_log_volume",
    "stock_log_amount",
    "stock_tradable_mask",
    "stock_volume_mask",
    "stock_amount_mask",
)

EXPECTED_OUTPUT_FIELDS = (
    "sample_id",
    "company_day_id",
    "stock_code",
    "published_date",
    "prediction_as_of",
    "horizon_sessions",
    "bullish_probability",
    "expected_excess_return",
    "return_interval_lower",
    "return_interval_upper",
    "risk_scale",
    "sequence_available",
    "unavailable_reason",
    "model_version",
    "data_as_of",
    "evidence_sha256",
)


@dataclass(frozen=True)
class LstmTimeseriesContract:
    contract: str
    parent_contract: str
    parent_contract_sha256: str
    split_contract: str
    split_contract_sha256: str
    development_contract: str
    development_contract_sha256: str
    source_directory: Path
    data_directory: Path
    model_directory: Path
    feature_frequency: str
    decision_point: str
    prediction_as_of_rule: str
    historical_announcement_time_precision: str
    market_daily_rule: str
    lookback_candidates_sessions: tuple[int, ...]
    maximum_sequence_sessions: int
    minimum_valid_stock_sessions: int
    prehistory_required_start: date
    company_day_id_algorithm: str
    sample_id_algorithm: str
    horizons_sessions: tuple[int, ...]
    missing_sequence_policy: str
    sequence_columns: tuple[str, ...]
    forbidden_feature_names: frozenset[str]
    forbidden_feature_prefixes: tuple[str, ...]
    model_family: str
    num_layers: int
    hidden_size_candidates: tuple[int, ...]
    external_dropout_candidates: tuple[float, ...]
    recurrent_dropout: float
    bidirectional: bool
    horizon_training: str
    interval_quantiles: tuple[float, float]
    baseline_statistics: tuple[str, ...]
    baseline_logistic_c: float
    baseline_ridge_alpha: float
    baseline_random_state: int
    baseline_selection_policy: str
    training_start: date
    training_end: date
    random_split_allowed: bool
    seed: int
    batch_size: int
    training_device: str
    cpu_threads: int
    deterministic_algorithms: bool
    optimizer_name: str
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    minimum_epochs: int
    maximum_epochs: int
    early_stopping_patience: int
    early_stopping_metric: str
    refit_after_early_stopping: bool
    huber_delta: float
    direction_loss_weight: float
    return_loss_weight: float
    interval_loss_weight: float
    inner_split_unit: str
    inner_outcome_maturity_required: bool
    inner_temporal_partitions: tuple[float, float, float]
    probability_calibration: str
    probability_temperature_bounds: tuple[float, float]
    calibration_identity_fallback: bool
    interval_calibration: str
    interval_target_coverage: float
    interval_quantile_method: str
    output_fields: tuple[str, ...]
    risk_scale_definition: str
    bullish_threshold: float
    round_trip_cost_bps: float
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


def _load_raw(path: Path) -> tuple[bytes, dict]:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("LSTM contract must be a YAML mapping")
    return raw_bytes, raw


def load_lstm_timeseries_contract(
    path: Path = DEFAULT_LSTM_CONTRACT_PATH,
    *,
    parent: TimeseriesModelContract | None = None,
    split: PredictionSplitContract | None = None,
    development: CausalDevelopmentContract | None = None,
) -> LstmTimeseriesContract:
    raw_bytes, raw = _load_raw(path)
    training = raw["training"]
    inner = training["inner_temporal_partitions"]
    contract = LstmTimeseriesContract(
        contract=str(raw["contract"]),
        parent_contract=str(raw["parent_contract"]),
        parent_contract_sha256=str(raw["parent_contract_sha256"]),
        split_contract=str(raw["split_contract"]),
        split_contract_sha256=str(raw["split_contract_sha256"]),
        development_contract=str(raw["development_contract"]),
        development_contract_sha256=str(raw["development_contract_sha256"]),
        source_directory=_project_path(raw["directories"]["source"]),
        data_directory=_project_path(raw["directories"]["data"]),
        model_directory=_project_path(raw["directories"]["models"]),
        feature_frequency=str(raw["sampling"]["feature_frequency"]),
        decision_point=str(raw["sampling"]["decision_point"]),
        prediction_as_of_rule=str(raw["sampling"]["prediction_as_of_rule"]),
        historical_announcement_time_precision=str(
            raw["sampling"]["historical_announcement_time_precision"]
        ),
        market_daily_rule=str(raw["sampling"]["market_daily_rule"]),
        lookback_candidates_sessions=tuple(
            int(value)
            for value in raw["sampling"]["lookback_candidates_sessions"]
        ),
        maximum_sequence_sessions=int(
            raw["sampling"]["maximum_sequence_sessions"]
        ),
        minimum_valid_stock_sessions=int(
            raw["sampling"]["minimum_valid_stock_sessions"]
        ),
        prehistory_required_start=date.fromisoformat(
            raw["sampling"]["prehistory_required_start"]
        ),
        company_day_id_algorithm=str(
            raw["identity"]["company_day_id_algorithm"]
        ),
        sample_id_algorithm=str(raw["identity"]["sample_id_algorithm"]),
        horizons_sessions=tuple(
            int(value) for value in raw["identity"]["horizons_sessions"]
        ),
        missing_sequence_policy=str(raw["identity"]["missing_sequence_policy"]),
        sequence_columns=tuple(str(value) for value in raw["features"]["sequence_columns"]),
        forbidden_feature_names=frozenset(
            str(value).strip().lower()
            for value in raw["features"]["forbidden_exact"]
        ),
        forbidden_feature_prefixes=tuple(
            str(value).strip().lower()
            for value in raw["features"]["forbidden_prefixes"]
        ),
        model_family=str(raw["model"]["family"]),
        num_layers=int(raw["model"]["num_layers"]),
        hidden_size_candidates=tuple(
            int(value) for value in raw["model"]["hidden_size_candidates"]
        ),
        external_dropout_candidates=tuple(
            float(value)
            for value in raw["model"]["external_dropout_candidates"]
        ),
        recurrent_dropout=float(raw["model"]["recurrent_dropout"]),
        bidirectional=bool(raw["model"]["bidirectional"]),
        horizon_training=str(raw["model"]["horizon_training"]),
        interval_quantiles=tuple(
            float(value) for value in raw["model"]["interval_quantiles"]
        ),
        baseline_statistics=tuple(
            str(value) for value in raw["baselines"]["pooled_sequence_statistics"]
        ),
        baseline_logistic_c=float(raw["baselines"]["logistic_c"]),
        baseline_ridge_alpha=float(raw["baselines"]["ridge_alpha"]),
        baseline_random_state=int(raw["baselines"]["random_state"]),
        baseline_selection_policy=str(
            raw["baselines"]["hyperparameter_selection"]
        ),
        training_start=date.fromisoformat(training["date_start"]),
        training_end=date.fromisoformat(training["date_end"]),
        random_split_allowed=bool(training["random_split_allowed"]),
        seed=int(training["seed"]),
        batch_size=int(training["batch_size"]),
        training_device=str(training["device"]),
        cpu_threads=int(training["cpu_threads"]),
        deterministic_algorithms=bool(training["deterministic_algorithms"]),
        optimizer_name=str(training["optimizer"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        minimum_epochs=int(training["minimum_epochs"]),
        maximum_epochs=int(training["maximum_epochs"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        early_stopping_metric=str(training["early_stopping_metric"]),
        refit_after_early_stopping=bool(training["refit_after_early_stopping"]),
        huber_delta=float(training["huber_delta"]),
        direction_loss_weight=float(training["loss_weights"]["direction_bce"]),
        return_loss_weight=float(training["loss_weights"]["return_huber"]),
        interval_loss_weight=float(training["loss_weights"]["interval_pinball"]),
        inner_split_unit=str(inner["split_unit"]),
        inner_outcome_maturity_required=bool(inner["outcome_maturity_required"]),
        inner_temporal_partitions=(
            float(inner["model_fit_fraction"]),
            float(inner["early_stop_fraction"]),
            float(inner["calibration_fraction"]),
        ),
        probability_calibration=str(raw["calibration"]["probability_method"]),
        probability_temperature_bounds=tuple(
            float(value)
            for value in raw["calibration"]["probability_temperature_bounds"]
        ),
        calibration_identity_fallback=bool(
            raw["calibration"]["identity_fallback_if_calibration_brier_worsens"]
        ),
        interval_calibration=str(raw["calibration"]["interval_method"]),
        interval_target_coverage=float(
            raw["calibration"]["interval_target_coverage"]
        ),
        interval_quantile_method=str(
            raw["calibration"]["interval_quantile_method"]
        ),
        output_fields=tuple(str(value) for value in raw["outputs"]["required"]),
        risk_scale_definition=str(raw["outputs"]["risk_scale_definition"]),
        bullish_threshold=float(raw["evaluation"]["bullish_threshold"]),
        round_trip_cost_bps=float(raw["evaluation"]["round_trip_cost_bps"]),
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
    validate_lstm_timeseries_contract(
        contract,
        raw=raw,
        parent=parent or load_timeseries_model_contract(),
        split=split or load_prediction_split_contract(),
        development=development or load_causal_development_contract(),
    )
    return contract


def validate_lstm_timeseries_contract(
    contract: LstmTimeseriesContract,
    *,
    raw: dict,
    parent: TimeseriesModelContract,
    split: PredictionSplitContract,
    development: CausalDevelopmentContract,
) -> None:
    if contract.contract != "causal-timeseries-lstm-daily-v1":
        raise ValueError("unexpected LSTM contract name")
    if contract.parent_contract != parent.contract:
        raise ValueError("LSTM parent contract name does not match daily-v1")
    if contract.parent_contract_sha256 != parent.source_sha256:
        raise ValueError("LSTM parent contract hash does not match daily-v1")
    if contract.split_contract != split.contract:
        raise ValueError("LSTM split contract name does not match")
    if contract.split_contract_sha256 != split.source_sha256:
        raise ValueError("LSTM split contract hash does not match current source")
    if contract.development_contract != development.contract:
        raise ValueError("LSTM development contract name does not match")
    if contract.development_contract_sha256 != development.source_sha256:
        raise ValueError("LSTM development contract hash does not match current source")
    if contract.source_directory != (
        PROJECT_ROOT / "src" / "prediction" / "timeseries"
    ).resolve():
        raise ValueError("unexpected LSTM source directory")
    if contract.data_directory != (
        PROJECT_ROOT / "data" / "timeseries" / "lstm_daily_v1"
    ).resolve():
        raise ValueError("unexpected LSTM data directory")
    if contract.model_directory != (
        PROJECT_ROOT / "models" / "timeseries" / "lstm_daily_v1"
    ).resolve():
        raise ValueError("unexpected LSTM model directory")
    if contract.feature_frequency != "daily_direct_sequence":
        raise ValueError("LSTM v1 must use direct daily sequences")
    if contract.decision_point != development.decision_name:
        raise ValueError("LSTM decision point does not match development contract")
    if contract.prediction_as_of_rule != (
        "next_trading_session_preopen_strictly_after_published_date"
    ):
        raise ValueError("LSTM prediction as-of must be the next trading pre-open")
    if contract.historical_announcement_time_precision != "date_only":
        raise ValueError("LSTM daily-v1 must preserve date-only timing")
    if contract.market_daily_rule != development.historical_market_daily_rule:
        raise ValueError("LSTM daily cutoff differs from development contract")
    if raw["sampling"]["publication_day_bars_allowed"]:
        raise ValueError("publication-day bars are forbidden in LSTM daily-v1")
    if raw["sampling"]["future_padding_allowed"]:
        raise ValueError("future padding is forbidden")
    if contract.lookback_candidates_sessions != (20, 60):
        raise ValueError("LSTM v1 lookbacks must remain exactly 20 and 60")
    if contract.maximum_sequence_sessions != 60:
        raise ValueError("LSTM v1 maximum sequence must remain 60 sessions")
    if contract.prehistory_required_start > date(2022, 9, 1):
        raise ValueError("LSTM prehistory must begin no later than 2022-09-01")
    if contract.company_day_id_algorithm != (
        "sha256_pipe_join_company_day_stock_code_published_date"
    ):
        raise ValueError("unexpected company-day identity algorithm")
    if contract.sample_id_algorithm != (
        "sha256_pipe_join_sample_company_day_id_horizon"
    ):
        raise ValueError("unexpected sample identity algorithm")
    if contract.horizons_sessions != development_friendly_horizons(development):
        raise ValueError("LSTM horizons do not match the development target")
    if contract.missing_sequence_policy != (
        "preserve_identity_and_mark_unavailable"
    ):
        raise ValueError("missing sequences must preserve their canonical identity")
    if contract.sequence_columns != EXPECTED_SEQUENCE_COLUMNS:
        raise ValueError("LSTM v1 sequence columns changed")
    if any(not contract.feature_name_allowed(name) for name in contract.sequence_columns):
        raise ValueError("LSTM sequence contains a forbidden feature")
    normalization = raw["features"]["normalization"]
    if not normalization["fit_on_outer_fold_train_only"]:
        raise ValueError("LSTM normalization must fit on outer-fold train only")
    if normalization["full_sample_fitted_allowed"]:
        raise ValueError("full-sample LSTM normalization is forbidden")
    if not raw["features"]["labels_stored_separately"]:
        raise ValueError("LSTM labels must remain physically separate")
    if contract.model_family != "small_single_layer_lstm":
        raise ValueError("LSTM v1 family must remain a small single-layer LSTM")
    if contract.num_layers != 1 or contract.bidirectional:
        raise ValueError("LSTM v1 must be one-layer and unidirectional")
    if contract.hidden_size_candidates != (16, 32):
        raise ValueError("LSTM v1 hidden sizes must remain exactly 16 and 32")
    if contract.external_dropout_candidates != (0.1,):
        raise ValueError("LSTM v1 external dropout must remain 0.10")
    if contract.recurrent_dropout != 0.0:
        raise ValueError("single-layer recurrent dropout must remain zero")
    if raw["model"]["attention_allowed"] or raw["model"]["transformer_allowed"]:
        raise ValueError("attention and Transformer are blocked in LSTM v1")
    if contract.horizon_training != "separate_models":
        raise ValueError("LSTM v1 must train separate horizon models")
    if contract.interval_quantiles != (0.1, 0.9):
        raise ValueError("LSTM v1 interval quantiles must remain 0.10/0.90")
    if contract.baseline_statistics != (
        "last",
        "mean",
        "std",
        "minimum",
        "maximum",
        "slope",
    ):
        raise ValueError("LSTM v1 pooled baseline statistics changed")
    if contract.baseline_logistic_c != 0.1 or contract.baseline_ridge_alpha != 1.0:
        raise ValueError("LSTM v1 simple baseline regularization changed")
    if contract.baseline_random_state != contract.seed:
        raise ValueError("LSTM v1 baseline and training seeds must match")
    if contract.baseline_selection_policy != (
        "no_search_report_all_registered_lookbacks"
    ):
        raise ValueError("LSTM v1 baseline hyperparameter search is forbidden")
    if not raw["baselines"]["fit_preprocessors_on_outer_fold_train_only"]:
        raise ValueError("simple baseline preprocessing must remain fold-local")
    if not raw["baselines"]["use_identical_samples_across_lookbacks"]:
        raise ValueError("simple baseline lookbacks must use identical samples")
    if (contract.training_start, contract.training_end) != (
        split.train.start,
        split.train.end,
    ):
        raise ValueError("LSTM training dates must equal the train partition")
    if contract.random_split_allowed:
        raise ValueError("random LSTM splits are forbidden")
    if raw["training"]["split_strategy"] != "expanding_window":
        raise ValueError("LSTM v1 must use expanding-window validation")
    if not raw["training"]["fit_preprocessors_on_outer_fold_train_only"]:
        raise ValueError("LSTM preprocessors must fit on outer-fold train only")
    if not raw["training"]["outcome_must_mature_by_fold_end"]:
        raise ValueError("LSTM labels must mature by each fold cutoff")
    if contract.training_device != "cpu" or contract.cpu_threads != 4:
        raise ValueError("LSTM v1 deterministic CPU execution changed")
    if not contract.deterministic_algorithms:
        raise ValueError("LSTM v1 must enable deterministic algorithms")
    if contract.optimizer_name != "adamw":
        raise ValueError("LSTM v1 optimizer must remain AdamW")
    if contract.gradient_clip_norm != 1.0:
        raise ValueError("LSTM v1 gradient clipping changed")
    if not 0 < contract.minimum_epochs <= contract.maximum_epochs:
        raise ValueError("invalid LSTM early-stopping epoch bounds")
    if contract.early_stopping_metric != "weighted_multi_head_loss":
        raise ValueError("LSTM v1 early-stopping metric changed")
    if contract.refit_after_early_stopping:
        raise ValueError("LSTM v1 cannot refit on early-stop/calibration segments")
    if contract.huber_delta != 0.02:
        raise ValueError("LSTM v1 Huber delta changed")
    if (
        contract.direction_loss_weight,
        contract.return_loss_weight,
        contract.interval_loss_weight,
    ) != (1.0, 100.0, 10.0):
        raise ValueError("LSTM v1 registered multi-head loss weights changed")
    if contract.inner_split_unit != "unique_published_date":
        raise ValueError("LSTM inner split must operate on unique dates")
    if not contract.inner_outcome_maturity_required:
        raise ValueError("LSTM inner partitions must enforce outcome maturity")
    if any(value <= 0 for value in contract.inner_temporal_partitions):
        raise ValueError("inner temporal partitions must all be positive")
    if abs(sum(contract.inner_temporal_partitions) - 1.0) > 1e-12:
        raise ValueError("inner temporal partitions must sum to one")
    if contract.probability_calibration != "temperature_scaling":
        raise ValueError("LSTM v1 probability calibration changed")
    if contract.probability_temperature_bounds != (0.05, 10.0):
        raise ValueError("LSTM v1 temperature bounds changed")
    if not contract.calibration_identity_fallback:
        raise ValueError("LSTM calibration requires the preregistered identity fallback")
    if contract.interval_calibration != "split_conformal_interval_expansion":
        raise ValueError("LSTM v1 interval calibration changed")
    if contract.interval_target_coverage != 0.8:
        raise ValueError("LSTM v1 interval target coverage changed")
    if contract.interval_quantile_method != "finite_sample_higher":
        raise ValueError("LSTM v1 interval quantile method changed")
    if raw["calibration"]["outer_validation_fit_allowed"]:
        raise ValueError("outer validation cannot fit LSTM calibrators")
    if contract.output_fields != EXPECTED_OUTPUT_FIELDS:
        raise ValueError("LSTM output schema changed")
    if contract.risk_scale_definition != (
        "half_width_of_calibrated_return_interval"
    ):
        raise ValueError("unexpected LSTM risk-scale definition")
    if not 0.0 < contract.bullish_threshold < 1.0:
        raise ValueError("bullish threshold must be a probability")
    if contract.round_trip_cost_bps < 0:
        raise ValueError("round-trip cost cannot be negative")
    _validate_evaluation_gates(raw, development=development)
    if (contract.official_test_start, contract.official_test_end) != (
        split.test.start,
        split.test.end,
    ):
        raise ValueError("LSTM official-test dates do not match")
    if contract.official_test_read_limit != 1:
        raise ValueError("LSTM official test must have a one-read limit")
    if not raw["official_test"][
        "requires_frozen_model_features_preprocessors_calibrators_thresholds"
    ]:
        raise ValueError("official test requires a completely frozen LSTM candidate")
    if raw["official_test"]["tuning_after_read_allowed"]:
        raise ValueError("official-test tuning is forbidden")
    if (contract.quarantine_start, contract.quarantine_end) != (
        split.quarantine.start,
        split.quarantine.end,
    ):
        raise ValueError("LSTM quarantine dates do not match")
    if contract.forward_validation_start != split.forward_validation_start:
        raise ValueError("LSTM forward-validation start does not match")
    if raw["quarantine"]["fitting_allowed"] or raw["quarantine"][
        "model_selection_allowed"
    ]:
        raise ValueError("quarantine cannot enter LSTM development")
    if raw["forward_validation"]["training_allowed"] or raw[
        "forward_validation"
    ]["model_selection_allowed"]:
        raise ValueError("forward validation cannot enter LSTM development")


def development_friendly_horizons(
    development: CausalDevelopmentContract,
) -> tuple[int, ...]:
    """Return the fixed horizons while keeping the dependency explicit."""
    target_gate = development.module_gates.get("predictive_general")
    if not isinstance(target_gate, dict):
        raise ValueError("development contract lacks the general predictive gate")
    return (1, 3, 5)


def _validate_evaluation_gates(
    raw: dict,
    *,
    development: CausalDevelopmentContract,
) -> None:
    evaluation = raw["evaluation"]
    general = development.module_gates["predictive_general"]
    for key in (
        "minimum_oof_bullish_signals_per_horizon",
        "bullish_wilson_lower_must_exceed",
        "minimum_folds_with_bullish_hit_rate_above_half",
        "minimum_worst_fold_bullish_hit_rate",
        "cost_adjusted_mean_excess_return_ci_lower_must_exceed",
        "brier_score_must_beat_constant_base_rate",
    ):
        if evaluation[key] != general[key]:
            raise ValueError(f"LSTM evaluation gate differs from general gate: {key}")
    for required in (
        "return_mae_must_beat_fold_train_median",
        "interval_coverage_must_beat_constant_width_baseline",
        "simple_sequence_increment_required",
        "table_increment_required",
        "identical_samples_and_folds_required",
    ):
        if not evaluation[required]:
            raise ValueError(f"LSTM evaluation requirement disabled: {required}")
    module_gate = development.module_gates["time_series_model"]
    if not module_gate["all_bars_must_precede_prediction_as_of"]:
        raise ValueError("development contract must reject future LSTM bars")
    if not module_gate["incremental_oof_improvement_over_table_required"]:
        raise ValueError("development contract must require table-model increment")

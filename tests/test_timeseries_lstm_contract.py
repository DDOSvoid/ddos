from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.prediction.development_contract import load_causal_development_contract
from src.prediction.splits import load_prediction_split_contract
from src.prediction.timeseries.contract import load_timeseries_model_contract
from src.prediction.timeseries.lstm_contract import (
    EXPECTED_OUTPUT_FIELDS,
    EXPECTED_SEQUENCE_COLUMNS,
    load_lstm_timeseries_contract,
)


def test_lstm_contract_is_versioned_and_matches_current_governing_sources():
    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
    parent = load_timeseries_model_contract(
        split=split,
        development=development,
    )
    contract = load_lstm_timeseries_contract(
        split=split,
        development=development,
        parent=parent,
    )

    assert contract.contract == "causal-timeseries-lstm-daily-v1"
    assert contract.parent_contract_sha256 == parent.source_sha256
    assert contract.split_contract_sha256 == split.source_sha256
    assert contract.development_contract_sha256 == development.source_sha256
    assert contract.lookback_candidates_sessions == (20, 60)
    assert contract.hidden_size_candidates == (16, 32)
    assert contract.baseline_statistics == (
        "last",
        "mean",
        "std",
        "minimum",
        "maximum",
        "slope",
    )
    assert contract.baseline_logistic_c == 0.1
    assert contract.baseline_ridge_alpha == 1.0
    assert contract.training_device == "cpu"
    assert contract.deterministic_algorithms
    assert contract.optimizer_name == "adamw"
    assert contract.minimum_epochs == 5
    assert contract.early_stopping_metric == "weighted_multi_head_loss"
    assert contract.inner_split_unit == "unique_published_date"
    assert contract.inner_outcome_maturity_required
    assert contract.probability_temperature_bounds == (0.05, 10.0)
    assert contract.interval_calibration == "split_conformal_interval_expansion"
    assert contract.num_layers == 1
    assert not contract.bidirectional
    assert contract.sequence_columns == EXPECTED_SEQUENCE_COLUMNS
    assert contract.output_fields == EXPECTED_OUTPUT_FIELDS


def test_lstm_contract_rejects_unregistered_complex_model(tmp_path: Path):
    source = load_lstm_timeseries_contract().source_path
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["model"]["family"] = "transformer"
    raw["model"]["transformer_allowed"] = True
    modified = tmp_path / "lstm.yaml"
    modified.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="small single-layer LSTM"):
        load_lstm_timeseries_contract(modified)


def test_lstm_contract_forbids_future_and_outcome_features():
    contract = load_lstm_timeseries_contract()

    assert contract.feature_name_allowed("stock_return_1d")
    assert not contract.feature_name_allowed("future_excess_return")
    assert not contract.feature_name_allowed("outcome_available_at")
    assert not contract.feature_name_allowed("post_announcement_high")

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.prediction.timeseries.lstm_research_contract import (
    load_lstm_v2_research_contract,
)


def test_lstm_v2_research_contract_is_train_only_and_inherits_v1():
    contract = load_lstm_v2_research_contract()

    assert contract.contract == "causal-timeseries-lstm-daily-v2-research"
    assert contract.status == "hypothesis_only"
    assert contract.dataset_role == "train"
    assert contract.candidate_names() == tuple(
        f"lstm_v2_lb{lookback}_h{hidden}_d{horizon}"
        for lookback in (20, 60)
        for hidden in (16, 32)
        for horizon in (1, 3, 5)
    )
    assert tuple(contract.hypotheses) == (
        "target_scale",
        "shrinkable_interval",
        "brier_temperature",
    )
    assert contract.research_rules["official_test_read_limit"] == 0
    assert not contract.research_rules["release_allowed"]
    assert contract.objective["target_representation"] == (
        "fold_local_empirical_return_range"
    )
    assert contract.objective["return_range"]["fit_scope"] == (
        "outer_fold_train_only"
    )
    assert contract.objective["acceptable_metric_ranges"]["interval_coverage"] == [
        0.80,
        0.95,
    ]


def test_lstm_v2_research_contract_rejects_sealed_partition_read(tmp_path: Path):
    source = Path("config/timeseries_lstm_daily_v2_research.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["official_test_read_allowed"] = True
    modified = tmp_path / "lstm_v2.yaml"
    modified.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="sealed partitions"):
        load_lstm_v2_research_contract(modified)

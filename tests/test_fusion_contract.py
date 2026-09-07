from datetime import date
from pathlib import Path

import pytest
import yaml

from src.prediction.fusion.contract import load_expert_fusion_contract


def test_fusion_contract_specializes_experts_and_owns_ranking():
    contract = load_expert_fusion_contract()
    assert contract.universe == "event_driven_company_days"
    assert contract.development_contract == "causal-expert-ranking-development-v2"
    assert contract.expert_names == ("text", "tabular", "timeseries")
    assert contract.standalone_full_prediction_gate_required is False
    assert contract.dynamic_gating_allowed is False
    assert contract.ranking_group_by == "eligible_entry_date"
    assert contract.development_end == date(2024, 12, 31)
    assert contract.strict_oos_start == date(2026, 1, 1)
    assert contract.strict_oos_features_allowed_before_system_freeze is False
    assert contract.strict_oos_outcomes_allowed_before_prediction_lock is False
    assert len(contract.strict_oos_contract_sha256) == 64


def test_fusion_contract_rejects_full_standalone_gate(tmp_path: Path):
    source = Path("config/expert_fusion_v1.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["experts"]["standalone_full_prediction_gate_required"] = True
    candidate = tmp_path / "invalid.yaml"
    candidate.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must not be forced"):
        load_expert_fusion_contract(candidate)


def test_fusion_contract_rejects_unsealed_2026_outcomes(tmp_path: Path):
    source = Path("config/expert_fusion_v1.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["validation"]["strict_oos_outcomes_allowed_before_prediction_lock"] = True
    candidate = tmp_path / "invalid-oos.yaml"
    candidate.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="outcomes must remain sealed"):
        load_expert_fusion_contract(candidate)

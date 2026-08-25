"""Machine checks for the independent LSTM v2 research registration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT
from src.prediction.timeseries.lstm_contract import load_lstm_timeseries_contract

DEFAULT_LSTM_V2_RESEARCH_CONTRACT_PATH = (
    PROJECT_ROOT / "config" / "timeseries_lstm_daily_v2_research.yaml"
)

EXPECTED_HYPOTHESES = (
    "target_scale",
    "shrinkable_interval",
    "brier_temperature",
)


@dataclass(frozen=True)
class LstmV2ResearchContract:
    contract: str
    parent_contract: str
    parent_contract_sha256: str
    status: str
    dataset_role: str
    source_path: Path
    source_sha256: str
    candidate_space: dict[str, object]
    objective: dict[str, object]
    hypotheses: dict[str, dict[str, object]]
    research_rules: dict[str, object]

    def candidate_names(self) -> tuple[str, ...]:
        return tuple(
            f"lstm_v2_lb{lookback}_h{hidden}_d{horizon}"
            for lookback in self.candidate_space["lookback_sessions"]
            for hidden in self.candidate_space["hidden_size"]
            for horizon in self.candidate_space["horizon_sessions"]
        )


def _load_raw(path: Path) -> tuple[bytes, dict[str, object]]:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("LSTM v2 research contract must be a YAML mapping")
    return raw_bytes, raw


def load_lstm_v2_research_contract(
    path: Path = DEFAULT_LSTM_V2_RESEARCH_CONTRACT_PATH,
) -> LstmV2ResearchContract:
    raw_bytes, raw = _load_raw(path)
    parent_path = (PROJECT_ROOT / str(raw["parent_contract_path"])).resolve()
    parent = load_lstm_timeseries_contract(parent_path)
    contract = LstmV2ResearchContract(
        contract=str(raw["contract"]),
        parent_contract=str(raw["parent_contract"]),
        parent_contract_sha256=str(raw["parent_contract_sha256"]),
        status=str(raw["status"]),
        dataset_role=str(raw["dataset_role"]),
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        candidate_space=dict(raw["candidate_space"]),
        objective=dict(raw["objective"]),
        hypotheses={
            str(name): dict(value)
            for name, value in dict(raw["hypotheses"]).items()
        },
        research_rules=dict(raw["research_rules"]),
    )
    validate_lstm_v2_research_contract(contract, raw=raw, parent=parent)
    return contract


def validate_lstm_v2_research_contract(
    contract: LstmV2ResearchContract,
    *,
    raw: dict[str, object],
    parent,
) -> None:
    if contract.contract != "causal-timeseries-lstm-daily-v2-research":
        raise ValueError("unexpected LSTM v2 research contract name")
    if contract.parent_contract != parent.contract:
        raise ValueError("LSTM v2 parent contract name does not match")
    if contract.parent_contract_sha256 != parent.source_sha256:
        raise ValueError("LSTM v2 parent contract hash does not match")
    if contract.status != "hypothesis_only":
        raise ValueError("LSTM v2 must remain hypothesis-only before passing gates")
    if contract.dataset_role != "train":
        raise ValueError("LSTM v2 research may only read the train role")
    if any(
        bool(raw[key])
        for key in (
            "official_test_read_allowed",
            "quarantine_read_allowed",
            "forward_validation_read_allowed",
        )
    ):
        raise PermissionError("LSTM v2 research cannot read sealed partitions")
    space = contract.candidate_space
    if tuple(space["lookback_sessions"]) != (20, 60):
        raise ValueError("LSTM v2 lookbacks must inherit v1 exactly")
    if tuple(space["hidden_size"]) != (16, 32):
        raise ValueError("LSTM v2 hidden sizes must inherit v1 exactly")
    if tuple(space["horizon_sessions"]) != (1, 3, 5):
        raise ValueError("LSTM v2 horizons must inherit v1 exactly")
    if int(space["num_layers"]) != 1 or bool(space["bidirectional"]):
        raise ValueError("LSTM v2 may not increase model complexity")
    if bool(space["attention_allowed"]) or bool(space["transformer_allowed"]):
        raise ValueError("LSTM v2 attention/Transformer changes are forbidden")
    if tuple(space["external_dropout"]) != (0.10,) or float(space["recurrent_dropout"]) != 0.0:
        raise ValueError("LSTM v2 dropout must inherit v1 exactly")
    if bool(space["hyperparameter_search"]):
        raise ValueError("LSTM v2 hyperparameter search is forbidden")
    objective = contract.objective
    if objective["target_representation"] != "fold_local_empirical_return_range":
        raise ValueError("LSTM v2 target must be a fold-local return range")
    return_range = dict(objective["return_range"])
    lower_quantile = float(return_range["lower_quantile"])
    upper_quantile = float(return_range["upper_quantile"])
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("LSTM v2 return range quantiles must be ordered in (0, 1)")
    if return_range["fit_scope"] != "outer_fold_train_only":
        raise ValueError("LSTM v2 return range must fit on outer-fold train only")
    if int(return_range["minimum_samples"]) < 1:
        raise ValueError("LSTM v2 return range needs a positive sample minimum")
    metric_ranges = dict(objective["acceptable_metric_ranges"])
    expected_metric_ranges = {
        "interval_coverage": (0.80, 0.95),
        "interval_width_ratio_to_constant": (0.80, 1.00),
        "brier_improvement_over_constant": (0.00, 1.00),
        "return_mae_improvement_over_fold_median": (0.00, 1.00),
    }
    if set(metric_ranges) != set(expected_metric_ranges):
        raise ValueError("LSTM v2 metric ranges changed or are incomplete")
    for name, expected in expected_metric_ranges.items():
        bounds = tuple(float(value) for value in metric_ranges[name])
        if bounds != expected or bounds[0] > bounds[1]:
            raise ValueError(f"invalid LSTM v2 acceptable range: {name}")
    if objective["range_inclusive"] is not True:
        raise ValueError("LSTM v2 objective ranges must be inclusive")
    if tuple(contract.hypotheses) != EXPECTED_HYPOTHESES:
        raise ValueError("LSTM v2 hypotheses must remain in preregistered order")
    for name in EXPECTED_HYPOTHESES:
        hypothesis = contract.hypotheses[name]
        if int(hypothesis["registered_order"]) != EXPECTED_HYPOTHESES.index(name) + 1:
            raise ValueError(f"invalid registration order for {name}")
        if hypothesis["complexity_change"] not in {"none", "calibration_only"}:
            raise ValueError(f"complexity change is not allowed for {name}")
    rules = contract.research_rules
    for key in (
        "one_change_at_a_time",
        "all_registered_candidates_must_run",
        "outer_validation_fit_allowed",
        "freeze_allowed",
        "table_oof_read_allowed",
        "release_allowed",
        "parent_v1_rewrite_allowed",
    ):
        if key in {"one_change_at_a_time", "all_registered_candidates_must_run"}:
            if not bool(rules[key]):
                raise ValueError(f"LSTM v2 rule disabled: {key}")
        elif bool(rules[key]):
            raise ValueError(f"LSTM v2 rule must remain disabled: {key}")
    if int(rules["official_test_read_limit"]) != 0:
        raise ValueError("LSTM v2 official-test read limit must remain zero")
    if raw["inherit"] != {
        "dates_and_split": "exact_parent_contract",
        "sequence_identity": "exact_parent_contract",
        "feature_columns": "exact_parent_contract",
        "sample_and_fold_set": "exact_parent_contract",
        "evaluation_gates": "exact_parent_contract",
    }:
        raise ValueError("LSTM v2 must inherit v1 data and gates exactly")

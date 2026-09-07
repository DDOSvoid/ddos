"""Machine-readable contract for specialist signals and stock ranking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT
from src.prediction.strict_oos import load_strict_oos_contract

CONTRACT_NAME = "causal-event-driven-expert-fusion-ranking-v1"
EXPERT_NAMES = ("text", "tabular", "timeseries")
HORIZONS = (1, 3, 5)


@dataclass(frozen=True)
class ExpertFusionContract:
    source_path: Path
    source_sha256: str
    contract: str
    development_contract: str
    status: str
    universe: str
    decision_local_time: str
    horizons_sessions: tuple[int, ...]
    expert_names: tuple[str, ...]
    required_expert_outputs: tuple[str, ...]
    minimum_available_experts: int
    standalone_full_prediction_gate_required: bool
    dynamic_gating_allowed: bool
    ranking_group_by: str
    risk_aversion: float
    round_trip_cost_bps: int
    top_k: tuple[int, ...]
    development_dataset_role: str
    development_end: date
    development_labels_require_explicit_role: bool
    strict_oos_start: date
    strict_oos_features_allowed_before_system_freeze: bool
    strict_oos_outcomes_allowed_before_prediction_lock: bool
    strict_oos_contract_sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_expert_fusion_contract(
    path: Path | None = None,
) -> ExpertFusionContract:
    source_path = path or PROJECT_ROOT / "config" / "expert_fusion_v1.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    scope = raw.get("scope") or {}
    experts = raw.get("experts") or {}
    fusion = raw.get("fusion") or {}
    ranking = raw.get("ranking") or {}
    validation = raw.get("validation") or {}
    strict_oos_path = PROJECT_ROOT / str(validation.get("strict_oos_contract") or "")
    strict_oos = load_strict_oos_contract(strict_oos_path)

    contract = ExpertFusionContract(
        source_path=source_path,
        source_sha256=_sha256(source_path),
        contract=str(raw.get("contract") or ""),
        development_contract=str(raw.get("development_contract") or ""),
        status=str(raw.get("status") or ""),
        universe=str(scope.get("universe") or ""),
        decision_local_time=str(scope.get("local_time") or ""),
        horizons_sessions=tuple(int(value) for value in scope.get("horizons_sessions", [])),
        expert_names=tuple(str(value) for value in experts.get("required_names", [])),
        required_expert_outputs=tuple(str(value) for value in experts.get("required_outputs", [])),
        minimum_available_experts=int(fusion.get("minimum_available_experts", 0)),
        standalone_full_prediction_gate_required=bool(
            experts.get("standalone_full_prediction_gate_required", True)
        ),
        dynamic_gating_allowed=bool(fusion.get("dynamic_gating_allowed", True)),
        ranking_group_by=str(ranking.get("group_by") or ""),
        risk_aversion=float(ranking.get("risk_aversion", 0.0)),
        round_trip_cost_bps=int(ranking.get("round_trip_cost_bps", 0)),
        top_k=tuple(int(value) for value in ranking.get("top_k", [])),
        development_dataset_role=str(validation.get("development_dataset_role") or ""),
        development_end=date.fromisoformat(str(validation.get("development_end"))),
        development_labels_require_explicit_role=bool(
            validation.get("development_labels_require_explicit_role", False)
        ),
        strict_oos_start=date.fromisoformat(str(validation.get("strict_oos_start"))),
        strict_oos_features_allowed_before_system_freeze=bool(
            validation.get("strict_oos_features_allowed_before_system_freeze", True)
        ),
        strict_oos_outcomes_allowed_before_prediction_lock=bool(
            validation.get("strict_oos_outcomes_allowed_before_prediction_lock", True)
        ),
        strict_oos_contract_sha256=strict_oos.source_sha256,
    )
    _validate_contract(contract)
    return contract


def _validate_contract(contract: ExpertFusionContract) -> None:
    if contract.contract != CONTRACT_NAME:
        raise ValueError(f"unexpected fusion contract: {contract.contract}")
    if contract.development_contract != "causal-expert-ranking-development-v2":
        raise ValueError("fusion v1 must use the expert-ranking development contract")
    if contract.status != "research_only":
        raise ValueError("fusion v1 must remain research_only until frozen")
    if contract.universe != "event_driven_company_days":
        raise ValueError("fusion v1 universe must be event-driven company-days")
    if contract.horizons_sessions != HORIZONS:
        raise ValueError("fusion horizons must remain 1/3/5 sessions")
    if contract.expert_names != EXPERT_NAMES:
        raise ValueError("fusion experts must be text/tabular/timeseries")
    if contract.standalone_full_prediction_gate_required:
        raise ValueError("specialists must not be forced through the full prediction gate")
    if contract.dynamic_gating_allowed:
        raise ValueError("dynamic gating is outside fusion v1")
    if not 1 <= contract.minimum_available_experts <= len(EXPERT_NAMES):
        raise ValueError("minimum_available_experts is invalid")
    if contract.ranking_group_by != "eligible_entry_date":
        raise ValueError("ranking must be grouped by eligible entry date")
    if contract.risk_aversion < 0 or contract.round_trip_cost_bps < 0:
        raise ValueError("risk aversion and costs must be non-negative")
    if not contract.top_k or any(value <= 0 for value in contract.top_k):
        raise ValueError("top_k must contain positive values")
    if contract.development_dataset_role != "train":
        raise ValueError("fusion development may use only the train role")
    if contract.development_end != date(2024, 12, 31):
        raise ValueError("fusion development must end on 2024-12-31")
    if not contract.development_labels_require_explicit_role:
        raise ValueError("development labels must carry an explicit train role")
    if contract.strict_oos_start != date(2026, 1, 1):
        raise ValueError("strict OOS must start on 2026-01-01")
    if contract.strict_oos_features_allowed_before_system_freeze:
        raise ValueError("2026 features must remain sealed before system freeze")
    if contract.strict_oos_outcomes_allowed_before_prediction_lock:
        raise ValueError("2026 outcomes must remain sealed before prediction lock")

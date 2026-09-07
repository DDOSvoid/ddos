"""Machine-readable seal for the 2026 expert-fusion outcome holdout."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT

CONTRACT = "strict-stock-ranking-oos-2026-v1"
DEFAULT_PATH = PROJECT_ROOT / "config" / "strict_oos_2026_v1.yaml"


@dataclass(frozen=True)
class StrictOOSContract:
    source_path: Path
    source_sha256: str
    contract: str
    status: str
    start: date
    end: date
    development_end: date
    feature_rows_allowed_pre_freeze: bool
    outcomes_allowed_pre_freeze: bool
    outcomes_allowed_before_prediction_lock: bool
    evaluation_count: int
    tuning_from_results_allowed: bool
    repository_wide_pristine_claim_allowed: bool


def load_strict_oos_contract(path: Path = DEFAULT_PATH) -> StrictOOSContract:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    window = raw.get("window") or {}
    development = raw.get("development_boundary") or {}
    pre_freeze = raw.get("pre_freeze_seal") or {}
    post_freeze = raw.get("post_freeze_protocol") or {}
    claim_scope = raw.get("claim_scope") or {}
    contract = StrictOOSContract(
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        contract=str(raw.get("contract") or ""),
        status=str(raw.get("status") or ""),
        start=date.fromisoformat(str(window.get("start"))),
        end=date.fromisoformat(str(window.get("end"))),
        development_end=date.fromisoformat(str(development.get("end"))),
        feature_rows_allowed_pre_freeze=bool(pre_freeze.get("feature_rows_2026_allowed", True)),
        outcomes_allowed_pre_freeze=any(
            bool(pre_freeze.get(name, True))
            for name in (
                "realized_stock_returns_2026_allowed",
                "market_direction_labels_2026_allowed",
                "ranking_metrics_2026_allowed",
                "aggregate_outcome_counts_2026_allowed",
            )
        ),
        outcomes_allowed_before_prediction_lock=bool(
            post_freeze.get("outcome_access_allowed_before_all_predictions_locked", True)
        ),
        evaluation_count=int(post_freeze.get("evaluation_count") or 0),
        tuning_from_results_allowed=bool(
            post_freeze.get("tuning_or_retraining_from_2026_results_allowed", True)
        ),
        repository_wide_pristine_claim_allowed=bool(
            claim_scope.get("repository_wide_never_touched_2026_claim_allowed", True)
        ),
    )
    validate_strict_oos_contract(contract)
    return contract


def validate_strict_oos_contract(contract: StrictOOSContract) -> None:
    if contract.contract != CONTRACT:
        raise ValueError(f"unexpected strict OOS contract: {contract.contract}")
    if contract.status != "sealed_before_system_freeze":
        raise ValueError("2026 strict OOS contract must remain sealed before freeze")
    if contract.start != date(2026, 1, 1) or contract.end != date(2026, 12, 31):
        raise ValueError("strict OOS window must equal calendar year 2026")
    if contract.development_end != date(2024, 12, 31):
        raise ValueError("strict OOS development boundary must end in 2024")
    if contract.feature_rows_allowed_pre_freeze or contract.outcomes_allowed_pre_freeze:
        raise ValueError("2026 inputs and outcomes must remain sealed before freeze")
    if contract.outcomes_allowed_before_prediction_lock:
        raise ValueError("2026 outcomes require a locked prediction ledger")
    if contract.evaluation_count != 1:
        raise ValueError("2026 is a one-time strict OOS evaluation")
    if contract.tuning_from_results_allowed:
        raise ValueError("2026 results may not tune or retrain the system")
    if contract.repository_wide_pristine_claim_allowed:
        raise ValueError("legacy 2026 metadata audits prohibit a repository-pristine claim")


def audit_current_database(database_path: Path) -> dict[str, int | bool]:
    """Count 2026 rows without selecting any outcome value columns."""
    queries = {
        "daily_prices_2026": ("SELECT COUNT(*) FROM daily_prices WHERE trade_date >= '2026-01-01'"),
        "market_targets_2026": (
            "SELECT COUNT(*) FROM announcement_market_targets t "
            "JOIN announcements a ON a.id = t.announcement_id "
            "WHERE a.published_date >= '2026-01-01'"
        ),
        "fused_predictions_2026": (
            "SELECT COUNT(*) FROM fused_stock_predictions WHERE eligible_entry_date >= '2026-01-01'"
        ),
        "fused_outcomes_2026": (
            "SELECT COUNT(*) FROM fused_prediction_outcomes o "
            "JOIN fused_stock_predictions p ON p.id = o.prediction_id "
            "WHERE p.eligible_entry_date >= '2026-01-01'"
        ),
    }
    counts: dict[str, int] = {}
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        requirements = {
            "daily_prices_2026": {"daily_prices"},
            "market_targets_2026": {"announcement_market_targets", "announcements"},
            "fused_predictions_2026": {"fused_stock_predictions"},
            "fused_outcomes_2026": {
                "fused_prediction_outcomes",
                "fused_stock_predictions",
            },
        }
        for name, query in queries.items():
            counts[name] = (
                int(connection.execute(query).fetchone()[0]) if requirements[name] <= tables else 0
            )
    return {
        **counts,
        "current_architecture_store_clean": all(value == 0 for value in counts.values()),
    }

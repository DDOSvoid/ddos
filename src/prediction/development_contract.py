"""Machine-enforced causal contract for modular model development."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from src.config import PROJECT_ROOT
from src.prediction.splits import (
    PredictionSplitContract,
    load_prediction_split_contract,
)

DEFAULT_DEVELOPMENT_CONTRACT_PATH = (
    PROJECT_ROOT / "config" / "causal_model_development.yaml"
)


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date


@dataclass(frozen=True)
class CausalDevelopmentContract:
    contract: str
    timezone: str
    split_contract: str
    decision_name: str
    decision_time: time
    historical_market_daily_rule: str
    folds: tuple[WalkForwardFold, ...]
    official_test_read_limit: int
    module_gates: dict
    source_path: Path
    source_sha256: str

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def require_available(
        self,
        *,
        feature_name: str,
        available_at: datetime,
        prediction_as_of: datetime,
    ) -> None:
        """Reject any feature that became available after the prediction cutoff."""
        available = _require_aware(available_at, "available_at")
        cutoff = _require_aware(prediction_as_of, "prediction_as_of")
        if available > cutoff:
            raise ValueError(
                f"future feature rejected: {feature_name} available at "
                f"{available.isoformat()} after as_of {cutoff.isoformat()}"
            )

    def historical_daily_bar_allowed(
        self,
        *,
        trade_date: date,
        announcement_published_date: date,
    ) -> bool:
        """Date-only history uses the conservative strictly-before rule."""
        return trade_date < announcement_published_date

    def fold_role(
        self,
        *,
        fold_name: str,
        published_date: date,
        outcome_available_at: datetime,
    ) -> str:
        """Classify an observation within a fold, including maturity cutoffs."""
        fold = next((item for item in self.folds if item.name == fold_name), None)
        if fold is None:
            raise ValueError(f"unknown walk-forward fold: {fold_name}")
        available = _require_aware(outcome_available_at, "outcome_available_at")
        if fold.train_start <= published_date <= fold.train_end:
            return (
                "fold_train"
                if available <= _end_of_day_utc(fold.train_end, self.zone)
                else "fold_train_outcome_after_cutoff"
            )
        if fold.validation_start <= published_date <= fold.validation_end:
            return (
                "fold_validation"
                if available <= _end_of_day_utc(fold.validation_end, self.zone)
                else "fold_validation_outcome_after_cutoff"
            )
        return "outside_fold"


def _require_aware(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _end_of_day_utc(day: date, zone: ZoneInfo) -> datetime:
    return datetime.combine(day, time(23, 59, 59), tzinfo=zone).astimezone(UTC)


def load_causal_development_contract(
    path: Path = DEFAULT_DEVELOPMENT_CONTRACT_PATH,
    *,
    split: PredictionSplitContract | None = None,
) -> CausalDevelopmentContract:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    folds = tuple(
        WalkForwardFold(
            name=str(item["name"]),
            train_start=date.fromisoformat(item["train_start"]),
            train_end=date.fromisoformat(item["train_end"]),
            validation_start=date.fromisoformat(item["validation_start"]),
            validation_end=date.fromisoformat(item["validation_end"]),
        )
        for item in raw["internal_walk_forward"]["folds"]
    )
    contract = CausalDevelopmentContract(
        contract=str(raw["contract"]),
        timezone=str(raw["timezone"]),
        split_contract=str(raw["split_contract"]),
        decision_name=str(raw["decision_point"]["name"]),
        decision_time=time.fromisoformat(raw["decision_point"]["local_time"]),
        historical_market_daily_rule=str(
            raw["historical_replay"]["market_daily_rule"]
        ),
        folds=folds,
        official_test_read_limit=int(raw["official_test"]["read_limit"]),
        module_gates=dict(raw["module_gates"]),
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    validate_causal_development_contract(
        contract,
        split=split or load_prediction_split_contract(),
        raw=raw,
    )
    return contract


def validate_causal_development_contract(
    contract: CausalDevelopmentContract,
    *,
    split: PredictionSplitContract,
    raw: dict,
) -> None:
    if contract.split_contract != split.contract:
        raise ValueError("development and temporal split contracts do not match")
    if contract.official_test_read_limit != 1:
        raise ValueError("official test must have exactly one permitted read")
    if raw["internal_walk_forward"]["random_split_allowed"]:
        raise ValueError("random split is forbidden")
    if not raw["internal_walk_forward"]["fit_preprocessors_on_fold_train_only"]:
        raise ValueError("preprocessors must be fitted on each fold's train data only")
    if not raw["internal_walk_forward"]["outcome_must_mature_by_fold_end"]:
        raise ValueError("fold outcomes must mature by the fold cutoff")
    if raw["official_test"]["tuning_after_read_allowed"]:
        raise ValueError("official-test tuning must be forbidden")
    if raw["official_test"]["quarantine_allowed"]:
        raise ValueError("quarantine data must not enter the official test")
    if raw["official_test"]["forward_validation_allowed"]:
        raise ValueError("forward-validation data must not enter the official test")
    if not contract.folds:
        raise ValueError("at least one walk-forward fold is required")

    previous_train_end = None
    previous_validation_end = None
    for fold in contract.folds:
        if not (
            split.train.start <= fold.train_start <= fold.train_end <= split.train.end
        ):
            raise ValueError(f"{fold.name} train range leaves the official train partition")
        if not (
            split.train.start
            <= fold.validation_start
            <= fold.validation_end
            <= split.train.end
        ):
            raise ValueError(
                f"{fold.name} validation range leaves the official train partition"
            )
        if fold.train_end >= fold.validation_start:
            raise ValueError(f"{fold.name} train and validation ranges overlap")
        if fold.train_start != split.train.start:
            raise ValueError(f"{fold.name} is not an expanding-window fold")
        if previous_train_end is not None and fold.train_end <= previous_train_end:
            raise ValueError("walk-forward train windows do not expand")
        if (
            previous_validation_end is not None
            and fold.validation_start <= previous_validation_end
        ):
            raise ValueError("walk-forward validation windows overlap")
        previous_train_end = fold.train_end
        previous_validation_end = fold.validation_end

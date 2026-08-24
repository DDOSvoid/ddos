"""Machine-enforced temporal split contract for impact prediction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from src.config import PROJECT_ROOT

SHANGHAI = ZoneInfo("Asia/Shanghai")
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "config" / "prediction_splits.yaml"


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def contains(self, value: date) -> bool:
        return self.start <= value <= self.end


@dataclass(frozen=True)
class PredictionSplitContract:
    contract: str
    train: DateRange
    test: DateRange
    quarantine: DateRange
    forward_validation_start: date
    test_must_not_extend_beyond: date
    source_path: Path
    source_sha256: str

    def role_for(
        self,
        published_date: date,
        outcome_available_at: datetime | None = None,
    ) -> str:
        if self.train.contains(published_date):
            role = "train"
            end = self.train.end
        elif self.test.contains(published_date):
            role = "test"
            end = self.test.end
        elif self.quarantine.contains(published_date):
            return "quarantine"
        elif published_date >= self.forward_validation_start:
            return "forward_validation"
        else:
            return "out_of_contract"

        if outcome_available_at is not None:
            available = (
                outcome_available_at.replace(tzinfo=UTC)
                if outcome_available_at.tzinfo is None
                else outcome_available_at.astimezone(UTC)
            )
            partition_end = datetime.combine(
                end,
                time(23, 59, 59),
                tzinfo=SHANGHAI,
            ).astimezone(UTC)
            if available > partition_end:
                return f"{role}_outcome_after_cutoff"
        return role


def load_prediction_split_contract(
    path: Path = DEFAULT_SPLIT_PATH,
) -> PredictionSplitContract:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    contract = PredictionSplitContract(
        contract=str(raw["contract"]),
        train=DateRange(
            date.fromisoformat(raw["train"]["start"]),
            date.fromisoformat(raw["train"]["end"]),
        ),
        test=DateRange(
            date.fromisoformat(raw["test"]["start"]),
            date.fromisoformat(raw["test"]["end"]),
        ),
        quarantine=DateRange(
            date.fromisoformat(raw["quarantine"]["start"]),
            date.fromisoformat(raw["quarantine"]["end"]),
        ),
        forward_validation_start=date.fromisoformat(
            raw["forward_validation"]["start"]
        ),
        test_must_not_extend_beyond=date.fromisoformat(
            raw["rules"]["test_must_not_extend_beyond"]
        ),
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    validate_prediction_split_contract(contract)
    return contract


def validate_prediction_split_contract(contract: PredictionSplitContract) -> None:
    if contract.train.end >= contract.test.start:
        raise ValueError("train and test overlap or are not chronological")
    if contract.test.end >= contract.quarantine.start:
        raise ValueError("test and quarantine overlap")
    if contract.quarantine.end >= contract.forward_validation_start:
        raise ValueError("quarantine and forward validation overlap")
    if contract.test.end > contract.test_must_not_extend_beyond:
        raise ValueError("test contains data after the permitted 2025 boundary")

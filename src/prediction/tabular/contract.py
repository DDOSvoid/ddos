"""Machine-enforced contract for the tabular return/risk model family."""

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

DEFAULT_TABULAR_CONTRACT_PATH = PROJECT_ROOT / "config" / "tabular_model.yaml"


@dataclass(frozen=True)
class TabularModelContract:
    contract: str
    split_contract: str
    development_contract: str
    source_directory: Path
    data_directory: Path
    model_directory: Path
    feature_frequency: str
    raw_intraday_frequency: str
    raw_intraday_usage: str
    minute_level_prediction_status: str
    model_families: tuple[str, ...]
    required_availability_metadata: bool
    forbidden_feature_names: frozenset[str]
    forbidden_feature_prefixes: tuple[str, ...]
    training_start: date
    training_end: date
    split_strategy: str
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


def load_tabular_model_contract(
    path: Path = DEFAULT_TABULAR_CONTRACT_PATH,
    *,
    split: PredictionSplitContract | None = None,
    development: CausalDevelopmentContract | None = None,
) -> TabularModelContract:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    contract = TabularModelContract(
        contract=str(raw["contract"]),
        split_contract=str(raw["split_contract"]),
        development_contract=str(raw["development_contract"]),
        source_directory=_project_path(raw["directories"]["source"]),
        data_directory=_project_path(raw["directories"]["data"]),
        model_directory=_project_path(raw["directories"]["models"]),
        feature_frequency=str(raw["sampling"]["feature_frequency"]),
        raw_intraday_frequency=str(raw["sampling"]["raw_intraday_frequency"]),
        raw_intraday_usage=str(raw["sampling"]["raw_intraday_usage"]),
        minute_level_prediction_status=str(
            raw["sampling"]["minute_level_prediction_status"]
        ),
        model_families=tuple(str(item) for item in raw["model_families"]),
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
    validate_tabular_model_contract(
        contract,
        raw=raw,
        split=split or load_prediction_split_contract(),
        development=development or load_causal_development_contract(),
    )
    return contract


def validate_tabular_model_contract(
    contract: TabularModelContract,
    *,
    raw: dict,
    split: PredictionSplitContract,
    development: CausalDevelopmentContract,
) -> None:
    if contract.split_contract != split.contract:
        raise ValueError("tabular and temporal split contracts do not match")
    if contract.development_contract != development.contract:
        raise ValueError("tabular and causal development contracts do not match")
    if (contract.training_start, contract.training_end) != (
        split.train.start,
        split.train.end,
    ):
        raise ValueError("tabular training dates must equal the train partition")
    if (contract.official_test_start, contract.official_test_end) != (
        split.test.start,
        split.test.end,
    ):
        raise ValueError("tabular official-test dates must equal the test partition")
    if (contract.quarantine_start, contract.quarantine_end) != (
        split.quarantine.start,
        split.quarantine.end,
    ):
        raise ValueError("tabular quarantine dates must equal the quarantine partition")
    if contract.forward_validation_start != split.forward_validation_start:
        raise ValueError("tabular forward-validation start does not match split contract")
    if contract.feature_frequency != "daily":
        raise ValueError("v1 tabular features must remain daily")
    if contract.raw_intraday_frequency != "15min":
        raise ValueError("tabular v1 raw intraday frequency must be 15min")
    if contract.raw_intraday_usage != "lagged_daily_aggregates_only":
        raise ValueError("intraday bars may only produce lagged daily aggregates")
    if contract.minute_level_prediction_status != (
        "blocked_pending_authoritative_announcement_timestamp"
    ):
        raise ValueError("minute-level prediction must remain blocked in tabular v1")
    if contract.split_strategy != "expanding_window":
        raise ValueError("tabular model must use expanding-window validation")
    if raw["training"]["random_split_allowed"]:
        raise ValueError("random tabular splits are forbidden")
    if not raw["training"]["fit_preprocessors_on_fold_train_only"]:
        raise ValueError("tabular preprocessors must fit on fold-train only")
    if set(contract.model_families) != {"lightgbm", "catboost"}:
        raise ValueError("tabular v1 model families must be LightGBM and CatBoost")
    if not contract.required_availability_metadata:
        raise ValueError("every tabular feature requires availability metadata")
    if not raw["labels"]["stored_separately_from_features"]:
        raise ValueError("tabular labels must be physically separate from features")
    if not raw["official_test"]["requires_frozen_model_and_hyperparameters"]:
        raise ValueError("official test requires frozen models and hyperparameters")
    if raw["official_test"]["tuning_after_read_allowed"]:
        raise ValueError("official-test tuning is forbidden")
    if contract.official_test_read_limit != 1:
        raise ValueError("official test must have a one-read limit")
    if raw["quarantine"]["model_selection_allowed"]:
        raise ValueError("quarantine cannot be used for model selection")
    if raw["forward_validation"]["training_allowed"]:
        raise ValueError("forward validation cannot be used for training")
    if raw["forward_validation"]["model_selection_allowed"]:
        raise ValueError("forward validation cannot be used for model selection")

    expected_directories = {
        contract.source_directory: PROJECT_ROOT / "src" / "prediction" / "tabular",
        contract.data_directory: PROJECT_ROOT / "data" / "tabular",
        contract.model_directory: PROJECT_ROOT / "models" / "tabular",
    }
    for actual, expected in expected_directories.items():
        if actual != expected.resolve():
            raise ValueError(f"unexpected tabular directory: {actual}")

"""Versioned feature, risk, and output catalog for tabular v2 research."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT
from src.prediction.development_contract import CausalDevelopmentContract
from src.prediction.splits import PredictionSplitContract
from src.prediction.tabular.contract import TabularModelContract
from src.prediction.tabular.features import FeatureSpec

DEFAULT_FEATURE_CATALOG_PATH = PROJECT_ROOT / "config" / "tabular_feature_catalog_v2.yaml"
FEATURE_CATALOG_CONTRACT = "causal-tabular-feature-catalog-v2"
V2_DATASET_CONTRACT = "causal-tabular-train-dataset-v2"


@dataclass(frozen=True)
class CatalogFeature:
    name: str
    group: str
    source: str
    available_at_column: str
    observation_date_column: str | None
    unit: str
    transformation: str
    missing_policy: str

    def as_feature_spec(self) -> FeatureSpec:
        return FeatureSpec(
            name=self.name,
            source=self.source,
            available_at_column=self.available_at_column,
            observation_date_column=self.observation_date_column,
        )


@dataclass(frozen=True)
class TabularFeatureCatalog:
    contract: str
    dataset_contract: str
    base_tabular_contract: str
    split_contract: str
    development_contract: str
    dataset_role: str
    official_test_allowed: bool
    primary_risk_target: str
    primary_risk_horizons: tuple[int, ...]
    primary_risk_output: str
    unified_output_schema: str
    unified_output_fields: tuple[str, ...]
    blocked_feature_groups: tuple[tuple[str, str], ...]
    features: tuple[CatalogFeature, ...]
    source_path: Path
    source_sha256: str

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.features)

    @property
    def feature_specs(self) -> tuple[FeatureSpec, ...]:
        return tuple(item.as_feature_spec() for item in self.features)

    @property
    def feature_groups(self) -> dict[str, tuple[str, ...]]:
        groups: dict[str, list[str]] = {}
        for item in self.features:
            groups.setdefault(item.group, []).append(item.name)
        return {name: tuple(values) for name, values in groups.items()}


def load_tabular_feature_catalog(
    path: Path = DEFAULT_FEATURE_CATALOG_PATH,
    *,
    tabular: TabularModelContract,
    split: PredictionSplitContract,
    development: CausalDevelopmentContract,
) -> TabularFeatureCatalog:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    features: list[CatalogFeature] = []
    for group_name, group in raw["feature_groups"].items():
        for name, item in group["features"].items():
            features.append(
                CatalogFeature(
                    name=str(name),
                    group=str(group_name),
                    source=str(group["source"]),
                    available_at_column=str(group["available_at_column"]),
                    observation_date_column=(
                        str(group["observation_date_column"])
                        if group.get("observation_date_column") is not None
                        else None
                    ),
                    unit=str(item["unit"]),
                    transformation=str(item["transformation"]),
                    missing_policy=str(group["missing_policy"]),
                )
            )
    catalog = TabularFeatureCatalog(
        contract=str(raw["contract"]),
        dataset_contract=str(raw["dataset_contract"]),
        base_tabular_contract=str(raw["base_tabular_contract"]),
        split_contract=str(raw["split_contract"]),
        development_contract=str(raw["development_contract"]),
        dataset_role=str(raw["dataset_role"]),
        official_test_allowed=bool(raw["official_test_allowed"]),
        primary_risk_target=str(raw["primary_risk"]["target"]),
        primary_risk_horizons=tuple(
            int(value) for value in raw["primary_risk"]["horizons_sessions"]
        ),
        primary_risk_output=str(raw["primary_risk"]["output"]),
        unified_output_schema=str(raw["unified_output"]["schema"]),
        unified_output_fields=tuple(
            str(value) for value in raw["unified_output"]["required_fields"]
        ),
        blocked_feature_groups=tuple(
            (str(name), str(reason))
            for name, reason in raw.get("blocked_feature_groups", {}).items()
        ),
        features=tuple(features),
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    validate_tabular_feature_catalog(
        catalog,
        tabular=tabular,
        split=split,
        development=development,
    )
    return catalog


def validate_tabular_feature_catalog(
    catalog: TabularFeatureCatalog,
    *,
    tabular: TabularModelContract,
    split: PredictionSplitContract,
    development: CausalDevelopmentContract,
) -> None:
    if catalog.contract != FEATURE_CATALOG_CONTRACT:
        raise ValueError("unexpected tabular feature catalog contract")
    if catalog.dataset_contract != V2_DATASET_CONTRACT:
        raise ValueError("unexpected tabular v2 dataset contract")
    if catalog.base_tabular_contract != tabular.contract:
        raise ValueError("feature catalog and base tabular contracts do not match")
    if catalog.split_contract != split.contract:
        raise ValueError("feature catalog and split contracts do not match")
    if catalog.development_contract != development.contract:
        raise ValueError("feature catalog and development contracts do not match")
    if catalog.dataset_role != "train" or catalog.official_test_allowed:
        raise PermissionError("tabular v2 catalog must remain train-only")
    if catalog.primary_risk_target != "future_absolute_excess_return":
        raise ValueError("tabular v2 primary risk target must be absolute excess return")
    if set(catalog.primary_risk_horizons) != {1, 3, 5}:
        raise ValueError("tabular v2 primary risk must cover 1/3/5 sessions")
    if catalog.primary_risk_output != "predicted_absolute_excess_return":
        raise ValueError("unexpected tabular v2 primary risk output")
    if catalog.unified_output_schema != "tabular-component-oof-v2":
        raise ValueError("unexpected tabular v2 output schema")
    required_output = {
        "sample_id",
        "horizon_sessions",
        "calibrated_bullish_probability",
        "predicted_excess_return",
        "predicted_absolute_excess_return",
        "component_available",
        "refusal_reason",
        "model_version",
        "feature_schema_sha256",
        "data_as_of",
        "evidence_sha256",
    }
    if not required_output.issubset(catalog.unified_output_fields):
        raise ValueError("tabular v2 output schema is incomplete")
    if not catalog.features:
        raise ValueError("tabular v2 feature catalog is empty")
    names = catalog.feature_names
    if len(names) != len(set(names)):
        raise ValueError("tabular v2 feature catalog contains duplicates")
    for item in catalog.features:
        if not tabular.feature_name_allowed(item.name):
            raise ValueError(f"forbidden tabular v2 feature: {item.name}")
        if not item.available_at_column or not item.unit or not item.transformation:
            raise ValueError(f"incomplete tabular v2 feature metadata: {item.name}")
        if item.source in {
            "market_daily",
            "market_intraday_aggregate",
            "tushare_daily_basic",
        } and not item.observation_date_column:
            raise ValueError(f"lagged feature lacks observation date: {item.name}")

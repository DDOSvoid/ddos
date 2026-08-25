"""Train-only v3 catalog extending the immutable v2 feature catalog."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT
from src.prediction.development_contract import CausalDevelopmentContract
from src.prediction.splits import PredictionSplitContract
from src.prediction.tabular.catalog import (
    CatalogFeature,
    TabularFeatureCatalog,
    load_tabular_feature_catalog,
)
from src.prediction.tabular.contract import TabularModelContract

V3_CATALOG_CONTRACT = "causal-tabular-feature-catalog-v3"
V3_DATASET_CONTRACT = "causal-tabular-train-dataset-v3"
DEFAULT_V3_CATALOG_PATH = PROJECT_ROOT / "config" / "tabular_feature_catalog_v3.yaml"


@dataclass(frozen=True)
class TabularFeatureCatalogV3:
    base: TabularFeatureCatalog
    features: tuple[CatalogFeature, ...]
    source_path: Path
    source_sha256: str

    @property
    def contract(self) -> str:
        return V3_CATALOG_CONTRACT

    @property
    def dataset_contract(self) -> str:
        return V3_DATASET_CONTRACT

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (*self.base.feature_names, *(item.name for item in self.features))

    @property
    def all_features(self) -> tuple[CatalogFeature, ...]:
        return (*self.base.features, *self.features)


def load_tabular_feature_catalog_v3(
    path: Path = DEFAULT_V3_CATALOG_PATH,
    *,
    tabular: TabularModelContract,
    split: PredictionSplitContract,
    development: CausalDevelopmentContract,
) -> TabularFeatureCatalogV3:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    if raw.get("contract") != V3_CATALOG_CONTRACT:
        raise ValueError("unexpected tabular v3 catalog contract")
    if raw.get("dataset_role") != "train" or raw.get("official_test_allowed") is not False:
        raise PermissionError("tabular v3 catalog must remain train-only")
    base_path = (PROJECT_ROOT / raw["base_catalog"]).resolve()
    base = load_tabular_feature_catalog(
        base_path, tabular=tabular, split=split, development=development
    )
    features: list[CatalogFeature] = []
    for group_name, group in raw["new_feature_groups"].items():
        if group_name == "point_in_time_financials":
            for endpoint, endpoint_group in group["endpoints"].items():
                for name, item in endpoint_group["features"].items():
                    features.append(
                        CatalogFeature(
                            name=str(name),
                            group=f"{group_name}_{endpoint}",
                            source=str(group["source"]),
                            available_at_column=str(endpoint_group["available_at_column"]),
                            observation_date_column=None,
                            unit=str(item["unit"]),
                            transformation=str(item["transformation"]),
                            missing_policy=str(group["missing_policy"]),
                        )
                    )
        else:
            for name, item in group["features"].items():
                features.append(
                    CatalogFeature(
                        name=str(name),
                        group=str(group_name),
                        source=str(group["source"]),
                        available_at_column=str(group["available_at_column"]),
                        observation_date_column=None,
                        unit=str(item["unit"]),
                        transformation=str(item["transformation"]),
                        missing_policy=str(group["missing_policy"]),
                    )
                )
    names = [*base.feature_names, *(item.name for item in features)]
    if len(names) != len(set(names)):
        raise ValueError("tabular v3 catalog contains duplicate feature names")
    for item in features:
        if not tabular.feature_name_allowed(item.name):
            raise ValueError(f"forbidden tabular v3 feature: {item.name}")
    return TabularFeatureCatalogV3(
        base=base,
        features=tuple(features),
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )

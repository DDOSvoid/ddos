from __future__ import annotations

from src.prediction.development_contract import load_causal_development_contract
from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.catalog_v3 import (
    V3_CATALOG_CONTRACT,
    V3_DATASET_CONTRACT,
    load_tabular_feature_catalog_v3,
)
from src.prediction.tabular.contract import load_tabular_model_contract


def test_v3_catalog_extends_v2_with_audited_groups() -> None:
    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
    tabular = load_tabular_model_contract(split=split, development=development)
    catalog = load_tabular_feature_catalog_v3(
        tabular=tabular, split=split, development=development
    )
    assert catalog.contract == V3_CATALOG_CONTRACT
    assert catalog.dataset_contract == V3_DATASET_CONTRACT
    assert len(catalog.base.feature_names) == 78
    assert len(catalog.feature_names) == 88
    assert "industry_l1_code_id" in catalog.feature_names
    assert "pit_cashflow_free_cashflow" in catalog.feature_names

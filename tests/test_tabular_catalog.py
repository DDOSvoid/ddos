"""Versioned tabular-v2 feature catalog tests."""

from src.prediction.development_contract import load_causal_development_contract
from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.catalog import load_tabular_feature_catalog
from src.prediction.tabular.contract import load_tabular_model_contract


def _catalog():
    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
    tabular = load_tabular_model_contract(split=split, development=development)
    return load_tabular_feature_catalog(
        tabular=tabular, split=split, development=development
    )


def test_v2_catalog_freezes_primary_risk_and_unified_output():
    catalog = _catalog()
    assert catalog.dataset_role == "train"
    assert catalog.official_test_allowed is False
    assert catalog.primary_risk_target == "future_absolute_excess_return"
    assert set(catalog.primary_risk_horizons) == {1, 3, 5}
    assert "calibrated_bullish_probability" in catalog.unified_output_fields
    assert "predicted_absolute_excess_return" in catalog.unified_output_fields


def test_v2_catalog_registers_every_feature_with_provenance():
    catalog = _catalog()
    assert len(catalog.features) == 78
    assert len(set(catalog.feature_names)) == 78
    daily_basic = catalog.feature_groups["valuation_liquidity"]
    assert len(daily_basic) == 15
    for feature in catalog.features:
        assert feature.source
        assert feature.available_at_column
        assert feature.unit
        assert feature.transformation
        assert feature.missing_policy

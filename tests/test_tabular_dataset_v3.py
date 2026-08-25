from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_train_v3_is_causal_and_explicit_about_industry_state() -> None:
    root = Path("data/tabular/train_v3")
    features = pd.read_parquet(root / "features.parquet")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract"] == "causal-tabular-train-dataset-v3"
    assert manifest["official_test_queried"] is False
    assert len(features) == 50_802
    assert int(features["industry_membership_ambiguous"].sum()) == 0
    assert int(
        (
            (features["industry_l1_code_id"] == 0)
            & ~features["industry_taxonomy_not_yet_effective"]
        ).sum()
    ) == 0
    published = pd.to_datetime(features["published_date"])
    for endpoint in ("pit_income", "pit_balancesheet", "pit_cashflow"):
        observed = pd.to_datetime(
            features[f"{endpoint}_observation_date"], errors="coerce"
        )
        assert int((observed.notna() & ~(observed < published)).sum()) == 0

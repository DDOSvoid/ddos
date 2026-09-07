from datetime import UTC, date, datetime

import pandas as pd

from src.prediction.fusion.baselines import run_equal_weight_baseline
from src.prediction.fusion.contract import load_expert_fusion_contract
from src.prediction.fusion.schema import build_joint_expert_frame


def _component(name: str, expected: float, risk: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["sample-1"],
            "company_day_id": ["company-day-1"],
            "stock_code": ["000001.SZ"],
            "published_date": [date(2024, 1, 2)],
            "prediction_as_of": [datetime(2024, 1, 3, 0, 30, tzinfo=UTC)],
            "eligible_entry_date": [date(2024, 1, 3)],
            "horizon_sessions": [1],
            "expert_signal": [0.3],
            "quality_score": [0.9],
            "available": [True],
            "abstain_reason": [None],
            "data_as_of": [datetime(2024, 1, 2, 10, tzinfo=UTC)],
            "evidence_sha256": [name * 64],
            "expert_version": [f"{name}-v1"],
            "bullish_probability": [0.6],
            "expected_excess_return": [expected],
            "risk_scale": [risk],
        }
    )


def test_equal_weight_baseline_averages_only_available_diagnostic_heads():
    contract = load_expert_fusion_contract()
    joint = build_joint_expert_frame(
        {
            "text": _component("t", 0.02, 0.01),
            "tabular": _component("x", 0.04, 0.03),
        },
        contract=contract,
    )
    output = run_equal_weight_baseline(joint, contract=contract)
    assert output.loc[0, "available_expert_count"] == 2
    assert output.loc[0, "expected_excess_return"] == 0.03
    assert output.loc[0, "risk_scale"] == 0.02
    assert output.loc[0, "daily_rank"] == 1

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from src.prediction.fusion.contract import load_expert_fusion_contract
from src.prediction.fusion.schema import (
    build_joint_expert_frame,
    validate_development_labels,
)


def _frame(expert: str, sample_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "company_day_id": [f"company-{value}" for value in sample_ids],
            "stock_code": [f"00000{index + 1}.SZ" for index in range(len(sample_ids))],
            "published_date": [date(2024, 1, index + 2) for index in range(len(sample_ids))],
            "prediction_as_of": [
                datetime(2024, 1, index + 3, 0, 30, tzinfo=UTC) for index in range(len(sample_ids))
            ],
            "eligible_entry_date": [date(2024, 1, index + 3) for index in range(len(sample_ids))],
            "horizon_sessions": [1] * len(sample_ids),
            "expert_signal": [0.2 + index / 10 for index in range(len(sample_ids))],
            "quality_score": [0.9] * len(sample_ids),
            "available": [True] * len(sample_ids),
            "abstain_reason": [None] * len(sample_ids),
            "data_as_of": [
                datetime(2024, 1, index + 2, 8, tzinfo=UTC) for index in range(len(sample_ids))
            ],
            "evidence_sha256": [expert * 64] * len(sample_ids),
            "expert_version": [f"{expert}-v1"] * len(sample_ids),
        }
    )


def test_joint_frame_uses_union_and_explicit_missing_masks():
    contract = load_expert_fusion_contract()
    text = _frame("t", ["a", "b"])
    tabular = _frame("x", ["a"])
    joined = build_joint_expert_frame({"text": text, "tabular": tabular}, contract=contract)
    assert len(joined) == 2
    missing_row = joined.loc[joined["sample_id"] == "b"].iloc[0]
    assert bool(missing_row["text__available"]) is True
    assert bool(missing_row["tabular__available"]) is False
    assert missing_row["tabular__missing_mask"] == 1
    assert missing_row["tabular__expert_signal"] == 0.0
    assert missing_row["available_expert_count"] == 1
    assert bool(missing_row["fusion_available"]) is True


def test_joint_frame_rejects_future_expert_data():
    contract = load_expert_fusion_contract()
    frame = _frame("t", ["a"])
    frame.loc[0, "data_as_of"] = datetime(2024, 2, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="after prediction_as_of"):
        build_joint_expert_frame({"text": frame}, contract=contract)


def test_joint_frame_rejects_2026_component_oof():
    contract = load_expert_fusion_contract()
    frame = _frame("t", ["a"])
    frame.loc[0, "published_date"] = date(2026, 1, 2)
    frame.loc[0, "prediction_as_of"] = datetime(2026, 1, 3, 0, 30, tzinfo=UTC)
    frame.loc[0, "eligible_entry_date"] = date(2026, 1, 3)
    frame.loc[0, "data_as_of"] = datetime(2026, 1, 2, 8, tzinfo=UTC)
    with pytest.raises(PermissionError, match="development cutoff"):
        build_joint_expert_frame({"text": frame}, contract=contract)


def test_development_labels_reject_sealed_roles_before_reading_metrics():
    contract = load_expert_fusion_contract()
    labels = pd.DataFrame(
        {
            "sample_id": ["sample-2026"],
            "published_date": [date(2026, 1, 2)],
            "dataset_role": ["forward_validation"],
            "future_excess_return": [0.99],
        }
    )
    with pytest.raises(PermissionError, match="sealed dataset role"):
        validate_development_labels(labels, contract=contract)

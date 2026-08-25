from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.prediction.development_contract import load_causal_development_contract
from src.prediction.timeseries.lstm_contract import load_lstm_timeseries_contract
from src.prediction.timeseries.sequence_baselines import (
    pool_direct_sequence_features,
    run_direct_sequence_baseline_oof,
)
from src.prediction.timeseries.sequence_identity import (
    canonical_company_day_id,
    canonical_sample_id,
)


def _synthetic_artifacts():
    periods = (
        date(2023, 2, 1),
        date(2023, 8, 1),
        date(2024, 2, 1),
        date(2024, 8, 1),
    )
    metadata_rows = []
    label_rows = []
    sequences = []
    for period_index, period in enumerate(periods):
        for item in range(4):
            published = period + timedelta(days=item)
            stock_code = f"{period_index * 4 + item + 1:06d}.SZ"
            company_day_id = canonical_company_day_id(stock_code, published)
            target = item % 2
            sequence = np.zeros((60, 10), dtype=np.float32)
            sequence[:, 0] = target + np.linspace(-0.1, 0.1, 60)
            sequence[:, 1:7] = period_index * 0.01
            sequence[:, 7:] = 1.0
            sequences.append(sequence)
            metadata_rows.append(
                {
                    "company_day_id": company_day_id,
                    "stock_code": stock_code,
                    "published_date": published,
                    "dataset_role": "train",
                    "prediction_as_of": datetime(
                        published.year,
                        published.month,
                        published.day,
                        0,
                        30,
                        tzinfo=UTC,
                    ),
                    "sequence_available": True,
                    "unavailable_reason": None,
                    "sequence_row": len(sequences) - 1,
                    "history_sessions": 60,
                }
            )
            for horizon in (1, 3, 5):
                label_rows.append(
                    {
                        "sample_id": canonical_sample_id(company_day_id, horizon),
                        "company_day_id": company_day_id,
                        "stock_code": stock_code,
                        "published_date": published,
                        "horizon_sessions": horizon,
                        "dataset_role": "train",
                        "outcome_available_at": datetime(
                            published.year,
                            published.month,
                            published.day,
                            12,
                            tzinfo=UTC,
                        )
                        + timedelta(days=horizon),
                        "future_excess_return": 0.02 if target else -0.01,
                        "actual_direction": 1 if target else -1,
                    }
                )
    values = np.stack(sequences)
    masks = np.ones(values.shape[:2], dtype=bool)
    return (
        values,
        masks,
        pd.DataFrame(metadata_rows),
        pd.DataFrame(label_rows),
    )


def test_pool_direct_sequence_features_uses_registered_within_sequence_statistics():
    values = np.zeros((1, 3, 2), dtype=np.float32)
    values[0, :, 0] = [1.0, 2.0, 3.0]
    values[0, :, 1] = [2.0, np.nan, 6.0]
    pooled = pool_direct_sequence_features(
        values,
        np.ones((1, 3), dtype=bool),
        feature_names=("a", "b"),
        statistics=("last", "mean", "std", "minimum", "maximum", "slope"),
    )

    assert pooled.loc[0, "a__last"] == pytest.approx(3.0)
    assert pooled.loc[0, "a__mean"] == pytest.approx(2.0)
    assert pooled.loc[0, "a__slope"] == pytest.approx(1.0)
    assert pooled.loc[0, "b__mean"] == pytest.approx(4.0)
    assert pooled.loc[0, "b__slope"] == pytest.approx(2.0)


def test_direct_sequence_baseline_uses_fixed_folds_and_identical_samples():
    sequences, masks, metadata, labels = _synthetic_artifacts()
    development = load_causal_development_contract()
    contract = load_lstm_timeseries_contract(development=development)
    result = run_direct_sequence_baseline_oof(
        sequences,
        masks,
        metadata,
        labels,
        contract=contract,
        development=development,
    )

    assert len(result.report["candidates"]) == 6
    assert set(result.oof_predictions["fold"]) == {
        "fold_1",
        "fold_2",
        "fold_3",
    }
    assert set(result.oof_predictions["lookback_sessions"]) == {20, 60}
    assert all(
        item["identical_across_lookbacks"]
        for item in result.report["sample_identity_audit"].values()
    )
    assert result.report["official_test_read"] is False
    assert result.report["model_selection"] == (
        "no_search_report_all_registered_lookbacks"
    )


def test_direct_sequence_baseline_rejects_sealed_role():
    sequences, masks, metadata, labels = _synthetic_artifacts()
    metadata.loc[0, "dataset_role"] = "test"
    development = load_causal_development_contract()
    with pytest.raises(PermissionError, match="train-only"):
        run_direct_sequence_baseline_oof(
            sequences,
            masks,
            metadata,
            labels,
            contract=load_lstm_timeseries_contract(development=development),
            development=development,
        )

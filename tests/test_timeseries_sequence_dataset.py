from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.prediction.timeseries.lstm_contract import load_lstm_timeseries_contract
from src.prediction.timeseries.sequence_dataset import (
    DirectSequenceDataset,
    assemble_direct_sequence_frames,
)
from src.prediction.timeseries.sequence_identity import (
    canonical_company_day_id,
    canonical_sample_id,
)
from src.prediction.timeseries.sequence_preprocessing import SequencePreprocessor


def _artifacts():
    day = date(2024, 4, 25)
    company_day_id = canonical_company_day_id("000001.SZ", day)
    values = np.arange(60 * 10, dtype=np.float32).reshape(1, 60, 10) / 100.0
    values[:, :, 7:] = 1.0
    masks = np.ones((1, 60), dtype=bool)
    metadata = pd.DataFrame(
        {
            "company_day_id": [company_day_id],
            "stock_code": ["000001.SZ"],
            "published_date": [day],
            "dataset_role": ["train"],
            "prediction_as_of": ["2024-04-26T08:30:00+08:00"],
            "sequence_available": [True],
            "unavailable_reason": [None],
            "sequence_row": [0],
            "history_sessions": [60],
        }
    )
    labels = pd.DataFrame(
        {
            "sample_id": [canonical_sample_id(company_day_id, 1)],
            "company_day_id": [company_day_id],
            "stock_code": ["000001.SZ"],
            "published_date": [day],
            "horizon_sessions": [1],
            "dataset_role": ["train"],
            "outcome_available_at": ["2024-04-26T15:00:00+00:00"],
            "future_excess_return": [0.02],
            "actual_direction": [1],
        }
    )
    return values, masks, metadata, labels


def test_sequence_and_label_artifacts_join_only_after_validation():
    values, masks, metadata, labels = _artifacts()
    contract = load_lstm_timeseries_contract()
    selected = values[:, -20:, :]
    selected_masks = masks[:, -20:]
    preprocessor = SequencePreprocessor.fit(selected, selected_masks)
    frames = assemble_direct_sequence_frames(
        values,
        masks,
        metadata,
        labels,
        horizon_sessions=1,
        lookback_sessions=20,
        contract=contract,
        preprocessor=preprocessor,
    )
    dataset = DirectSequenceDataset(frames)

    assert len(dataset) == 1
    item = dataset[0]
    assert item["sequence"].shape == (20, 10)
    assert item["time_mask"].all()
    assert item["target"].item() == 1.0
    assert item["excess_return"].item() == pytest.approx(0.02)


def test_preprocessor_fit_does_not_depend_on_validation_sequence():
    train = np.ones((2, 20, 10), dtype=np.float32)
    train[:, :, 0] = np.arange(40, dtype=np.float32).reshape(2, 20)
    train[:, :, 7:] = 1.0
    mask = np.ones((2, 20), dtype=bool)
    first = SequencePreprocessor.fit(train, mask)
    validation = train.copy()
    validation[:, :, 0] = 1_000_000.0
    second = SequencePreprocessor.fit(train, mask)

    np.testing.assert_array_equal(first.medians, second.medians)
    np.testing.assert_array_equal(first.means, second.means)
    np.testing.assert_array_equal(first.scales, second.scales)
    transformed = first.transform(validation, mask)
    assert transformed[:, :, 0].mean() > 10_000

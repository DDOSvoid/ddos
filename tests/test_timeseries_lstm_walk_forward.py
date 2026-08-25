from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import torch

from src.prediction.development_contract import load_causal_development_contract
from src.prediction.timeseries.lstm_contract import load_lstm_timeseries_contract
from src.prediction.timeseries.sequence_dataset import DirectSequenceFrames
from src.prediction.timeseries.train_lstm import (
    build_inner_temporal_split,
    predict_fold_lstm,
    train_fold_lstm,
)


def _frames() -> tuple[DirectSequenceFrames, pd.DataFrame]:
    rng = np.random.default_rng(7)
    sequences = []
    rows = []
    start = date(2023, 1, 2)
    for day_index in range(40):
        published = start + timedelta(days=day_index)
        for item in range(4):
            target = item % 2
            values = rng.normal(0.0, 0.02, size=(20, 10)).astype(np.float32)
            values[:, 0] += target * 0.05
            values[:, 7:] = 1.0
            sequences.append(values)
            rows.append(
                {
                    "sample_id": f"sample-{day_index}-{item}",
                    "published_date": published,
                    "outcome_available_at": datetime(
                        published.year,
                        published.month,
                        published.day,
                        12,
                        tzinfo=UTC,
                    )
                    + timedelta(days=1),
                    "dataset_role": "train",
                    "target": target,
                    "excess_return": 0.02 if target else -0.01,
                }
            )
    values = np.stack(sequences)
    metadata = pd.DataFrame(rows)
    indexed = metadata.copy()
    indexed["__row_index"] = np.arange(len(indexed))
    return (
        DirectSequenceFrames(
            sequences=values,
            time_masks=np.ones(values.shape[:2], dtype=bool),
            metadata=metadata,
        ),
        indexed,
    )


def test_inner_temporal_split_is_disjoint_chronological_and_matured():
    _, indexed = _frames()
    development = load_causal_development_contract()
    contract = load_lstm_timeseries_contract(development=development)
    split = build_inner_temporal_split(
        indexed,
        contract=contract,
        development=development,
    )
    partitions = [
        split.model_fit_indices,
        split.early_stop_indices,
        split.calibration_indices,
    ]

    assert not (set(partitions[0]) & set(partitions[1]))
    assert not (set(partitions[1]) & set(partitions[2]))
    assert indexed.iloc[partitions[0]]["published_date"].max() < indexed.iloc[
        partitions[1]
    ]["published_date"].min()
    assert indexed.iloc[partitions[1]]["published_date"].max() < indexed.iloc[
        partitions[2]
    ]["published_date"].min()
    for segment in split.audit["segments"].values():
        assert segment["mature_samples"] <= segment["samples_before_maturity_filter"]


def test_fold_training_is_deterministic_for_fixed_seed():
    frames, indexed = _frames()
    development = load_causal_development_contract()
    base = load_lstm_timeseries_contract(development=development)
    contract = replace(
        base,
        batch_size=32,
        minimum_epochs=2,
        maximum_epochs=4,
        early_stopping_patience=2,
    )
    torch.set_num_threads(1)
    split = build_inner_temporal_split(
        indexed,
        contract=contract,
        development=development,
    )
    first = train_fold_lstm(
        frames,
        split,
        hidden_size=16,
        seed=123,
        contract=contract,
    )
    mutated_sequences = frames.sequences.copy()
    mutated_sequences[split.calibration_indices, :, :7] = 1_000_000.0
    calibration_mutated_frames = DirectSequenceFrames(
        sequences=mutated_sequences,
        time_masks=frames.time_masks.copy(),
        metadata=frames.metadata.copy(),
    )
    second = train_fold_lstm(
        calibration_mutated_frames,
        split,
        hidden_size=16,
        seed=123,
        contract=contract,
    )
    first_prediction = predict_fold_lstm(
        first,
        frames,
        split.calibration_indices,
        batch_size=32,
    )
    second_prediction = predict_fold_lstm(
        second,
        frames,
        split.calibration_indices,
        batch_size=32,
    )

    for name in first_prediction:
        np.testing.assert_array_equal(first_prediction[name], second_prediction[name])
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name], rtol=0, atol=0)

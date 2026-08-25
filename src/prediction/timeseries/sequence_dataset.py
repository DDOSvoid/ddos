"""PyTorch dataset assembly for physically separated sequence and label artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.prediction.timeseries.lstm_contract import LstmTimeseriesContract
from src.prediction.timeseries.sequence_identity import canonical_sample_id
from src.prediction.timeseries.sequence_preprocessing import (
    SequencePreprocessor,
    select_tail_lookback,
)

LABEL_COLUMNS = frozenset(
    {
        "sample_id",
        "company_day_id",
        "stock_code",
        "published_date",
        "horizon_sessions",
        "dataset_role",
        "outcome_available_at",
        "future_excess_return",
        "actual_direction",
    }
)


@dataclass(frozen=True)
class DirectSequenceFrames:
    sequences: np.ndarray
    time_masks: np.ndarray
    metadata: pd.DataFrame


def assemble_direct_sequence_frames(
    sequences: np.ndarray,
    time_masks: np.ndarray,
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    horizon_sessions: int,
    lookback_sessions: int,
    contract: LstmTimeseriesContract,
    preprocessor: SequencePreprocessor | None = None,
) -> DirectSequenceFrames:
    missing_labels = LABEL_COLUMNS - set(labels.columns)
    if missing_labels:
        raise ValueError(f"LSTM labels missing columns: {sorted(missing_labels)}")
    forbidden_metadata = [
        name for name in metadata.columns if not contract.feature_name_allowed(name)
    ]
    if forbidden_metadata:
        raise ValueError(
            f"outcome fields rejected from sequence metadata: {forbidden_metadata}"
        )
    if set(metadata["dataset_role"].astype(str).unique()) != {"train"}:
        raise PermissionError("LSTM sequence metadata must be train-only")
    if set(labels["dataset_role"].astype(str).unique()) != {"train"}:
        raise PermissionError("LSTM labels must be train-only")
    if horizon_sessions not in contract.horizons_sessions:
        raise ValueError(f"unsupported LSTM horizon: {horizon_sessions}")
    if lookback_sessions not in contract.lookback_candidates_sessions:
        raise ValueError(f"unregistered LSTM lookback: {lookback_sessions}")
    if sequences.shape[:2] != time_masks.shape:
        raise ValueError("sequence and time-mask artifacts do not align")

    available = metadata.loc[metadata["sequence_available"].astype(bool)].copy()
    if available["sequence_row"].isna().any():
        raise ValueError("available sequence metadata lacks sequence_row")
    available["sequence_row"] = available["sequence_row"].astype(int)
    if available["company_day_id"].duplicated().any():
        raise ValueError("duplicate company_day_id in sequence metadata")
    if (available["sequence_row"] < 0).any() or (
        available["sequence_row"] >= len(sequences)
    ).any():
        raise ValueError("sequence_row points outside the sequence artifact")

    horizon_labels = labels.loc[
        labels["horizon_sessions"].astype(int) == horizon_sessions
    ].copy()
    if horizon_labels["company_day_id"].duplicated().any():
        raise ValueError("duplicate company-day label for the same horizon")
    expected_ids = [
        canonical_sample_id(value, horizon_sessions)
        for value in horizon_labels["company_day_id"].astype(str)
    ]
    if expected_ids != horizon_labels["sample_id"].astype(str).tolist():
        raise ValueError("label sample_id does not match the canonical algorithm")
    joined = available.merge(
        horizon_labels,
        on=["company_day_id", "stock_code", "published_date", "dataset_role"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_label"),
    )
    joined = joined.sort_values(
        ["published_date", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)
    source_rows = joined["sequence_row"].to_numpy(dtype=int)
    raw_sequences = np.asarray(sequences, dtype=np.float32)[source_rows]
    raw_masks = np.asarray(time_masks, dtype=bool)[source_rows]
    selected, selected_masks, lengths = select_tail_lookback(
        raw_sequences,
        raw_masks,
        lookback_sessions=lookback_sessions,
    )
    eligible = lengths >= lookback_sessions
    joined = joined.loc[eligible].reset_index(drop=True)
    selected = selected[eligible]
    selected_masks = selected_masks[eligible]
    if preprocessor is not None:
        selected = preprocessor.transform(selected, selected_masks)
    joined["target"] = (
        pd.to_numeric(joined["actual_direction"], errors="raise") > 0
    ).astype(int)
    joined["excess_return"] = pd.to_numeric(
        joined["future_excess_return"], errors="raise"
    )
    return DirectSequenceFrames(
        sequences=selected,
        time_masks=selected_masks,
        metadata=joined,
    )


class DirectSequenceDataset(Dataset):
    def __init__(self, frames: DirectSequenceFrames):
        if len(frames.sequences) != len(frames.metadata):
            raise ValueError("LSTM sequence and metadata row counts differ")
        self.sequences = torch.as_tensor(frames.sequences, dtype=torch.float32)
        self.time_masks = torch.as_tensor(frames.time_masks, dtype=torch.bool)
        self.targets = torch.as_tensor(
            frames.metadata["target"].to_numpy(dtype=np.float32)
        )
        self.excess_returns = torch.as_tensor(
            frames.metadata["excess_return"].to_numpy(dtype=np.float32)
        )
        self.sample_ids = frames.metadata["sample_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "sequence": self.sequences[index],
            "time_mask": self.time_masks[index],
            "target": self.targets[index],
            "excess_return": self.excess_returns[index],
            "sample_id": self.sample_ids[index],
        }

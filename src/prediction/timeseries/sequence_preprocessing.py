"""Fold-local preprocessing for padded direct market sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_arrays(
    sequences: np.ndarray,
    time_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(sequences, dtype=np.float32)
    masks = np.asarray(time_masks, dtype=bool)
    if values.ndim != 3:
        raise ValueError("sequences must have shape [samples, time, features]")
    if masks.shape != values.shape[:2]:
        raise ValueError("time masks must match sequence sample/time dimensions")
    if len(values) == 0:
        raise ValueError("sequence preprocessing requires at least one sample")
    if (~masks.any(axis=1)).any():
        raise ValueError("every sequence must contain at least one valid timestep")
    return values, masks


@dataclass(frozen=True)
class SequencePreprocessor:
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    continuous_feature_count: int

    @classmethod
    def fit(
        cls,
        sequences: np.ndarray,
        time_masks: np.ndarray,
        *,
        continuous_feature_count: int = 7,
    ) -> "SequencePreprocessor":
        values, masks = _validate_arrays(sequences, time_masks)
        feature_count = values.shape[2]
        if not 0 < continuous_feature_count <= feature_count:
            raise ValueError("invalid continuous feature count")
        medians = np.zeros(feature_count, dtype=np.float32)
        means = np.zeros(feature_count, dtype=np.float32)
        scales = np.ones(feature_count, dtype=np.float32)
        for feature_index in range(continuous_feature_count):
            observed = values[:, :, feature_index][masks]
            finite = observed[np.isfinite(observed)]
            if len(finite) == 0:
                raise ValueError(
                    f"fold-train sequence feature {feature_index} is entirely missing"
                )
            median = float(np.median(finite))
            imputed = np.where(np.isfinite(observed), observed, median)
            mean = float(imputed.mean())
            scale = float(imputed.std(ddof=0))
            medians[feature_index] = median
            means[feature_index] = mean
            scales[feature_index] = scale if scale > 1e-12 else 1.0
        return cls(
            medians=medians,
            means=means,
            scales=scales,
            continuous_feature_count=continuous_feature_count,
        )

    def transform(
        self,
        sequences: np.ndarray,
        time_masks: np.ndarray,
    ) -> np.ndarray:
        values, masks = _validate_arrays(sequences, time_masks)
        if values.shape[2] != len(self.medians):
            raise ValueError("sequence feature count differs from fitted preprocessor")
        transformed = values.copy()
        for feature_index in range(self.continuous_feature_count):
            column = transformed[:, :, feature_index]
            column = np.where(
                np.isfinite(column),
                column,
                self.medians[feature_index],
            )
            transformed[:, :, feature_index] = (
                column - self.means[feature_index]
            ) / self.scales[feature_index]
        if self.continuous_feature_count < transformed.shape[2]:
            binary = transformed[:, :, self.continuous_feature_count :]
            transformed[:, :, self.continuous_feature_count :] = np.where(
                np.isfinite(binary), binary, 0.0
            )
        transformed[~masks] = 0.0
        if not np.isfinite(transformed).all():
            raise ValueError("non-finite value remains after sequence preprocessing")
        return transformed.astype(np.float32, copy=False)


def select_tail_lookback(
    sequences: np.ndarray,
    time_masks: np.ndarray,
    *,
    lookback_sessions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, masks = _validate_arrays(sequences, time_masks)
    if lookback_sessions <= 0 or lookback_sessions > values.shape[1]:
        raise ValueError("lookback must fit within the stored maximum sequence")
    selected = np.zeros(
        (len(values), lookback_sessions, values.shape[2]), dtype=np.float32
    )
    selected_masks = np.zeros((len(values), lookback_sessions), dtype=bool)
    lengths = np.zeros(len(values), dtype=np.int64)
    for index in range(len(values)):
        observed = values[index, masks[index]]
        tail = observed[-lookback_sessions:]
        length = len(tail)
        selected[index, :length] = tail
        selected_masks[index, :length] = True
        lengths[index] = length
    return selected, selected_masks, lengths

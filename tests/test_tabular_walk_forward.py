"""Chronological tabular expanding-window tests."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from src.prediction.development_contract import load_causal_development_contract
from src.prediction.tabular.walk_forward import iter_expanding_window_frames


def _row(day: date, outcome_day: date, value: int) -> dict:
    return {
        "published_date": day,
        "outcome_available_at": datetime(
            outcome_day.year, outcome_day.month, outcome_day.day, 8, tzinfo=UTC
        ),
        "dataset_role": "train",
        "value": value,
    }


def test_expanding_folds_are_chronological_and_exclude_immature_labels():
    frame = pd.DataFrame(
        [
            _row(date(2024, 7, 2), date(2024, 7, 5), 5),
            _row(date(2023, 2, 1), date(2023, 2, 5), 1),
            _row(date(2023, 7, 3), date(2023, 7, 7), 3),
            _row(date(2023, 6, 29), date(2023, 7, 2), 2),
            _row(date(2024, 1, 2), date(2024, 1, 5), 4),
        ]
    )
    folds = list(
        iter_expanding_window_frames(
            frame,
            development=load_causal_development_contract(),
        )
    )
    assert [fold.name for fold in folds] == ["fold_1", "fold_2", "fold_3"]
    assert folds[0].train["value"].tolist() == [1]
    assert folds[0].validation["value"].tolist() == [3]
    # The June-29 outcome matured after fold_1's train cutoff, so it is
    # excluded there but becomes eligible in the later expanding window.
    assert folds[1].train["value"].tolist() == [1, 2, 3]
    assert folds[1].validation["value"].tolist() == [4]
    assert folds[2].train["value"].tolist() == [1, 2, 3, 4]
    assert folds[2].validation["value"].tolist() == [5]


def test_walk_forward_rejects_sealed_roles():
    frame = pd.DataFrame([_row(date(2023, 2, 1), date(2023, 2, 5), 1)])
    frame.loc[0, "dataset_role"] = "test"
    with pytest.raises(PermissionError, match="accepts train only"):
        list(
            iter_expanding_window_frames(
                frame,
                development=load_causal_development_contract(),
            )
        )

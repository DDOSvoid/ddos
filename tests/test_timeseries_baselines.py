"""Fold-local time-series baseline tests."""

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from src.prediction.development_contract import load_causal_development_contract
from src.prediction.timeseries.baselines import run_logistic_baseline_oof
from src.prediction.timeseries.contract import load_timeseries_model_contract


def _frame() -> pd.DataFrame:
    periods = (
        date(2023, 2, 1),
        date(2023, 8, 1),
        date(2024, 2, 1),
        date(2024, 8, 1),
    )
    rows = []
    for horizon in (1, 3, 5):
        for period_index, period in enumerate(periods):
            for item in range(4):
                published = period + timedelta(days=item)
                target = item % 2
                rows.append(
                    {
                        "sample_id": f"h{horizon}-p{period_index}-{item}",
                        "published_date": published,
                        "prediction_as_of": datetime(
                            published.year,
                            published.month,
                            published.day,
                            0,
                            30,
                            tzinfo=UTC,
                        ),
                        "last_bar_trade_date": published - timedelta(days=1),
                        "last_bar_available_at": datetime(
                            published.year,
                            published.month,
                            published.day,
                            0,
                            0,
                            tzinfo=UTC,
                        ),
                        "dataset_role": "train",
                        "outcome_available_at": datetime(
                            published.year,
                            published.month,
                            published.day,
                            12,
                            tzinfo=UTC,
                        )
                        + timedelta(days=horizon),
                        "horizon_sessions": horizon,
                        "target": target,
                        "stock_momentum_20d": float(target) + period_index * 0.01,
                    }
                )
    return pd.DataFrame(rows)


def test_logistic_baseline_fits_preprocessing_inside_each_fold():
    development = load_causal_development_contract()
    contract = load_timeseries_model_contract(development=development)
    result = run_logistic_baseline_oof(
        _frame(),
        feature_columns=["stock_momentum_20d"],
        contract=contract,
        development=development,
        c_values=(0.1, 1.0),
        class_weights=("none", "balanced"),
    )
    assert set(result.selected_c_by_horizon) == {1, 3, 5}
    assert set(result.selected_class_weight_by_horizon) == {1, 3, 5}
    assert set(result.oof_predictions["fold"]) == {"fold_1", "fold_2", "fold_3"}
    assert result.report["dataset_role"] == "train"
    assert "fit inside each fold" in result.report["preprocessing"]


def test_logistic_baseline_rejects_sealed_role():
    frame = _frame()
    frame.loc[0, "dataset_role"] = "test"
    development = load_causal_development_contract()
    with pytest.raises(PermissionError, match="accepts train only"):
        run_logistic_baseline_oof(
            frame,
            feature_columns=["stock_momentum_20d"],
            contract=load_timeseries_model_contract(development=development),
            development=development,
            c_values=(1.0,),
            class_weights=("none",),
        )

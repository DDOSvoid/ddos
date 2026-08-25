from __future__ import annotations

import numpy as np
import pandas as pd

from src.prediction.development_contract import load_causal_development_contract
from src.prediction.timeseries.evaluate_lstm import evaluate_lstm_candidate
from src.prediction.timeseries.lstm_contract import load_lstm_timeseries_contract


def _comparison_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    lstm_rows = []
    baseline_rows = []
    for fold_index, fold in enumerate(("fold_1", "fold_2", "fold_3")):
        for item in range(200):
            target = item % 2
            actual_return = 0.02 if target else -0.01
            common = {
                "sample_id": f"{fold}-{item}",
                "fold": fold,
                "lookback_sessions": 20,
                "horizon_sessions": 1,
                "target": target,
                "future_excess_return": actual_return,
            }
            probability = 0.9 if target else 0.1
            lstm_rows.append(
                {
                    **common,
                    "raw_bullish_probability": probability,
                    "bullish_probability": probability,
                    "expected_excess_return": actual_return,
                    "return_interval_lower": actual_return - 0.002,
                    "return_interval_upper": actual_return + 0.002,
                }
            )
            baseline_rows.append(
                {
                    **common,
                    "bullish_probability": 0.51,
                    "expected_excess_return": 0.005,
                    "return_interval_lower": -0.02,
                    "return_interval_upper": 0.03,
                    "constant_probability": 0.5,
                    "constant_expected_excess_return": 0.005,
                    "constant_interval_lower": -0.01,
                    "constant_interval_upper": 0.02,
                }
            )
    return pd.DataFrame(lstm_rows), pd.DataFrame(baseline_rows)


def test_lstm_evaluation_requires_absolute_and_simple_increment_gates():
    lstm, baseline = _comparison_frames()
    development = load_causal_development_contract()
    contract = load_lstm_timeseries_contract(development=development)
    result = evaluate_lstm_candidate(
        lstm,
        baseline,
        lookback_sessions=20,
        horizon_sessions=1,
        contract=contract,
        development=development,
    )

    assert result.report["absolute_gates"][
        "predictive_general_and_return_passed"
    ]
    assert result.report["simple_sequence_increment"]["passed"]
    assert result.report["train_gate_passed_before_table_increment"]
    assert result.report["table_increment_evaluated"] is False
    assert np.isfinite(
        result.report["pooled_metrics"]["cost_adjusted_mean_excess_return_ci_lower"]
    )

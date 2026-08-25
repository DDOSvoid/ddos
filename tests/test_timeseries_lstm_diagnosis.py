from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.prediction.timeseries.diagnose_lstm_v1 import diagnose_lstm_v1
from src.prediction.timeseries.sequence_artifacts import _file_sha256


def _write_fixture(tmp_path: Path, *, with_bad_role: bool = False) -> tuple[Path, Path]:
    oof_path = tmp_path / "oof.parquet"
    frame = pd.DataFrame(
        {
            "model_version": ["lstm_lb20_h16_d1"] * 4,
            "fold": ["fold_1", "fold_1", "fold_2", "fold_2"],
            "target": [0.0, 1.0, 0.0, 1.0],
            "raw_bullish_probability": [0.4, 0.6, 0.45, 0.55],
            "bullish_probability": [0.45, 0.55, 0.5, 0.5],
            "future_excess_return": [-0.01, 0.02, -0.02, 0.01],
            "raw_interval_lower": [-0.02] * 4,
            "raw_interval_upper": [0.02] * 4,
            "return_interval_lower": [-0.03] * 4,
            "return_interval_upper": [0.03] * 4,
        }
    )
    if with_bad_role:
        frame["dataset_role"] = ["train", "test", "train", "train"]
    frame.to_parquet(oof_path, index=False)
    report_path = tmp_path / "report.json"
    report = {
        "contract": "causal-timeseries-lstm-daily-v1",
        "contract_sha256": "contract",
        "official_test_read": False,
        "oof_sha256": _file_sha256(oof_path),
        "oof_rows": 4,
        "candidates": {
            "lstm_lb20_h16_d1": {
                "lookback_sessions": 20,
                "hidden_size": 16,
                "horizon_sessions": 1,
                "fold_training": [
                    {
                        "best_epoch": 5,
                        "temperature_calibration": {"temperature": 1.0},
                        "interval_calibration": {"correction": 0.0},
                    },
                    {
                        "best_epoch": 7,
                        "temperature_calibration": {"temperature": 1.1},
                        "interval_calibration": {"correction": 0.01},
                    },
                ],
                "evaluation": {
                    "probability_calibration": {
                        "calibrated_brier_score": 0.25,
                        "raw_brier_score": 0.24,
                    },
                    "pooled_metrics": {
                        "interval_mean_width": 0.06,
                        "constant_interval_mean_width": 0.04,
                    },
                    "absolute_gates": {
                        "predictive_general_and_return_passed": False,
                    },
                    "simple_sequence_increment": {"passed": True},
                },
            }
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, oof_path


def test_v1_diagnosis_accepts_legacy_oof_without_role_column(tmp_path: Path):
    report_path, oof_path = _write_fixture(tmp_path)

    result = diagnose_lstm_v1(report_path, oof_path)

    assert result["official_test_read"] is False
    assert result["candidate_count"] == 1
    assert result["aggregate_findings"][
        "temperature_calibration_outer_brier_worsened_candidates"
    ] == 1
    assert result["aggregate_findings"]["interval_width_wider_than_constant_candidates"] == 1
    assert result["candidates"]["lstm_lb20_h16_d1"]["v1_absolute_gate_passed"] is False


def test_v1_diagnosis_rejects_non_train_role(tmp_path: Path):
    report_path, oof_path = _write_fixture(tmp_path, with_bad_role=True)

    with pytest.raises(PermissionError, match="non-train OOF role"):
        diagnose_lstm_v1(report_path, oof_path)

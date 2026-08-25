"""Read-only diagnosis of LSTM v1 train OOF; never reads sealed partitions."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.prediction.timeseries.sequence_artifacts import _file_sha256


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _fold_diagnostics(frame: pd.DataFrame) -> dict[str, object]:
    actual = frame["target"].to_numpy(dtype=float)
    raw_probability = frame["raw_bullish_probability"].to_numpy(dtype=float)
    calibrated_probability = frame["bullish_probability"].to_numpy(dtype=float)
    actual_return = frame["future_excess_return"].to_numpy(dtype=float)
    raw_lower = frame["raw_interval_lower"].to_numpy(dtype=float)
    raw_upper = frame["raw_interval_upper"].to_numpy(dtype=float)
    lower = frame["return_interval_lower"].to_numpy(dtype=float)
    upper = frame["return_interval_upper"].to_numpy(dtype=float)
    return {
        "samples": int(len(frame)),
        "positive_rate": float(actual.mean()),
        "raw_probability_mean": float(raw_probability.mean()),
        "calibrated_probability_mean": float(calibrated_probability.mean()),
        "raw_probability_std": float(raw_probability.std(ddof=0)),
        "calibrated_probability_std": float(calibrated_probability.std(ddof=0)),
        "raw_brier": float(np.mean(np.square(raw_probability - actual))),
        "calibrated_brier": float(
            np.mean(np.square(calibrated_probability - actual))
        ),
        "raw_interval_coverage": float(
            np.mean((actual_return >= raw_lower) & (actual_return <= raw_upper))
        ),
        "calibrated_interval_coverage": float(
            np.mean((actual_return >= lower) & (actual_return <= upper))
        ),
        "raw_interval_width": float(np.mean(raw_upper - raw_lower)),
        "calibrated_interval_width": float(np.mean(upper - lower)),
        "interval_correction_mean": float(
            np.mean((lower - raw_lower + raw_upper - upper) / -2.0)
        ),
    }


def diagnose_lstm_v1(
    report_path: Path,
    oof_path: Path,
) -> dict[str, object]:
    report = _load_json(report_path)
    if report.get("official_test_read") is not False:
        raise PermissionError("v1 diagnosis cannot read official-test evidence")
    if _file_sha256(oof_path) != report["oof_sha256"]:
        raise ValueError("v1 OOF hash changed before diagnosis")
    frame = pd.read_parquet(oof_path)
    # v1 OOF artifacts predate the explicit role column.  If the column is
    # present, it must still be train-only; its absence is the expected form.
    if "dataset_role" in frame and set(frame["dataset_role"].astype(str).unique()) != {"train"}:
        raise PermissionError("v1 diagnosis found a non-train OOF role")
    candidates = {}
    for candidate_name, candidate in report["candidates"].items():
        subset = frame.loc[frame["model_version"] == candidate_name].copy()
        if subset.empty:
            raise ValueError(f"missing v1 OOF rows for {candidate_name}")
        folds = {
            str(fold): _fold_diagnostics(group)
            for fold, group in subset.groupby("fold", sort=True)
        }
        training = candidate["fold_training"]
        best_epochs = [int(item["best_epoch"]) for item in training]
        temperatures = [
            float(item["temperature_calibration"]["temperature"])
            for item in training
        ]
        corrections = [
            float(item["interval_calibration"]["correction"])
            for item in training
        ]
        candidates[candidate_name] = {
            "lookback_sessions": candidate["lookback_sessions"],
            "hidden_size": candidate["hidden_size"],
            "horizon_sessions": candidate["horizon_sessions"],
            "folds": folds,
            "best_epoch_min": min(best_epochs),
            "best_epoch_max": max(best_epochs),
            "best_epoch_mean": float(np.mean(best_epochs)),
            "temperature_min": min(temperatures),
            "temperature_max": max(temperatures),
            "temperature_mean": float(np.mean(temperatures)),
            "interval_correction_min": min(corrections),
            "interval_correction_max": max(corrections),
            "interval_correction_mean": float(np.mean(corrections)),
            "v1_absolute_gate_passed": candidate["evaluation"][
                "absolute_gates"
            ]["predictive_general_and_return_passed"],
            "v1_simple_increment_passed": candidate["evaluation"][
                "simple_sequence_increment"
            ]["passed"],
        }
    temperature_deltas = [
        value["evaluation"]["probability_calibration"]["calibrated_brier_score"]
        - value["evaluation"]["probability_calibration"]["raw_brier_score"]
        for value in report["candidates"].values()
    ]
    width_deltas = [
        value["evaluation"]["pooled_metrics"]["interval_mean_width"]
        - value["evaluation"]["pooled_metrics"]["constant_interval_mean_width"]
        for value in report["candidates"].values()
    ]
    return {
        "diagnosis": "causal-timeseries-lstm-daily-v1-train-oof",
        "created_at": datetime.now(UTC).isoformat(),
        "contract": report["contract"],
        "contract_sha256": report["contract_sha256"],
        "dataset_role_read": "train",
        "official_test_read": False,
        "quarantine_read": False,
        "forward_validation_read": False,
        "v1_oof_sha256": report["oof_sha256"],
        "v1_oof_rows": report["oof_rows"],
        "candidate_count": len(candidates),
        "aggregate_findings": {
            "temperature_calibration_outer_brier_worsened_candidates": int(
                sum(value > 0 for value in temperature_deltas)
            ),
            "temperature_calibration_outer_brier_improved_candidates": int(
                sum(value < 0 for value in temperature_deltas)
            ),
            "interval_width_wider_than_constant_candidates": int(
                sum(value > 0 for value in width_deltas)
            ),
            "interval_width_delta_min": float(min(width_deltas)),
            "interval_width_delta_max": float(max(width_deltas)),
            "interval_correction_zero_candidates": int(
                sum(
                    value["interval_correction_max"] == 0.0
                    for value in candidates.values()
                )
            ),
        },
        "registered_v2_hypotheses": {
            "target_scale": {
                "hypothesis": "fold-train target centering/scaling will reduce multi-task loss imbalance",
                "change": "fit target mean and scale on inner model-fit only; invert expected return and interval outputs before OOF scoring",
                "complexity_change": "none; same one-layer LSTM family",
            },
            "shrinkable_interval": {
                "hypothesis": "a calibration rule that can shrink the raw interval can improve width without using outer validation",
                "change": "fit symmetric split-conformal residual interval around expected return on inner calibration only",
                "complexity_change": "none; calibration-only change",
            },
            "brier_temperature": {
                "hypothesis": "temperature optimized for calibration Brier rather than NLL is more aligned with the registered gate",
                "change": "optimize bounded temperature on inner calibration Brier, retain identity fallback",
                "complexity_change": "none; calibration objective only",
            },
        },
        "candidates": candidates,
        "decision": "V1_FAIL_REGISTER_V2_RESEARCH_ONLY",
    }


def _write_json_atomic(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "timeseries"
        / "lstm_daily_v1"
        / "reports"
        / "train_lstm_candidates.json",
    )
    parser.add_argument(
        "--oof",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "timeseries"
        / "lstm_daily_v1"
        / "oof"
        / "train_lstm_candidates.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "timeseries"
        / "lstm_daily_v1"
        / "reports"
        / "v1_diagnostic_report.json",
    )
    args = parser.parse_args()
    result = diagnose_lstm_v1(args.report, args.oof)
    _write_json_atomic(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

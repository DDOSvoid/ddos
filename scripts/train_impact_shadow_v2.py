#!/usr/bin/env python
"""Train on 2023-2024 only, then perform one sealed 2025 test read."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.database.engine import get_engine, init_db
from src.prediction.datasets import load_company_day_samples
from src.prediction.memory import wilson_interval
from src.prediction.splits import load_prediction_split_contract

CONTRACT = "impact-shadow-temporal-v2"
CV_FOLDS = (
    (date(2023, 6, 30), date(2023, 7, 1), date(2023, 12, 31)),
    (date(2023, 12, 31), date(2024, 1, 1), date(2024, 6, 30)),
    (date(2024, 6, 30), date(2024, 7, 1), date(2024, 12, 31)),
)
C_VALUES = (0.1, 0.3, 1.0, 3.0)
HORIZONS = (1, 3, 5)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset_sha256(samples: list[dict]) -> str:
    evidence = [
        {
            "company_id": item["company_id"],
            "published_date": item["published_date"].isoformat(),
            "horizon_sessions": item["horizon_sessions"],
            "announcement_ids": sorted(item["announcement_ids"]),
            "text": item["text"],
            "target": item["target"],
        }
        for item in sorted(
            samples,
            key=lambda row: (
                row["horizon_sessions"],
                row["published_date"],
                row["company_id"],
            ),
        )
    ]
    return _json_sha256(evidence)


def _pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=3,
                    max_features=120_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="liblinear",
                    random_state=0,
                ),
            ),
        ]
    )


def _metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = (probability >= 0.5).astype(int)
    bullish = probability >= threshold
    bullish_count = int(bullish.sum())
    bullish_hits = int(target[bullish].sum()) if bullish_count else 0
    lower, upper = wilson_interval(bullish_hits, bullish_count)
    auc = None if len(np.unique(target)) < 2 else float(roc_auc_score(target, probability))
    return {
        "samples": int(len(target)),
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "roc_auc": auc,
        "bullish_threshold": threshold,
        "bullish_signals": bullish_count,
        "bullish_coverage": bullish_count / len(target) if len(target) else 0.0,
        "bullish_hits": bullish_hits,
        "bullish_hit_rate": bullish_hits / bullish_count if bullish_count else None,
        "bullish_hit_rate_wilson_95pct": [lower, upper],
    }


def _select_threshold(target: np.ndarray, probability: np.ndarray) -> tuple[float, dict]:
    candidates = []
    minimum_signals = max(100, int(len(target) * 0.05))
    for threshold in np.arange(0.50, 0.86, 0.02):
        value = float(round(threshold, 2))
        metrics = _metrics(target, probability, value)
        if metrics["bullish_signals"] >= minimum_signals:
            candidates.append(metrics)
    if not candidates:
        return 0.5, _metrics(target, probability, 0.5)
    selected = max(
        candidates,
        key=lambda item: (
            item["bullish_hit_rate_wilson_95pct"][0],
            item["bullish_signals"],
        ),
    )
    return selected["bullish_threshold"], selected


def _fold_predictions(samples: list[dict], c_value: float) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    all_targets = []
    all_probabilities = []
    folds = []
    for train_end, validation_start, validation_end in CV_FOLDS:
        fit = [item for item in samples if item["published_date"] <= train_end]
        validation = [
            item
            for item in samples
            if validation_start <= item["published_date"] <= validation_end
        ]
        if not fit or not validation:
            raise ValueError(f"empty internal CV fold ending {validation_end}")
        model = _pipeline(c_value)
        model.fit(
            [item["text"] for item in fit],
            np.asarray([item["target"] for item in fit]),
        )
        target = np.asarray([item["target"] for item in validation])
        probability = model.predict_proba([item["text"] for item in validation])[:, 1]
        all_targets.extend(target.tolist())
        all_probabilities.extend(probability.tolist())
        folds.append(
            {
                "train_end": train_end.isoformat(),
                "validation": [validation_start.isoformat(), validation_end.isoformat()],
                "train_samples": len(fit),
                "validation_samples": len(validation),
                "metrics_at_0_5": _metrics(target, probability, 0.5),
            }
        )
    return np.asarray(all_targets), np.asarray(all_probabilities), folds


def train_and_lock(output_dir: Path) -> dict:
    init_db()
    split = load_prediction_split_contract()
    lock_path = output_dir / "LOCK.json"
    if lock_path.exists():
        raise FileExistsError(f"locked v2 experiment already exists: {lock_path}")
    with Session(get_engine()) as session:
        samples = load_company_day_samples(session, role="train", split=split)
    if not samples:
        raise ValueError("no eligible train samples")

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "contract": CONTRACT,
        "status": "locked_pending_official_test",
        "created_at": datetime.now(UTC).isoformat(),
        "split_contract": split.contract,
        "split_source_sha256": split.source_sha256,
        "train_role": "train",
        "test_role": "test",
        "quarantine_used": False,
        "forward_validation_used": False,
        "unit": "company-publication-day",
        "features": "accepted causal title classification plus announcement titles",
        "target": "positive stock excess return versus CSI300",
        "candidate_c_values": list(C_VALUES),
        "selection_rule": "highest pooled expanding-CV bullish Wilson lower bound",
        "threshold_rule": "0.50..0.84 by 0.02; at least max(100, 5% of OOF samples)",
        "official_test_gate": {
            "minimum_bullish_signals": 100,
            "bullish_hit_rate_wilson_lower_strictly_above": 0.5,
            "eligibility": "each horizon passes or fails independently",
        },
        "train_dataset_sha256": _dataset_sha256(samples),
        "horizons": {},
    }

    for horizon in HORIZONS:
        horizon_samples = [
            item for item in samples if item["horizon_sessions"] == horizon
        ]
        candidates = []
        for c_value in C_VALUES:
            target, probability, folds = _fold_predictions(horizon_samples, c_value)
            threshold, pooled_metrics = _select_threshold(target, probability)
            candidates.append(
                {
                    "c": c_value,
                    "threshold": threshold,
                    "pooled_metrics": pooled_metrics,
                    "folds": folds,
                }
            )
        selected = max(
            candidates,
            key=lambda item: (
                item["pooled_metrics"]["bullish_hit_rate_wilson_95pct"][0],
                item["pooled_metrics"]["bullish_signals"],
            ),
        )
        model = _pipeline(selected["c"])
        model.fit(
            [item["text"] for item in horizon_samples],
            np.asarray([item["target"] for item in horizon_samples]),
        )
        model_path = output_dir / f"horizon_{horizon}.joblib"
        joblib.dump(model, model_path)
        report["horizons"][str(horizon)] = {
            "train_samples": len(horizon_samples),
            "selected_c": selected["c"],
            "selected_threshold": selected["threshold"],
            "internal_cv": selected["pooled_metrics"],
            "all_candidates": candidates,
            "model_file": model_path.name,
            "model_sha256": _file_sha256(model_path),
        }
        print(
            f"horizon={horizon} train={len(horizon_samples)} "
            f"cv_bullish_hit={selected['pooled_metrics']['bullish_hit_rate']:.4f} "
            f"cv_wilson_lower={selected['pooled_metrics']['bullish_hit_rate_wilson_95pct'][0]:.4f}",
            flush=True,
        )

    manifest_path = output_dir / "training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    lock = {
        "contract": CONTRACT,
        "locked_at": datetime.now(UTC).isoformat(),
        "training_manifest_file": manifest_path.name,
        "training_manifest_sha256": _file_sha256(manifest_path),
        "test_read_limit": 1,
        "test_read_count": 0,
    }
    with open(lock_path, "x", encoding="utf-8") as file:
        json.dump(lock, file, ensure_ascii=False, indent=2)
    print(json.dumps(lock, ensure_ascii=False, indent=2))
    return report


def evaluate_official_test(output_dir: Path, *, confirmed: bool) -> dict:
    if not confirmed:
        raise PermissionError("pass --confirm-one-time-test to consume the sealed test")
    split = load_prediction_split_contract()
    lock_path = output_dir / "LOCK.json"
    started_path = output_dir / "OFFICIAL_TEST_STARTED.json"
    result_path = output_dir / "OFFICIAL_TEST_RESULT.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest_path = output_dir / lock["training_manifest_file"]
    if _file_sha256(manifest_path) != lock["training_manifest_sha256"]:
        raise ValueError("training manifest changed after lock")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for horizon, details in manifest["horizons"].items():
        model_path = output_dir / details["model_file"]
        if _file_sha256(model_path) != details["model_sha256"]:
            raise ValueError(f"model changed after lock: horizon {horizon}")
    if started_path.exists() or result_path.exists():
        raise RuntimeError("official test has already been consumed; second read refused")

    started = {
        "contract": CONTRACT,
        "started_at": datetime.now(UTC).isoformat(),
        "training_manifest_sha256": lock["training_manifest_sha256"],
        "split_source_sha256": split.source_sha256,
    }
    with open(started_path, "x", encoding="utf-8") as file:
        json.dump(started, file, ensure_ascii=False, indent=2)

    init_db()
    with Session(get_engine()) as session:
        samples = load_company_day_samples(
            session,
            role="test",
            split=split,
            allow_sealed=True,
        )
    result = {
        "contract": CONTRACT,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "training_manifest_sha256": lock["training_manifest_sha256"],
        "test_dataset_sha256": _dataset_sha256(samples),
        "test_period": [split.test.start.isoformat(), split.test.end.isoformat()],
        "test_read_count": 1,
        "quarantine_used": False,
        "forward_validation_used": False,
        "horizons": {},
    }
    for horizon in HORIZONS:
        details = manifest["horizons"][str(horizon)]
        horizon_samples = [
            item for item in samples if item["horizon_sessions"] == horizon
        ]
        model = joblib.load(output_dir / details["model_file"])
        target = np.asarray([item["target"] for item in horizon_samples])
        probability = model.predict_proba(
            [item["text"] for item in horizon_samples]
        )[:, 1]
        metrics = _metrics(target, probability, details["selected_threshold"])
        gate = (
            metrics["bullish_signals"] >= 100
            and metrics["bullish_hit_rate_wilson_95pct"][0] > 0.5
        )
        result["horizons"][str(horizon)] = {
            **metrics,
            "production_gate_pass": gate,
        }
    result["production_eligible_horizons"] = [
        int(horizon)
        for horizon, details in result["horizons"].items()
        if details["production_gate_pass"]
    ]
    with open(result_path, "x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "evaluate"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "impact-shadow-v2",
    )
    parser.add_argument("--confirm-one-time-test", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.action == "train":
        train_and_lock(output_dir)
    else:
        evaluate_official_test(
            output_dir,
            confirmed=args.confirm_one_time_test,
        )


if __name__ == "__main__":
    main()

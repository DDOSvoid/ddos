#!/usr/bin/env python
"""Train a local, strictly temporal shadow model for bullish excess returns."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.database.engine import get_engine
from src.database.models import (
    Announcement,
    AnnouncementMarketTarget,
    Classification,
    Company,
)
from src.prediction.memory import wilson_interval

SHANGHAI = ZoneInfo("Asia/Shanghai")
CONTRACT = "impact-shadow-temporal-v1"


def _cutoff(day: date) -> datetime:
    return datetime.combine(day, time(23, 59, 59), tzinfo=SHANGHAI).astimezone(UTC)


def _aware_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_company_day_samples() -> list[dict]:
    engine = get_engine()
    with Session(engine) as session:
        rows = (
            session.query(
                Announcement.id,
                Announcement.company_id,
                Announcement.published_date,
                Announcement.title,
                Company.industry,
                Classification.sub_category,
                Classification.needs_review,
                AnnouncementMarketTarget.horizon_sessions,
                AnnouncementMarketTarget.actual_direction,
                AnnouncementMarketTarget.outcome_available_at,
            )
            .join(Company, Company.id == Announcement.company_id)
            .join(Classification, Classification.announcement_id == Announcement.id)
            .join(
                AnnouncementMarketTarget,
                AnnouncementMarketTarget.announcement_id == Announcement.id,
            )
            .order_by(
                AnnouncementMarketTarget.horizon_sessions,
                Announcement.published_date,
                Announcement.company_id,
                Announcement.id,
            )
            .all()
        )

    groups = defaultdict(
        lambda: {
            "titles": [],
            "sub_categories": set(),
            "has_accepted_classification": False,
        }
    )
    for row in rows:
        key = (row.company_id, row.published_date, row.horizon_sessions)
        group = groups[key]
        group["company_id"] = row.company_id
        group["published_date"] = row.published_date
        group["horizon_sessions"] = row.horizon_sessions
        group["industry"] = row.industry or "未知行业"
        group["titles"].append((row.id, row.title))
        group["sub_categories"].add(row.sub_category)
        group["has_accepted_classification"] |= not bool(row.needs_review)
        direction = 1 if row.actual_direction > 0 else 0
        if "target" in group and group["target"] != direction:
            raise ValueError(f"inconsistent company-day target: {key}")
        group["target"] = direction
        available_at = _aware_utc(row.outcome_available_at)
        group["outcome_available_at"] = max(
            available_at,
            group.get("outcome_available_at", available_at),
        )

    samples = []
    for group in groups.values():
        if not group["has_accepted_classification"]:
            continue
        titles = [title for _, title in sorted(set(group["titles"]))]
        categories = " ".join(
            f"类别{category}" for category in sorted(group["sub_categories"])
        )
        samples.append(
            {
                **group,
                "text": f"行业{group['industry']} {categories} 公告 " + "；".join(titles),
            }
        )
    return samples


def _metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = (probability >= 0.5).astype(int)
    bullish = probability >= threshold
    bullish_count = int(bullish.sum())
    bullish_hits = int(target[bullish].sum()) if bullish_count else 0
    lower, upper = wilson_interval(bullish_hits, bullish_count)
    return {
        "samples": int(len(target)),
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "bullish_threshold": threshold,
        "bullish_signals": bullish_count,
        "bullish_coverage": bullish_count / len(target) if len(target) else 0.0,
        "bullish_hit_rate": bullish_hits / bullish_count if bullish_count else None,
        "bullish_hit_rate_wilson_95pct": [lower, upper],
    }


def _select_threshold(target: np.ndarray, probability: np.ndarray) -> tuple[float, dict]:
    candidates = []
    minimum_signals = max(100, int(len(target) * 0.05))
    for threshold in np.arange(0.50, 0.86, 0.02):
        metrics = _metrics(target, probability, float(round(threshold, 2)))
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


def _pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=3,
                    max_features=80_000,
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


def train(*, output_dir: Path) -> dict:
    samples = load_company_day_samples()
    train_end = date(2026, 4, 30)
    validation_start = date(2026, 5, 1)
    validation_end = date(2026, 6, 30)
    test_start = date(2026, 7, 1)
    report = {
        "contract": CONTRACT,
        "created_at": datetime.now(UTC).isoformat(),
        "unit": "company-publication-day",
        "target": "positive stock excess return versus CSI300",
        "split": {
            "train_published_through": train_end.isoformat(),
            "train_outcomes_available_through": train_end.isoformat(),
            "validation": [validation_start.isoformat(), validation_end.isoformat()],
            "validation_outcomes_available_through": validation_end.isoformat(),
            "test_published_from": test_start.isoformat(),
        },
        "production_eligible": False,
        "horizons": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    for horizon in (1, 3, 5):
        horizon_samples = [item for item in samples if item["horizon_sessions"] == horizon]
        train_samples = [
            item
            for item in horizon_samples
            if item["published_date"] <= train_end
            and item["outcome_available_at"] <= _cutoff(train_end)
        ]
        validation_samples = [
            item
            for item in horizon_samples
            if validation_start <= item["published_date"] <= validation_end
            and item["outcome_available_at"] <= _cutoff(validation_end)
        ]
        test_samples = [
            item for item in horizon_samples if item["published_date"] >= test_start
        ]
        if not train_samples or not validation_samples or not test_samples:
            raise ValueError(f"empty temporal split for horizon {horizon}")

        train_text = [item["text"] for item in train_samples]
        train_target = np.asarray([item["target"] for item in train_samples])
        validation_text = [item["text"] for item in validation_samples]
        validation_target = np.asarray([item["target"] for item in validation_samples])
        test_text = [item["text"] for item in test_samples]
        test_target = np.asarray([item["target"] for item in test_samples])

        candidates = []
        for c_value in (0.1, 0.3, 1.0, 3.0):
            candidate = _pipeline(c_value)
            candidate.fit(train_text, train_target)
            validation_probability = candidate.predict_proba(validation_text)[:, 1]
            threshold, validation_metrics = _select_threshold(
                validation_target, validation_probability
            )
            candidates.append((validation_metrics, c_value, threshold))
        validation_metrics, c_value, threshold = max(
            candidates,
            key=lambda item: (
                item[0]["bullish_hit_rate_wilson_95pct"][0],
                item[0]["bullish_signals"],
            ),
        )

        fit_samples = train_samples + validation_samples
        model = _pipeline(c_value)
        model.fit(
            [item["text"] for item in fit_samples],
            np.asarray([item["target"] for item in fit_samples]),
        )
        test_probability = model.predict_proba(test_text)[:, 1]
        test_metrics = _metrics(test_target, test_probability, threshold)

        model_path = output_dir / f"horizon_{horizon}.joblib"
        joblib.dump(model, model_path)
        report["horizons"][str(horizon)] = {
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "test_samples": len(test_samples),
            "selected_c": c_value,
            "validation": validation_metrics,
            "test": test_metrics,
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
        }
        print(
            f"horizon={horizon} train={len(train_samples)} "
            f"validation={len(validation_samples)} test={len(test_samples)} "
            f"test_bullish_hit={test_metrics['bullish_hit_rate']}",
            flush=True,
        )

    manifest_path = output_dir / "shadow_evaluation.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="训练严格时间切分的利多影子模型")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "impact-shadow-v1",
    )
    args = parser.parse_args()
    train(output_dir=args.output_dir.resolve())


if __name__ == "__main__":
    main()

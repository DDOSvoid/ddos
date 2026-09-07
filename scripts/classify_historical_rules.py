#!/usr/bin/env python
"""Classify historical announcements with frozen high-precision title rules only."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, industry_registry
from src.database.engine import get_engine
from src.database.models import Announcement, Classification, Company
from src.pipeline.classification_rules import (
    ClassificationDecisionEngine,
    detect_document_type,
    detect_secondary_tags,
)

CONTRACT = "historical-rule-classification-v1"


def classify_historical(
    *,
    start: date,
    end: date,
    batch_size: int,
    report_path: Path,
) -> dict:
    engine = get_engine()
    decision_engine = ClassificationDecisionEngine()
    total = 0
    accepted = 0
    abstained = 0
    category_counts = Counter()

    while True:
        with Session(engine) as session:
            rows = (
                session.query(Announcement, Company.industry)
                .join(Company, Company.id == Announcement.company_id)
                .outerjoin(
                    Classification,
                    Classification.announcement_id == Announcement.id,
                )
                .filter(
                    Announcement.published_date >= start,
                    Announcement.published_date <= end,
                    Classification.id.is_(None),
                )
                .order_by(Announcement.id)
                .limit(batch_size)
                .all()
            )
            if not rows:
                break

            for announcement, industry in rows:
                decision = decision_engine.decide_rule(announcement.title)
                if decision is None:
                    major_category = "X"
                    sub_category = "other"
                    confidence = 0.0
                    classification_source = "abstained"
                    rule_id = None
                    document_type = detect_document_type(announcement.title)
                    relevance = "uncertain"
                    secondary_tags = detect_secondary_tags(announcement.title)
                    needs_review = True
                    review_reason = "历史规则未覆盖；未调用未通过时间外验证的模型"
                    abstained += 1
                else:
                    major_category = decision.major_category
                    sub_category = decision.sub_category
                    confidence = decision.confidence
                    classification_source = decision.classification_source
                    rule_id = decision.rule_id
                    document_type = decision.document_type
                    relevance = decision.relevance
                    secondary_tags = decision.secondary_tags
                    needs_review = decision.needs_review
                    review_reason = decision.review_reason
                    accepted += 1

                session.add(
                    Classification(
                        announcement_id=announcement.id,
                        major_category=major_category,
                        sub_category=sub_category,
                        confidence=confidence,
                        model_version=None,
                        industry=industry,
                        industry_group=industry_registry.resolve(industry),
                        classification_source=classification_source,
                        rule_id=rule_id,
                        document_type=document_type,
                        relevance=relevance,
                        secondary_tags=json.dumps(secondary_tags, ensure_ascii=False),
                        needs_review=needs_review,
                        review_status="pending" if needs_review else "auto_accepted",
                        review_reason=review_reason,
                        taxonomy_version="v2",
                    )
                )
                announcement.processing_status = "classified"
                category_counts[sub_category] += 1
                total += 1
            session.commit()
            print(
                f"classified={total}; accepted={accepted}; abstained={abstained}",
                flush=True,
            )

    with Session(engine) as session:
        summary_rows = (
            session.query(
                Classification.classification_source,
                Classification.sub_category,
                func.count(Classification.id),
            )
            .join(Announcement, Announcement.id == Classification.announcement_id)
            .filter(
                Announcement.published_date >= start,
                Announcement.published_date <= end,
            )
            .group_by(
                Classification.classification_source,
                Classification.sub_category,
            )
            .all()
        )
    total = sum(int(count) for _, _, count in summary_rows)
    abstained = sum(
        int(count)
        for source, _, count in summary_rows
        if source == "abstained"
    )
    accepted = total - abstained
    category_counts = Counter()
    for _, sub_category, count in summary_rows:
        category_counts[str(sub_category)] += int(count)

    report = {
        "contract": CONTRACT,
        "created_at": datetime.now(UTC).isoformat(),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "classifier": "frozen high-precision title rules only",
        "model_used": False,
        "total": total,
        "accepted": accepted,
        "abstained": abstained,
        "coverage": accepted / total if total else 0.0,
        "category_counts": dict(category_counts.most_common()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="历史公告高精度规则分类")
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data" / "backfills" / "historical_classification.json",
    )
    args = parser.parse_args()
    classify_historical(
        start=args.start_date,
        end=args.end_date,
        batch_size=max(args.batch_size, 1),
        report_path=args.report.resolve(),
    )


if __name__ == "__main__":
    main()

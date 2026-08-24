#!/usr/bin/env python
"""审计现有公告分类；默认只读，显式 --apply-rules 才应用高精度规则。

示例：
    python scripts/audit_classifications.py
    python scripts/audit_classifications.py --limit 500 --show 30
    python scripts/audit_classifications.py --apply-rules
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session

from src.config import PROJECT_ROOT
from src.database.engine import get_engine, init_db, set_database_url
from src.database.models import Announcement, ClassificationRevision
from src.database.repository import ClassificationRepository
from src.ml.classifier_wrapper import ClassificationResult
from src.pipeline.classification_rules import ClassificationDecisionEngine


def _rows(db_path: Path, limit: int | None) -> list[sqlite3.Row]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sql = (
            "SELECT a.id, a.title, c.major_category, c.sub_category, c.confidence "
            "FROM announcements a JOIN classifications c ON c.announcement_id = a.id "
            "ORDER BY a.id"
        )
        if limit is not None:
            sql += " LIMIT ?"
            return list(conn.execute(sql, (limit,)))
        return list(conn.execute(sql))


def _decision(engine: ClassificationDecisionEngine, row: sqlite3.Row):
    model = ClassificationResult(
        major_category=row["major_category"],
        sub_category=row["sub_category"],
        confidence=row["confidence"],
        # 旧库没有保存 top-2，审计时不伪造 margin。
        margin=None,
    )
    return engine.decide(row["title"], model)


def audit(db_path: Path, limit: int | None, show: int) -> list[tuple[sqlite3.Row, object]]:
    rows = _rows(db_path, limit)
    engine = ClassificationDecisionEngine()
    results = [(row, _decision(engine, row)) for row in rows]
    rule_matches = [(row, decision) for row, decision in results if decision.rule_id]
    changes = [
        (row, decision)
        for row, decision in rule_matches
        if (row["major_category"], row["sub_category"])
        != (decision.major_category, decision.sub_category)
    ]
    pending = [(row, decision) for row, decision in results if decision.needs_review]

    sources = Counter(decision.classification_source for _, decision in results)
    rules = Counter(decision.rule_id for _, decision in results if decision.rule_id)
    pending_review = sum(decision.needs_review for _, decision in results)
    print(f"审计公告: {len(results)}")
    print(f"高精度规则覆盖: {sum(rules.values())}")
    print(f"规则建议修改: {len(changes)}")
    print(f"建议进入复核队列: {pending_review}")
    print(f"决策来源: {dict(sources)}")
    print("命中规则 Top 15:")
    for rule_id, count in rules.most_common(15):
        print(f"  {rule_id}: {count}")

    if changes:
        print("\n建议修改样例:")
        for row, decision in changes[:show]:
            print(
                f"  #{row['id']} {row['sub_category']} -> {decision.sub_category} "
                f"[{decision.rule_id}] {row['title']}"
            )
    if pending:
        print("\n待复核样例:")
        for row, decision in pending[:show]:
            print(
                f"  #{row['id']} current={row['sub_category']} "
                f"candidate={decision.model_sub_category or '-'} "
                f"reason={decision.review_reason}: {row['title']}"
            )
    return rule_matches


def apply_rules(db_path: Path, rule_matches: list[tuple[sqlite3.Row, object]]) -> None:
    if not rule_matches:
        print("没有命中可应用的高精度规则。")
        return

    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = backup_dir / f"{db_path.stem}_before_classification_v2_{stamp}{db_path.suffix}"
    shutil.copy2(db_path, backup)
    print(f"数据库备份: {backup}")

    set_database_url(f"sqlite:///{db_path.as_posix()}")
    init_db()
    engine = get_engine(f"sqlite:///{db_path.as_posix()}")
    row_by_id = {row["id"]: row for row, _ in rule_matches}
    decision_by_id = {row["id"]: decision for row, decision in rule_matches}
    changed_count = 0
    protected_manual_count = 0

    with Session(engine) as session:
        announcements = (
            session.query(Announcement).filter(Announcement.id.in_(list(decision_by_id))).all()
        )
        for ann in announcements:
            current = ann.classification
            if current.classification_source == "manual":
                protected_manual_count += 1
                continue
            original_row = row_by_id[ann.id]
            decision = decision_by_id[ann.id]
            category_changed = (
                original_row["major_category"],
                original_row["sub_category"],
            ) != (decision.major_category, decision.sub_category)
            if category_changed:
                changed_count += 1
                session.add(
                    ClassificationRevision(
                        announcement_id=ann.id,
                        previous_major_category=current.major_category,
                        previous_sub_category=current.sub_category,
                        previous_confidence=current.confidence,
                        new_major_category=decision.major_category,
                        new_sub_category=decision.sub_category,
                        new_confidence=decision.confidence,
                        change_source="rule_audit_v2",
                        changed_by="audit_classifications.py",
                        note=decision.review_reason,
                    )
                )
            ClassificationRepository.upsert(
                session,
                announcement_id=ann.id,
                major_category=decision.major_category,
                sub_category=decision.sub_category,
                confidence=decision.confidence,
                model_version=current.model_version,
                industry=current.industry,
                industry_group=current.industry_group,
                classification_source=decision.classification_source,
                rule_id=decision.rule_id,
                document_type=decision.document_type,
                relevance=decision.relevance,
                secondary_tags=decision.secondary_tags,
                model_sub_category=current.sub_category,
                model_confidence=current.confidence,
                model_margin=None,
                needs_review=decision.needs_review,
                review_reason=decision.review_reason,
                taxonomy_version="v2",
            )
        session.commit()
    print(f"已写入规则元数据: {len(announcements) - protected_manual_count}")
    print(f"其中分类发生修改: {changed_count}")
    print(f"跳过已人工确认: {protected_manual_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="审计并可选应用公告标题高精度规则")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "ddos.db")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--show", type=int, default=20)
    parser.add_argument(
        "--apply-rules",
        action="store_true",
        help="备份数据库后，只应用高精度规则建议；不应用模型拒绝结果",
    )
    args = parser.parse_args()
    db_path = args.db.resolve()
    if not db_path.exists():
        parser.error(f"数据库不存在: {db_path}")

    rule_matches = audit(db_path, args.limit, args.show)
    if args.apply_rules:
        apply_rules(db_path, rule_matches)
    else:
        print("\n只读审计完成。添加 --apply-rules 才会备份并写入规则修改。")


if __name__ == "__main__":
    main()

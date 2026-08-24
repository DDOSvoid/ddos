#!/usr/bin/env python
"""真实公告标注助手。

流程：
  1. 导出待复核真实公告：
     python scripts/label_data.py --export --output data/labeled/to_review.jsonl
  2. 人工填写类别、reviewer、labeled_at、dataset_role，并将 label_status 改为 verified。
  3. 校验：
     python scripts/label_data.py --import --input data/labeled/to_review.jsonl
  4. 合并真实已验证批次：
     python scripts/label_data.py --merge --inputs data/labeled/batch_*.jsonl \
         --output data/labeled/real_verified.jsonl
  5. 生产训练：
     python scripts/setup_model.py --data data/labeled/real_verified.jsonl --force

合成数据、AI 建议和未经人工确认的规则标签不能进入生产训练、验证或校准。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session

from src.database.engine import get_engine
from src.database.models import Announcement, Classification
from src.training.label_contract import SCHEMA_VERSION, validate_label_item

DEFAULT_VERIFIED = Path("data/labeled/real_verified.jsonl")


def _already_labeled_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                announcement_id = json.loads(line).get("announcement_id")
            except json.JSONDecodeError:
                continue
            if announcement_id:
                ids.add(str(announcement_id))
    return ids


def export_unlabeled(
    output_path: str,
    count: int = 200,
    status: str | None = None,
    skip_labeled: bool = True,
    review_only: bool = True,
) -> int:
    """导出真实公告候选；默认只导出 needs_review=true。"""
    engine = get_engine()
    labeled_ids = _already_labeled_ids(DEFAULT_VERIFIED) if skip_labeled else set()

    with Session(engine) as session:
        query = session.query(Announcement)
        if status:
            query = query.filter_by(processing_status=status)
        if review_only:
            query = query.join(Classification, Announcement.classification).filter(
                Classification.needs_review.is_(True)
            )
        announcements = (
            query.order_by(
                Announcement.full_text.is_(None),
                Announcement.published_date.asc(),
                Announcement.id.asc(),
            )
            .limit(count + len(labeled_ids))
            .all()
        )

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        exported = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for ann in announcements:
                if ann.announcement_id in labeled_ids or exported >= count:
                    continue
                text = f"{ann.title or ''}\n{ann.full_text or ''}".strip()
                if len(text) < 10:
                    continue
                classification = ann.classification
                item = {
                    "schema_version": SCHEMA_VERSION,
                    "announcement_id": ann.announcement_id,
                    "published_at": (
                        ann.published_date.isoformat() if ann.published_date else ""
                    ),
                    "text": text[:3000],
                    "major_category": "",
                    "sub_category": "",
                    "label_source": "human",
                    "label_status": "candidate",
                    "labeled_at": "",
                    "reviewer": "",
                    "dataset_role": "",
                    "evidence": "",
                    "suggested_major_category": (
                        classification.major_category if classification else None
                    ),
                    "suggested_sub_category": (
                        classification.sub_category if classification else None
                    ),
                    "suggested_confidence": (
                        classification.confidence if classification else None
                    ),
                    "suggested_source": (
                        classification.classification_source if classification else None
                    ),
                    "suggested_rule_id": classification.rule_id if classification else None,
                    "suggested_document_type": (
                        classification.document_type if classification else None
                    ),
                    "suggested_relevance": classification.relevance if classification else None,
                    "model_candidate_sub_category": (
                        classification.model_sub_category if classification else None
                    ),
                    "model_candidate_confidence": (
                        classification.model_confidence if classification else None
                    ),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                exported += 1

    print(f"Exported {exported} real announcements to {out_path}")
    print("确认后填写 major_category/sub_category/reviewer/labeled_at/dataset_role，")
    print("并将 label_status 改为 verified。AI/规则建议不能直接作为生产真值。")
    return exported


def import_labeled(input_path: str, show_errors: int = 20) -> int:
    """校验文件中可用于生产训练的真实标注。"""
    path = Path(input_path)
    if not path.exists():
        print(f"File not found: {input_path}")
        return 0

    valid = 0
    invalid = 0
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                validate_label_item(json.loads(line), production=True)
                valid += 1
            except (json.JSONDecodeError, ValueError) as exc:
                if invalid < show_errors:
                    print(f"  Line {line_number}: {exc}")
                invalid += 1
    if invalid > show_errors:
        print(f"  ... 另有 {invalid - show_errors} 条错误未展开")
    print(f"Validation result: {valid} valid, {invalid} invalid")
    return valid


def merge_labeled(inputs: list[str], output_path: str) -> int:
    """合并真实人工验证标注；拒绝候选、AI、规则和 synthetic 标签。"""
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    merged: list[dict] = []
    rejected = 0

    for input_name in inputs:
        path = Path(input_name)
        if not path.exists():
            print(f"  [merge] 文件不存在，跳过: {input_name}")
            continue
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    validate_label_item(item, production=True)
                except (json.JSONDecodeError, ValueError) as exc:
                    print(f"  [merge] {path}:{line_number}: {exc}")
                    rejected += 1
                    continue
                announcement_id = str(item["announcement_id"])
                if announcement_id in seen_ids or item["text"] in seen_texts:
                    continue
                seen_ids.add(announcement_id)
                seen_texts.add(item["text"])
                merged.append(item)

    merged.sort(key=lambda item: (item["published_at"], item["announcement_id"]))
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[merge] {len(merged)} verified real labels -> {out_path}; rejected={rejected}")
    return len(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real announcement labeling helper")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--import", dest="import_file", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--output", default="data/labeled/to_review.jsonl")
    parser.add_argument("--input", default="data/labeled/to_review.jsonl")
    parser.add_argument("--inputs", nargs="+")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--show-errors", type=int, default=20)
    parser.add_argument("--status")
    parser.add_argument(
        "--all",
        action="store_true",
        help="导出全部公告；默认只导出 needs_review=true",
    )
    args = parser.parse_args()

    if args.export:
        export_unlabeled(
            args.output,
            args.count,
            status=args.status,
            review_only=not args.all,
        )
    elif args.import_file:
        import_labeled(args.input, show_errors=max(args.show_errors, 0))
    elif args.merge:
        if not args.inputs:
            parser.error("--merge 需要 --inputs")
        merge_labeled(args.inputs, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

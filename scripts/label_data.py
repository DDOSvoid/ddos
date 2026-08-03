#!/usr/bin/env python
"""数据标注助手 — 从数据库导出未标注的公告为 JSONL 文件。

人工标注流程:
  1. python scripts/label_data.py --export --output data/labeled/to_label.jsonl --count 500
  2. 打开 JSONL 文件，为每行填写正确的 sub_category
  3. python scripts/label_data.py --import --input data/labeled/labeled.jsonl

JSONL 格式（一行一个样本）:
  {"text": "公告标题\\n公告正文", "sub_category": "earnings_q1", "major_category": "A"}
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session

from src.database.engine import get_engine
from src.database.models import Announcement
from src.database.repository import AnnouncementRepository


def export_unlabeled(output_path: str, count: int = 200) -> int:
    """导出未标注公告为 JSONL。"""
    engine = get_engine()
    exported = 0

    with Session(engine) as session:
        announcements = AnnouncementRepository.get_by_status(
            session, "preprocessed", limit=count
        )

        if not announcements:
            print("No preprocessed announcements. Run fetcher + preprocessor first.")
            return 0

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            for ann in announcements:
                text = f"{ann.title or ''}\n{ann.full_text or ''}"
                item = {
                    "announcement_id": ann.announcement_id,
                    "text": text[:3000],  # 截断避免太长
                    "sub_category": "",   # 待标注
                    "major_category": "", # 待标注
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                exported += 1

    print(f"Exported {exported} announcements to {out_path}")
    print("\n标注说明:")
    print("  修改 sub_category 和 major_category 字段")
    print("  sub_category 选项见 config/event_types.yaml")
    print("  major_category: A/B/C/D/E/F/G")
    return exported


def import_labeled(input_path: str) -> int:
    """导入已标注数据（验证格式）。"""
    path = Path(input_path)
    if not path.exists():
        print(f"File not found: {input_path}")
        return 0

    valid = 0
    invalid = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if not item.get("sub_category"):
                    print(f"  Line {i}: missing sub_category")
                    invalid += 1
                    continue
                if not item.get("text"):
                    print(f"  Line {i}: missing text")
                    invalid += 1
                    continue
                valid += 1
            except json.JSONDecodeError as e:
                print(f"  Line {i}: JSON error: {e}")
                invalid += 1

    print(f"\nValidation result: {valid} valid, {invalid} invalid out of {valid + invalid} lines")
    return valid


def main():
    parser = argparse.ArgumentParser(description="Label data helper")
    parser.add_argument("--export", action="store_true", help="Export unlabeled data")
    parser.add_argument("--import", dest="import_file", action="store_true", help="Import (validate) labeled data")
    parser.add_argument("--output", default="data/labeled/to_label.jsonl", help="Output path for export")
    parser.add_argument("--input", default="data/labeled/labeled.jsonl", help="Input path for import")
    parser.add_argument("--count", type=int, default=200, help="Number of samples to export")
    args = parser.parse_args()

    if args.export:
        export_unlabeled(args.output, args.count)
    elif args.import_file:
        import_labeled(args.input)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

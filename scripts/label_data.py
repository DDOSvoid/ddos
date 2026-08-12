#!/usr/bin/env python
"""数据标注助手 — 打通"真实公告 → 人工标注 → 合并重训"的完整路径。

人工标注流程:
  1. 导出真实公告（默认全部状态，优先有正文的；可选 --status 过滤）
     python scripts/label_data.py --export --output data/labeled/to_label.jsonl --count 200
  2. 打开 JSONL 文件，为每行填写正确的 sub_category / major_category
  3. 校验已标注文件（可选，看格式是否合法）
     python scripts/label_data.py --import --input data/labeled/to_label.jsonl
  4. 与种子数据合并（按 announcement_id/text 去重，校验类别合法性）
     python scripts/label_data.py --merge \
         --inputs data/labeled/seed.jsonl data/labeled/to_label.jsonl \
         --output data/labeled/combined.jsonl
  5. 用合并数据重训分类器 + 温度校准
     python -m src.training.train_classifier --data data/labeled/combined.jsonl --output models/bert-classifier
     python scripts/calibrate_temperature.py --data data/labeled/combined.jsonl --model models/bert-classifier

JSONL 格式（一行一个样本）:
  {"announcement_id": "AN...", "text": "公告标题\\n公告正文", "sub_category": "earnings_q1", "major_category": "A"}

说明:
  - 默认 --export 从全部状态导出（管线跑完后公告已推进到 reported，
    旧的只导 preprocessed 会导致永远导出 0 条）。
  - 已出现在 data/labeled/combined.jsonl 中的公告会被自动跳过，避免重复标注。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session

from src.config import event_registry
from src.database.engine import get_engine
from src.database.models import Announcement

# 默认合并输出：已标注的真实数据 + 种子数据
DEFAULT_COMBINED = Path("data/labeled/combined.jsonl")


def export_unlabeled(
    output_path: str,
    count: int = 200,
    status: str | None = None,
    skip_labeled: bool = True,
) -> int:
    """导出未标注公告为 JSONL。

    status=None 时导出全部状态（默认，因管线跑完后无 preprocessed 残留）；
    指定 --status 则只导该状态。优先选有正文的公告（对标注更有信息量）。
    """
    engine = get_engine()
    exported = 0

    # 已标注过的 announcement_id 集合（避免重复导出）
    labeled_ids: set[str] = set()
    if skip_labeled and DEFAULT_COMBINED.exists():
        with open(DEFAULT_COMBINED, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    aid = json.loads(line).get("announcement_id")
                    if aid:
                        labeled_ids.add(aid)
                except json.JSONDecodeError:
                    continue

    with Session(engine) as session:
        q = session.query(Announcement)
        if status:
            q = q.filter_by(processing_status=status)
        # 有正文的排前面（full_text IS NULL → 0），再按发布日期倒序
        announcements = (
            q.order_by(
                Announcement.full_text.is_(None),
                Announcement.published_date.desc(),
            )
            .limit(count)
            .all()
        )

        if not announcements:
            print(
                f"No announcements to export (status={status or 'all'}). "
                "Run fetcher + preprocessor first."
            )
            return 0

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            for ann in announcements:
                if ann.announcement_id in labeled_ids:
                    continue
                text = f"{ann.title or ''}\n{ann.full_text or ''}"
                if len(text.strip()) < 10:
                    continue  # 标题+正文都空的不值得标
                item = {
                    "announcement_id": ann.announcement_id,
                    "text": text[:3000],  # 截断避免太长
                    "sub_category": "",    # 待标注
                    "major_category": "",  # 待标注
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                exported += 1

    skipped = len(labeled_ids)
    print(f"Exported {exported} announcements to {out_path} (skipped {skipped} already-labeled)")
    print("\n标注说明:")
    print("  修改 sub_category 和 major_category 字段")
    print("  sub_category 选项见 config/event_types.yaml")
    print("  major_category: A/B/C/D/E/F/G")
    print("  标注完成后运行 --merge 合并种子数据并重训")
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


def merge_labeled(inputs: list[str], output_path: str) -> int:
    """合并多个标注/种子 JSONL 为一个训练集。

    - 跳过缺 sub_category 的行（未标注）
    - 校验 sub_category 必须存在于 event_types.yaml（防错别字悄悄新增类别）
    - 按 announcement_id 去重（无 id 的行退化为按 text 去重）
    """
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    merged: list[dict] = []
    skipped = 0
    invalid_cat = 0

    for inp in inputs:
        path = Path(inp)
        if not path.exists():
            print(f"  [merge] 文件不存在，跳过: {inp}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if not item.get("sub_category") or not item.get("text"):
                    skipped += 1  # 未标注或空文本
                    continue
                sub = item["sub_category"]
                if sub not in event_registry.sub_to_major_map:
                    print(f"  [merge] 非法 sub_category '{sub}'（不在 event_types.yaml），丢弃")
                    invalid_cat += 1
                    continue
                # 幂等：id 优先，无 id 退化 text
                dedup_key = item.get("announcement_id") or f"text:{item['text']}"
                if dedup_key in seen_ids or item["text"] in seen_texts:
                    skipped += 1
                    continue
                seen_ids.add(dedup_key)
                seen_texts.add(item["text"])
                merged.append(item)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"[merge] {len(merged)} 条 → {out_path} "
        f"(跳过 {skipped} 未标注/重复, 丢弃 {invalid_cat} 非法类别)"
    )
    return len(merged)


def main():
    parser = argparse.ArgumentParser(description="Label data helper")
    parser.add_argument("--export", action="store_true", help="Export unlabeled real announcements")
    parser.add_argument("--import", dest="import_file", action="store_true", help="Validate a labeled file")
    parser.add_argument("--merge", action="store_true", help="Merge seed + labeled files into a training set")
    parser.add_argument("--output", default="data/labeled/to_label.jsonl", help="Output path for export/merge")
    parser.add_argument("--input", default="data/labeled/labeled.jsonl", help="Input path for import")
    parser.add_argument("--inputs", nargs="+", default=None,
                        help="Merge 的输入文件列表（如 seed.jsonl to_label.jsonl）")
    parser.add_argument("--count", type=int, default=200, help="Number of samples to export")
    parser.add_argument("--status", default=None, help="Export 仅导出指定状态（默认全部）")
    args = parser.parse_args()

    if args.export:
        export_unlabeled(args.output, args.count, status=args.status)
    elif args.import_file:
        import_labeled(args.input)
    elif args.merge:
        if not args.inputs:
            print("--merge 需要 --inputs 文件列表")
            parser.print_help()
            sys.exit(1)
        merge_labeled(args.inputs, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

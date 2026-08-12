#!/usr/bin/env python
"""诊断 BERT 分类器置信度 — 对若干无歧义样例输出 top-3 概率。

Usage:
    python scripts/check_confidence.py [--model models/bert-classifier]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    with open(Path(model_path) / "label_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    # 校准温度（与 wrapper 一致），缺失默认 1.0
    temp_file = Path(model_path) / "temperature.json"
    temperature = 1.0
    if temp_file.exists():
        temperature = float(json.load(open(temp_file, "r", encoding="utf-8"))["temperature"])
        logger.info(f"Using calibration temperature: {temperature}")
    return tokenizer, model, id2label, temperature


def softmax_topk(logits, id2label, k=3):
    probs = torch.softmax(logits, dim=-1)[0]
    topk = torch.topk(probs, k=k)
    return [(id2label[int(i)], float(p)) for i, p in zip(topk.indices, topk.values)]


# (text, 期望子类别) — 无歧义示例，来自真实公告常见措辞
EXAMPLES = [
    ("因公司2025年度经审计的净利润为负值且营业收入低于3亿元，公司股票可能被实施退市风险警示。",
     "st_delisting_risk"),
    ("公司2026年半年度报告已于6月30日披露，上半年实现营业收入50亿元，同比增长15%，归属于上市公司股东的净利润5亿元。",
     "earnings_h1"),
    ("公司近日收到法院送达的《应诉通知书》，因合同纠纷被起诉，涉案金额2000万元。",
     "litigation"),
    ("公司拟以自有资金回购公司股份，回购金额不低于500万元，不超过1000万元，回购价格不超过30元/股。",
     "buyback"),
    ("公司收到招标单位发来的中标通知书，确认公司为光伏组件采购项目的中标单位，中标金额8亿元。",
     "bid_win"),
    ("公司拟向150名激励对象授予限制性股票300万股，授予价格为12元/股，分三期解锁。",
     "equity_incentive"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/bert-classifier")
    args = parser.parse_args()

    tokenizer, model, id2label, temperature = load_model(args.model)
    logger.info(f"Loaded model: {args.model} ({len(id2label)} labels)")

    # 若库里有真实公告，追加一条
    texts = [t for t, _ in EXAMPLES]
    expected = [e for _, e in EXAMPLES]
    labels = [e for _, e in EXAMPLES]

    try:
        from sqlalchemy.orm import Session
        from src.database.engine import get_engine
        from src.database.models import Announcement
        with Session(get_engine()) as session:
            ann = session.query(Announcement).filter(
                Announcement.announcement_id.like("%中国宝安%")
            ).first()
            if ann is None:
                ann = session.query(Announcement).first()
            if ann is not None:
                from src.pipeline.preprocessor import Preprocessor
                combined = Preprocessor().preprocess_title_and_body(ann.title, ann.full_text)
                texts.append(combined)
                labels.append("?真实公告")
                logger.info(f"真实公告: {ann.title}")
    except Exception as e:
        logger.warning(f"跳过真实公告: {e}")

    ok = 0
    top1_probs = []
    for text, exp in zip(texts, labels):
        inputs = tokenizer(text, max_length=256, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits / temperature
        topk = softmax_topk(logits, id2label)
        top1 = topk[0]
        top1_probs.append(top1[1])
        mark = "✓" if top1[0] == exp else "✗"
        print(f"\n[{mark}] 期望={exp}  top1={top1[0]} ({top1[1]:.3f})")
        print(f"  text: {text[:60]}...")
        for label, p in topk:
            print(f"    {label}: {p:.4f}")
        if top1[0] == exp:
            ok += 1

    print(f"\n=== top-1 命中率: {ok}/{len(texts)} ===")
    print(f"=== top-1 平均概率: {np.mean(top1_probs):.3f} (min {min(top1_probs):.3f}, max {max(top1_probs):.3f}) ===")


if __name__ == "__main__":
    main()

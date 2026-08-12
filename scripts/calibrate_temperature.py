#!/usr/bin/env python
"""温度缩放校准 — 在验证集上搜索最优温度 T，锐化 softmax 以校正置信度。

种子模型在 30 类间 softmax 分布偏平（top-1 置信度 0.10-0.25），
低于下游 extraction_min_confidence=0.6 的可用门槛。温度缩放是标准校准手段：
推理时 logits /= T（T<1 锐化），argmax 不变，只重标定置信度。

用法:
    python scripts/calibrate_temperature.py \
        --data data/labeled/seed.jsonl \
        --model models/bert-classifier \
        --output models/bert-classifier/temperature.json

选择准则: 最小化 Expected Calibration Error（置信度与真实准确率一致）。
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

from src.training.dataset import build_label_mappings, create_splits, load_labeled_data


def ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error — 按置信度分桶，衡量 |准确率 - 置信度|。"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidences)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # 最后一桶包含边界
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        n = mask.sum()
        if n == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece_val += (n / total) * abs(acc - conf)
    return ece_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune softmax temperature on validation set")
    parser.add_argument("--data", default="data/labeled/seed.jsonl")
    parser.add_argument("--model", default="models/bert-classifier")
    parser.add_argument("--output", default="models/bert-classifier/temperature.json")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # ── 复现与训练一致的 val 划分 ─────────────────────
    texts, sub_labels, _ = load_labeled_data(args.data)
    unique_labels = sorted(set(sub_labels))
    label2id, id2label = build_label_mappings(unique_labels)
    splits = create_splits(texts, sub_labels, label2id, random_state=args.random_state)
    val_texts, val_ids = splits["val"]
    val_ids = np.asarray(val_ids)
    logger.info(f"Val samples: {len(val_texts)}")

    # ── 加载模型 ───────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()

    # ── 批量推理 val 集 ────────────────────────────────
    logits_all = []
    with torch.no_grad():
        for i in range(0, len(val_texts), 32):
            batch = val_texts[i : i + 32]
            inputs = tokenizer(batch, max_length=256, truncation=True, padding=True,
                               return_tensors="pt")
            logits_all.append(model(**inputs).logits.numpy())
    logits_all = np.concatenate(logits_all, axis=0)
    pred_ids = logits_all.argmax(axis=-1)
    correct = (pred_ids == val_ids).astype(float)
    logger.info(f"Val accuracy: {correct.mean():.4f}")

    # ── 网格搜索温度 ───────────────────────────────────
    temps = np.linspace(0.10, 1.00, 19)  # 0.10, 0.15, ..., 1.00
    results = []
    for T in temps:
        probs = torch.softmax(torch.from_numpy(logits_all / T), dim=-1).numpy()
        conf = probs.max(axis=-1)
        results.append({
            "T": round(float(T), 2),
            "ece": ece(conf, correct),
            "mean_conf": conf.mean(),
            "conf_correct": conf[correct == 1].mean(),
            "conf_wrong": conf[correct == 0].mean() if (correct == 0).any() else 0.0,
        })

    # 打印搜索表
    print("\n  T     ECE     mean_conf  conf_correct  conf_wrong")
    for r in results:
        print(f"  {r['T']:.2f}   {r['ece']:.4f}   {r['mean_conf']:.3f}     "
              f"{r['conf_correct']:.3f}         {r['conf_wrong']:.3f}")

    best = min(results, key=lambda r: r["ece"])
    print(f"\n✅ 最优温度 T={best['T']:.2f}  (ECE={best['ece']:.4f})")
    print(f"   校准后 top-1 平均置信度: {best['mean_conf']:.3f}")

    # ── 保存 ───────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"temperature": best["T"]}, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存 → {out}")


if __name__ == "__main__":
    main()

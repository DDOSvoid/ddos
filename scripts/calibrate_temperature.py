#!/usr/bin/env python
"""温度缩放校准 — 仅使用真实、人工验证的时间外 validation 数据。

推理时 logits /= T，argmax 不变，只重标定置信度。合成数据、AI 候选标签
和训练集均不允许用于生产校准。

用法:
    python scripts/calibrate_temperature.py \
        --data data/labeled/real_verified.jsonl \
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

from src.training.dataset import load_verified_labels


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
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="models/bert-classifier")
    parser.add_argument("--output", default="models/bert-classifier/temperature.json")
    parser.add_argument("--min-samples", type=int, default=100)
    args = parser.parse_args()

    # ── 只读取显式声明的时间外 validation 分区 ─────────
    samples = load_verified_labels(args.data)
    validation = [sample for sample in samples if sample.dataset_role == "validation"]
    if len(validation) < args.min_samples:
        raise ValueError(
            f"真实 validation 样本不足: {len(validation)} < {args.min_samples}"
        )

    mapping_path = Path(args.model) / "label_mapping.json"
    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)
    label2id = {key: int(value) for key, value in mapping["label2id"].items()}
    unknown = sorted({sample.sub_category for sample in validation} - set(label2id))
    if unknown:
        raise ValueError("validation 含模型未知类别: " + ", ".join(unknown))

    val_texts = [sample.text for sample in validation]
    val_ids = [label2id[sample.sub_category] for sample in validation]
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
    temps = np.geomspace(0.25, 4.0, 33)
    results = []
    for temperature in temps:
        probs = torch.softmax(
            torch.from_numpy(logits_all / temperature), dim=-1
        ).numpy()
        conf = probs.max(axis=-1)
        true_probs = np.clip(probs[np.arange(len(val_ids)), val_ids], 1e-12, 1.0)
        results.append({
            "T": float(temperature),
            "nll": float(-np.log(true_probs).mean()),
            "ece": float(ece(conf, correct)),
            "mean_conf": float(conf.mean()),
            "conf_correct": float(conf[correct == 1].mean()),
            "conf_wrong": (
                float(conf[correct == 0].mean()) if (correct == 0).any() else 0.0
            ),
        })

    # 打印搜索表
    print("\n  T     NLL      ECE     mean_conf  conf_correct  conf_wrong")
    for r in results:
        print(f"  {r['T']:.3f}  {r['nll']:.4f}  {r['ece']:.4f}   {r['mean_conf']:.3f}     "
              f"{r['conf_correct']:.3f}         {r['conf_wrong']:.3f}")

    best = min(results, key=lambda r: r["nll"])
    print(
        f"\n✅ 最优温度 T={best['T']:.3f}  "
        f"(NLL={best['nll']:.4f}, ECE={best['ece']:.4f})"
    )
    print(f"   校准后 top-1 平均置信度: {best['mean_conf']:.3f}")

    # ── 保存 ───────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "temperature": best["T"],
                "calibration_contract": "real-label-v1",
                "validation_samples": len(validation),
                "selection_metric": "negative_log_likelihood",
                "nll": best["nll"],
                "ece": best["ece"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"✅ 已保存 → {out}")


if __name__ == "__main__":
    main()

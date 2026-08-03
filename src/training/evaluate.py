"""模型评估 — 在测试集上评估分类器性能。"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.ml.classifier_wrapper import ClassifierWrapper


def evaluate_model(
    model_path: str,
    test_data_path: str,
    device: str = "cpu",
) -> dict:
    """评估已训练的模型。

    Args:
        model_path: 模型目录路径
        test_data_path: 测试数据 JSONL 路径

    Returns:
        {"macro_f1": float, "accuracy": float, "report": str, "confusion": list}
    """
    # 加载模型
    wrapper = ClassifierWrapper(model_path=model_path, device=device)

    # 加载测试数据
    texts = []
    true_labels = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            texts.append(item["text"])
            true_labels.append(item["sub_category"])

    # 加载 label mapping
    mapping_path = Path(model_path) / "label_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    label2id = mapping["label2id"]

    # 预测
    results = wrapper.classify_batch(texts)
    pred_labels = [r.sub_category for r in results]
    true_ids = [label2id.get(lbl, -1) for lbl in true_labels]

    # 计算指标
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    accuracy = sum(
        1 for t, p in zip(true_labels, pred_labels) if t == p
    ) / len(true_labels)

    report = classification_report(
        true_labels, pred_labels, zero_division=0
    )

    # 混淆矩阵
    all_labels = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)

    results_dict = {
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "num_samples": len(texts),
        "classification_report": report,
        "confusion_matrix_labels": all_labels,
        "confusion_matrix": cm.tolist(),
    }

    return results_dict


def print_evaluation(results: dict) -> None:
    """打印评估结果。"""
    print(f"\n{'='*60}")
    print(f"  Model Evaluation Results")
    print(f"{'='*60}")
    print(f"  Samples:    {results['num_samples']}")
    print(f"  Accuracy:   {results['accuracy']:.4f}")
    print(f"  Macro F1:   {results['macro_f1']:.4f}")
    print(f"{'='*60}")
    print(results["classification_report"])


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.training.evaluate <model_path> <test_data.jsonl>")
        sys.exit(1)

    results = evaluate_model(sys.argv[1], sys.argv[2])
    print_evaluation(results)

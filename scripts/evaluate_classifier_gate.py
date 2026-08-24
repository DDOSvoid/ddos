#!/usr/bin/env python
"""用严格时间外真实 test 分区评估分类模型并生成生产资格门禁。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import config, event_registry
from src.training.dataset import load_verified_labels
from src.training.label_contract import validate_temporal_roles


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if upper_inclusive
            else confidence < edges[index + 1]
        )
        if mask.any():
            value += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate real-data classifier gate")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="models/bert-classifier")
    parser.add_argument("--output", default="models/bert-classifier/evaluation_manifest.json")
    parser.add_argument("--min-test-per-class", type=int, default=10)
    parser.add_argument("--min-test-samples", type=int, default=200)
    parser.add_argument("--min-selective-accuracy", type=float, default=0.90)
    parser.add_argument("--min-coverage", type=float, default=0.20)
    args = parser.parse_args()

    samples = load_verified_labels(args.data)
    validate_temporal_roles(samples)
    test_samples = [sample for sample in samples if sample.dataset_role == "test"]

    model_path = Path(args.model)
    with open(model_path / "label_mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)
    label2id = {key: int(value) for key, value in mapping["label2id"].items()}
    id2label = {int(key): value for key, value in mapping["id2label"].items()}
    with open(model_path / "temperature.json", encoding="utf-8") as f:
        temperature = float(json.load(f)["temperature"])

    unknown = sorted({sample.sub_category for sample in test_samples} - set(label2id))
    if unknown:
        raise ValueError("test 含模型未知类别: " + ", ".join(unknown))

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    logits_batches = []
    texts = [sample.text for sample in test_samples]
    with torch.no_grad():
        for start in range(0, len(texts), 32):
            inputs = tokenizer(
                texts[start : start + 32],
                max_length=config.models.classifier.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            logits_batches.append(model(**inputs).logits.numpy())
    logits = np.concatenate(logits_batches, axis=0) / temperature
    probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    top_ids = np.argsort(probabilities, axis=-1)[:, -2:][:, ::-1]
    top1 = top_ids[:, 0]
    confidence = probabilities[np.arange(len(top1)), top1]
    second_confidence = probabilities[np.arange(len(top1)), top_ids[:, 1]]
    margin = confidence - second_confidence
    true_ids = np.asarray([label2id[sample.sub_category] for sample in test_samples])

    accepted = (
        (confidence >= config.pipeline.classifier_accept_confidence)
        & (margin >= config.pipeline.classifier_min_margin)
    )
    accepted_count = int(accepted.sum())
    selective_accuracy = (
        float(accuracy_score(true_ids[accepted], top1[accepted])) if accepted_count else 0.0
    )
    coverage = float(accepted.mean()) if len(accepted) else 0.0
    abstaining_predictions = np.where(accepted, top1, -1)
    labels = sorted(id2label)
    macro_f1_with_abstention = float(
        f1_score(
            true_ids,
            abstaining_predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    true_major = np.asarray(
        [event_registry.sub_to_major_map[sample.sub_category] for sample in test_samples]
    )
    predicted_major = np.asarray(
        [event_registry.sub_to_major_map.get(id2label[int(index)], "?") for index in top1]
    )
    selective_major_accuracy = (
        float(accuracy_score(true_major[accepted], predicted_major[accepted]))
        if accepted_count
        else 0.0
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        true_ids,
        abstaining_predictions,
        labels=labels,
        zero_division=0,
    )
    accepted_predictions: dict[int, int] = defaultdict(int)
    for prediction in top1[accepted]:
        accepted_predictions[int(prediction)] += 1
    per_class = {
        id2label[label_id]: {
            "test_support": int(support[index]),
            "accepted_predictions": accepted_predictions[label_id],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, label_id in enumerate(labels)
    }
    insufficient_classes = sorted(
        label for label, values in per_class.items()
        if values["test_support"] < args.min_test_per_class
    )

    correct = (top1 == true_ids).astype(float)
    checks = {
        "enough_test_samples": len(test_samples) >= args.min_test_samples,
        "per_class_test_support": not insufficient_classes,
        "selective_accuracy": selective_accuracy >= args.min_selective_accuracy,
        "coverage": coverage >= args.min_coverage,
    }
    manifest = {
        "evaluation_contract": "real-label-v1",
        "test_samples": len(test_samples),
        "accepted_samples": accepted_count,
        "coverage": coverage,
        "selective_subcategory_accuracy": selective_accuracy,
        "selective_major_accuracy": selective_major_accuracy,
        "macro_f1_with_abstention": macro_f1_with_abstention,
        "ece_all_predictions": expected_calibration_error(confidence, correct),
        "accept_confidence": config.pipeline.classifier_accept_confidence,
        "min_margin": config.pipeline.classifier_min_margin,
        "insufficient_test_classes": insufficient_classes,
        "per_class": per_class,
        "gate_checks": checks,
        "production_gate_pass": all(checks.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not manifest["production_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

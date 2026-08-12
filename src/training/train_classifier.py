#!/usr/bin/env python
"""微调 bert-base-chinese 做公告分类。

Usage:
    python -m src.training.train_classifier \
        --data data/labeled/annotations.jsonl \
        --output models/bert-classifier \
        --epochs 5 \
        --batch_size 16 \
        --lr 2e-5

标注数据格式 (JSONL):
    {"text": "公告标题\\n公告正文开头", "sub_category": "earnings_q1", "major_category": "A"}
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset as HFDataset
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.training.dataset import build_label_mappings, create_splits, load_labeled_data


def compute_metrics(eval_pred) -> dict:
    """计算评估指标。"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
    }


def train(
    data_path: str,
    output_dir: str,
    model_name: str = "bert-base-chinese",
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    fp16: bool = False,
    save_total_limit: int = 2,
    random_state: int = 42,
) -> str:
    """执行微调训练。返回模型保存路径。"""

    # ── 1. 加载数据 ─────────────────────────────────
    logger.info(f"Loading labeled data from {data_path}...")
    texts, sub_labels, _ = load_labeled_data(data_path)
    unique_labels = sorted(set(sub_labels))
    label2id, id2label = build_label_mappings(unique_labels)
    num_labels = len(unique_labels)

    logger.info(f"Loaded {len(texts)} samples, {num_labels} unique labels")
    logger.info(f"Labels: {unique_labels}")

    # ── 2. 划分数据集 ───────────────────────────────
    splits = create_splits(texts, sub_labels, label2id, random_state=random_state)
    logger.info(f"Train: {len(splits['train'][0])}, Val: {len(splits['val'][0])}, Test: {len(splits['test'][0])}")

    # ── 3. 加载 Tokenizer 和模型 ─────────────────────
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    # ── 4. 创建 HuggingFace Dataset ─────────────────
    def make_hf_dataset(texts_list, labels_list):
        # 列名必须用 "labels"：Trainer 的签名列与 compute_metrics 都依赖它，
        # "label" 会被 _remove_unused_columns 移除导致 loss/metrics 拿不到标签。
        return HFDataset.from_dict({
            "text": texts_list,
            "labels": labels_list,
        })

    train_hf = make_hf_dataset(*splits["train"])
    val_hf = make_hf_dataset(*splits["val"])
    test_hf = make_hf_dataset(*splits["test"])

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            max_length=max_length,
            padding=False,  # DataCollatorWithPadding handles this
            truncation=True,
        )

    train_hf = train_hf.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_hf = val_hf.map(tokenize_fn, batched=True, remove_columns=["text"])
    test_hf = test_hf.map(tokenize_fn, batched=True, remove_columns=["text"])

    # ── 5. 训练参数 ─────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # transformers 5.x: warmup_ratio 已移除 → 换算为 warmup_steps；evaluation_strategy → eval_strategy
    steps_per_epoch = max(1, len(train_hf) // batch_size)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",  # 不上报 wandb/tensorboard（transformers 5.x 已移除 logging_dir）
        dataloader_num_workers=0,  # Windows 多进程可能有问题
        seed=random_state,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── 6. 训练 ─────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training...")
    trainer.train()

    # ── 7. 评估 ─────────────────────────────────────
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_hf)
    logger.info(f"Test results: {test_results}")

    # 打印分类报告
    predictions = trainer.predict(test_hf)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    report = classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(num_labels)])
    logger.info(f"\nClassification Report:\n{report}")

    # ── 8. 保存模型 ─────────────────────────────────
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # 保存 label 映射
    with open(output_path / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    logger.info(f"Model saved to {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Train announcement classifier")
    parser.add_argument("--data", required=True, help="Path to labeled JSONL file")
    parser.add_argument("--output", default="models/bert-classifier", help="Output directory")
    parser.add_argument("--model", default="bert-base-chinese", help="Pretrained model name")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision (GPU only)")
    parser.add_argument("--grad_accum", type=int, default=1)
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        fp16=args.fp16,
        gradient_accumulation_steps=args.grad_accum,
    )


if __name__ == "__main__":
    main()

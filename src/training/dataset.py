"""PyTorch Dataset — 公告分类训练。

使用 transformers 的 AutoTokenizer 将公告文本转为模型输入。
分层抽样保证各大类别的 train/val/test 分布一致。
"""

import json
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.training.label_contract import VerifiedLabel, validate_label_item


class AnnouncementDataset(Dataset):
    """公告分类 PyTorch Dataset。

    Args:
        texts: 公告文本列表（已清洗的 title + body）
        labels: 类别索引列表（0 到 num_classes-1）
        tokenizer: HuggingFace tokenizer
        max_length: 最大 token 数
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),       # [max_length]
            "attention_mask": encoding["attention_mask"].squeeze(0),  # [max_length]
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_verified_labels(data_path: str | Path) -> list[VerifiedLabel]:
    """加载真实、已验证的生产标注。候选、AI 和合成标签会被拒绝。"""
    samples: list[VerifiedLabel] = []
    with open(data_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(validate_label_item(json.loads(line), production=True))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{data_path}:{line_number}: {exc}") from exc
    if not samples:
        raise ValueError(f"没有可用于生产训练的真实已验证标注: {data_path}")
    return samples


def load_labeled_data(data_path: str | Path) -> tuple[list[str], list[str], list[str]]:
    """从 JSONL 加载标注数据。

    每行 JSON 格式:
      {"text": "公告标题+正文", "label": "earnings_q1"}

    Returns:
        (texts, sub_category_labels, major_category_labels)
    """
    samples = load_verified_labels(data_path)
    return (
        [sample.text for sample in samples],
        [sample.sub_category for sample in samples],
        [sample.major_category for sample in samples],
    )


def create_declared_splits(
    samples: list[VerifiedLabel], label2id: dict[str, int]
) -> dict[str, tuple[list[str], list[int]]]:
    """使用标注中声明的严格时间分区，不进行随机拆分。"""
    role_map = {"train": "train", "validation": "val", "test": "test"}
    result: dict[str, tuple[list[str], list[int]]] = {}
    for source_role, output_role in role_map.items():
        selected = [sample for sample in samples if sample.dataset_role == source_role]
        result[output_role] = (
            [sample.text for sample in selected],
            [label2id[sample.sub_category] for sample in selected],
        )
    return result


def build_label_mappings(
    unique_labels: list[str],
) -> tuple[dict[str, int], dict[int, str]]:
    """构建 label → index 和 index → label 映射。"""
    label2id = {label: i for i, label in enumerate(sorted(unique_labels))}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def create_splits(
    texts: list[str],
    sub_labels: list[str],
    label2id: dict[str, int],
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, list]:
    """分层抽样创建 train/val/test 划分。

    Returns:
        {"train": (texts, ids), "val": (texts, ids), "test": (texts, ids)}
    """
    # 转为数值标签
    y = [label2id[lbl] for lbl in sub_labels]

    # 先分出 train / temp (val+test)
    x_train, x_temp, y_train, y_temp = train_test_split(
        texts, y,
        test_size=(val_size + test_size),
        stratify=y,
        random_state=random_state,
    )

    # 再从 temp 分出 val / test
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp,
        test_size=test_size / (val_size + test_size),
        stratify=y_temp,
        random_state=random_state,
    )

    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }

"""分类模型推理封装 — 加载微调后的 BERT 模型，提供批量推理接口。"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class ClassificationResult:
    """单个公告的分类结果。"""
    major_category: str    # "A" ~ "G"
    sub_category: str      # e.g., "earnings_q1"
    confidence: float      # 0.0 ~ 1.0


class ClassifierWrapper:
    """BERT 分类模型推理封装。

    Usage:
        wrapper = ClassifierWrapper(model_path="models/bert-classifier")
        result = wrapper.classify_single("公告标题正文...")
        results = wrapper.classify_batch(["文本1", "文本2", ...])
    """

    def __init__(
        self,
        model_path: str | None = None,
        model_name: str = "bert-base-chinese",
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        """
        Args:
            model_path: 微调模型路径（如果为 None，使用预训练模型但不做分类）
            model_name: HuggingFace 模型名（当 model_path 为 None 时用作后备）
            device: "cuda", "cpu", 或 None（自动检测）
            max_length: 最大 token 数
            batch_size: 批量推理大小
        """
        self.batch_size = batch_size
        self.max_length = max_length

        # 自动检测设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 确定加载路径
        load_path = model_path if model_path and Path(model_path).exists() else model_name
        self.model_path = model_path
        self._is_trained = model_path is not None and Path(model_path).exists()

        logger.info(f"Loading classifier from {load_path} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.model.to(self.device)
        self.model.eval()

        # 加载 label 映射
        self.label2id: dict[str, int] = {}
        self.id2label: dict[int, str] = {}
        self._load_label_mapping()

        # 温度缩放（校准）— 模型目录下 temperature.json 可选；
        # 缺失时默认 1.0（不缩放），保持向后兼容。
        self.temperature = self._load_temperature()

        # 子类别 → 大类别的静态映射（后备）
        self._sub_to_major: dict[str, str] = {}

    def _load_temperature(self) -> float:
        """从模型目录加载校准温度；缺失时返回 1.0。"""
        if not (self.model_path and Path(self.model_path).exists()):
            return 1.0
        temp_file = Path(self.model_path) / "temperature.json"
        if not temp_file.exists():
            return 1.0
        try:
            with open(temp_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            t = float(data.get("temperature", 1.0))
            logger.info(f"Loaded calibration temperature: {t}")
            return t if t > 0 else 1.0
        except Exception as e:
            logger.warning(f"Failed to load temperature, defaulting to 1.0: {e}")
            return 1.0

    def _load_label_mapping(self) -> None:
        """从模型目录加载 label 映射。"""
        if self.model_path and Path(self.model_path).exists():
            mapping_file = Path(self.model_path) / "label_mapping.json"
            if mapping_file.exists():
                with open(mapping_file, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
                    # id2label 的 key 是字符串，需要转 int
                    raw_id2label = mapping.get("id2label", {})
                    self.id2label = {int(k): v for k, v in raw_id2label.items()}
                    self.label2id = mapping.get("label2id", {})
                    logger.info(f"Loaded {len(self.id2label)} label mappings")

        # 如果模型自带 id2label（HuggingFace config 中保存的）
        if not self.id2label and hasattr(self.model.config, "id2label"):
            self.id2label = {
                int(k): v for k, v in self.model.config.id2label.items()
            }

    def classify_single(self, text: str) -> ClassificationResult:
        """对单个文本做分类。"""
        results = self.classify_batch([text])
        return results[0]

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """对一批文本做分类。返回与输入顺序一致的结果列表。"""
        if not texts:
            return []

        results: list[ClassificationResult] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_results = self._classify_batch_inner(batch)
            results.extend(batch_results)

        return results

    @torch.no_grad()
    def _classify_batch_inner(self, texts: list[str]) -> list[ClassificationResult]:
        """实际执行批量推理。"""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 推理（应用校准温度：logits / T，T<1 锐化、T>1 平滑，argmax 不变）
        outputs = self.model(**inputs)
        logits = outputs.logits / self.temperature
        probs = torch.softmax(logits, dim=-1)
        confidences, pred_ids = torch.max(probs, dim=-1)

        # 转换为结果
        results = []
        for pred_id, conf in zip(pred_ids.tolist(), confidences.tolist()):
            sub_category = self.id2label.get(pred_id, f"unknown_{pred_id}")
            major = self._get_major_category(sub_category)
            results.append(ClassificationResult(
                major_category=major,
                sub_category=sub_category,
                confidence=conf,
            ))

        return results

    def _get_major_category(self, sub_category: str) -> str:
        """从子类别推导大类别。"""
        # 尝试从静态映射获取
        if sub_category in self._sub_to_major:
            return self._sub_to_major[sub_category]
        # 后缀解析: "earnings_q1" → 前缀映射
        from src.config import event_registry
        return event_registry.get_major(sub_category) or "?"

    def set_sub_to_major_mapping(self, mapping: dict[str, str]) -> None:
        """设置子类别到大类别的静态映射（从 event_types.yaml 加载）。"""
        self._sub_to_major = mapping

    @property
    def num_labels(self) -> int:
        return len(self.id2label)

    @property
    def is_trained(self) -> bool:
        """是否加载了已训练的模型（而非仅预训练权重）。"""
        return self._is_trained

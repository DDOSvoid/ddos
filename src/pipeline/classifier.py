"""管线分类步骤 — 批量推理，将公告分类为 A-G 大类和 30+ 子类。"""

from pathlib import Path

from loguru import logger
from sqlalchemy.orm import Session

from src.config import config, event_registry, industry_registry
from src.database.engine import get_engine
from src.database.repository import AnnouncementRepository, ClassificationRepository
from src.ml.classifier_wrapper import ClassifierWrapper
from src.pipeline.classification_rules import ClassificationDecisionEngine
from src.pipeline.preprocessor import Preprocessor


class ClassificationStep:
    """管线分类步骤。

    1. 读取 processing_status='preprocessed' 的公告
    2. BERT 批量推理 → 大类别 + 子类别 + 置信度
    3. 写入 classifications 表
    4. 更新 processing_status='classified'
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        model_path = model_path or config.models.classifier.local_path
        batch_size = batch_size or config.pipeline.classifier_batch_size

        # 无微调模型时 fail-fast：直接加载原始 bert-base-chinese 只有 2 个标签，
        # 会产出 LABEL_0/1 垃圾分类；且会白白下载 ~400MB 权重。
        if not Path(model_path).exists():
            raise RuntimeError(
                f"分类模型未训练: {model_path} 不存在。"
                "请运行一键复现脚本: python scripts/setup_model.py"
            )

        self._wrapper: ClassifierWrapper | None = None
        self._wrapper_kwargs = {
            "model_path": model_path,
            "model_name": config.models.classifier.name,
            "device": device,
            "max_length": config.models.classifier.max_length,
            "batch_size": batch_size,
        }
        self.preprocessor = Preprocessor()
        self.decision_engine = ClassificationDecisionEngine()

    @property
    def wrapper(self) -> ClassifierWrapper:
        """只在标题规则无法判定时加载模型。"""
        if self._wrapper is None:
            self._wrapper = ClassifierWrapper(**self._wrapper_kwargs)
            self._wrapper.set_sub_to_major_mapping(event_registry.sub_to_major_map)
        return self._wrapper

    def run(self, limit: int = 500) -> int:
        """运行分类步骤。返回处理的公告数量。"""
        engine = get_engine()
        count = 0
        failed = 0

        with Session(engine) as session:
            announcements = AnnouncementRepository.get_by_status(
                session, "preprocessed", limit=limit
            )

            if not announcements:
                logger.info("No announcements to classify")
                return 0

            logger.info(f"Classifying {len(announcements)} announcements...")

            decisions = [self.decision_engine.decide_rule(ann.title) for ann in announcements]
            unresolved_indices = [
                index for index, decision in enumerate(decisions) if decision is None
            ]
            if unresolved_indices:
                texts = [
                    self.preprocessor.preprocess_title_and_body(
                        announcements[index].title,
                        announcements[index].full_text,
                    )
                    for index in unresolved_indices
                ]
                results = self.wrapper.classify_batch(texts)
                for index, result in zip(unresolved_indices, results, strict=True):
                    decisions[index] = self.decision_engine.decide(
                        announcements[index].title,
                        result,
                    )
            logger.info(
                "Classification routing: {} rule fast-path, {} model candidates",
                len(announcements) - len(unresolved_indices),
                len(unresolved_indices),
            )

            # 写入结果
            for ann, decision in zip(announcements, decisions, strict=True):
                try:
                    assert decision is not None
                    # 行业是公司自带属性，不经模型推理，直接从 company 带过来
                    industry = ann.company.industry if ann.company else None
                    industry_group = industry_registry.resolve(industry)
                    ClassificationRepository.upsert(
                        session,
                        announcement_id=ann.id,
                        major_category=decision.major_category,
                        sub_category=decision.sub_category,
                        confidence=decision.confidence,
                        model_version=config.models.classifier.name,
                        industry=industry,
                        industry_group=industry_group,
                        classification_source=decision.classification_source,
                        rule_id=decision.rule_id,
                        document_type=decision.document_type,
                        relevance=decision.relevance,
                        secondary_tags=decision.secondary_tags,
                        model_sub_category=decision.model_sub_category,
                        model_confidence=decision.model_confidence,
                        model_margin=decision.model_margin,
                        needs_review=decision.needs_review,
                        review_reason=decision.review_reason,
                        taxonomy_version="v2",
                    )
                    AnnouncementRepository.update_status(
                        session, ann.id, "classified"
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"Classification failed for {ann.announcement_id}: {e}")
                    AnnouncementRepository.update_status(
                        session, ann.id, "failed", str(e)
                    )
                    failed += 1

            session.commit()

        logger.info(f"Classification done: {count} classified, {failed} failed")
        return count

    def classify_single_text(self, text: str) -> dict:
        """对单条文本做分类（用于即时 API 调用）。"""
        title = text.splitlines()[0] if text else ""
        decision = self.decision_engine.decide_rule(title)
        if decision is None:
            result = self.wrapper.classify_single(text)
            decision = self.decision_engine.decide(title, result)
        return {
            "major_category": decision.major_category,
            "sub_category": decision.sub_category,
            "confidence": decision.confidence,
            "classification_source": decision.classification_source,
            "document_type": decision.document_type,
            "relevance": decision.relevance,
            "secondary_tags": decision.secondary_tags,
            "needs_review": decision.needs_review,
            "review_reason": decision.review_reason,
        }

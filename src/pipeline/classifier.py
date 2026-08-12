"""管线分类步骤 — 批量推理，将公告分类为 A-G 大类和 30+ 子类。"""

from pathlib import Path

from loguru import logger
from sqlalchemy.orm import Session

from src.config import config, event_registry
from src.database.engine import get_engine
from src.database.models import Announcement
from src.database.repository import AnnouncementRepository, ClassificationRepository
from src.ml.classifier_wrapper import ClassifierWrapper
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

        self.wrapper = ClassifierWrapper(
            model_path=model_path,
            model_name=config.models.classifier.name,
            device=device,
            max_length=config.models.classifier.max_length,
            batch_size=batch_size,
        )
        # 注入子类别→大类别的映射
        self.wrapper.set_sub_to_major_mapping(event_registry.sub_to_major_map)
        self.preprocessor = Preprocessor()

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

            # 准备文本（标题 + 正文拼接）
            texts = []
            for ann in announcements:
                combined = self.preprocessor.preprocess_title_and_body(
                    ann.title, ann.full_text
                )
                texts.append(combined)

            # 批量推理
            results = self.wrapper.classify_batch(texts)

            # 写入结果
            for ann, result in zip(announcements, results):
                try:
                    ClassificationRepository.upsert(
                        session,
                        announcement_id=ann.id,
                        major_category=result.major_category,
                        sub_category=result.sub_category,
                        confidence=result.confidence,
                        model_version=config.models.classifier.name,
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
        result = self.wrapper.classify_single(text)
        return {
            "major_category": result.major_category,
            "sub_category": result.sub_category,
            "confidence": result.confidence,
        }

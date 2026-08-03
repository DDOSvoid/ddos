"""文本预处理器 — 清洗公告正文，为分类模型准备输入。"""

from loguru import logger
from sqlalchemy.orm import Session

from src.config import config
from src.database.engine import get_engine
from src.database.models import Announcement
from src.database.repository import AnnouncementRepository
from src.utils.text_utils import clean_chinese_text, truncate_for_model


class Preprocessor:
    """公告文本预处理器。

    处理步骤:
      1. 从 DB 读取 processing_status='fetched' 的公告
      2. 清洗正文：HTML 标签 → 空白规范化 → 控制字符清理
      3. 截断到模型最大输入长度
      4. 更新 full_text（清洗后）和 processing_status='preprocessed'
    """

    def __init__(self, max_length: int | None = None) -> None:
        self.max_length = max_length or config.models.classifier.max_length

    def run(self, batch_size: int = 200) -> int:
        """执行预处理管线。返回处理的公告数量。"""
        engine = get_engine()
        count = 0
        failed = 0

        with Session(engine) as session:
            announcements = AnnouncementRepository.get_by_status(
                session, "fetched", limit=batch_size
            )

            if not announcements:
                logger.info("No announcements to preprocess")
                return 0

            logger.info(f"Preprocessing {len(announcements)} announcements...")

            for ann in announcements:
                try:
                    cleaned = self.preprocess_text(ann.full_text or ann.title)

                    # 更新公告文本
                    ann.full_text = cleaned
                    ann.processing_status = "preprocessed"
                    ann.error_message = None
                    count += 1

                except Exception as e:
                    logger.warning(f"Preprocess failed for {ann.announcement_id}: {e}")
                    AnnouncementRepository.update_status(
                        session, ann.id, "failed", str(e)
                    )
                    failed += 1

            session.commit()

        logger.info(f"Preprocessor done: {count} processed, {failed} failed")
        return count

    def preprocess_text(self, text: str) -> str:
        """执行完整文本预处理。"""
        if not text:
            return ""
        cleaned = clean_chinese_text(text)
        # 为分类模型的 BERT tokenizer 截断到合理长度
        # 512 tokens ≈ 约 300-400 中文字符（保守）
        # 但我们用字符截断做粗筛，tokenizer 会做精确截断
        max_chars = self.max_length * 3  # BERT 中文约 1.5-2 chars/token
        return truncate_for_model(cleaned, max_chars)

    def preprocess_title_and_body(self, title: str, body: str | None) -> str:
        """合并标题和正文并预处理（用于分类模型输入）。

        标题包含重要信息（如"业绩预告""回购公告"），拼接在前面提升分类准确率。
        """
        title_clean = clean_chinese_text(title or "")
        body_clean = clean_chinese_text(body or "")
        combined = f"{title_clean}\n{body_clean}"
        max_chars = self.max_length * 3
        return truncate_for_model(combined, max_chars)

"""管线字段提取步骤 — 用 LLM 从公告正文中提取结构化字段。

根据分类结果路由到对应的提取提示词。
提取与深度分析均使用 DeepSeek 模型。
"""

from loguru import logger
from sqlalchemy.orm import Session

from src.config import config, event_registry
from src.database.engine import get_engine
from src.database.models import Announcement
from src.database.repository import AnnouncementRepository, ExtractedFieldRepository
from src.ml.llm_client import LlmClient


# ── 提取提示词模板 ──────────────────────────────────────────────


def _build_extraction_prompt(announcement: Announcement, classification) -> str:
    """根据分类结果构建字段提取提示词。

    返回 LLM 的 user_prompt。
    """
    sub_cat = classification.sub_category
    sub_def = None
    for cat in event_registry.categories.values():
        if sub_cat in cat.subcategories:
            sub_def = cat.subcategories[sub_cat]
            break

    if sub_def is None:
        # 无特定模板，用通用提取
        return _generic_extraction_prompt(announcement)

    fields_desc = "\n".join(
        f'    "{field}": "提取的值（如无可设为 null）"'
        for field in sub_def.fields
    )

    prompt = f"""请从以下{announcement.title}中提取关键信息，返回 JSON 格式。

需要提取的字段:
{{
{fields_desc}
}}

注意事项:
1. 金额统一转为"万元"，百分比保留数值（如 15% 记为 15）
2. 如果公告中未提及某个字段，设值为 null
3. 只返回 JSON，不要其他文字

公告标题: {announcement.title}

公告正文:
{announcement.full_text or "（无正文）"}
"""
    return prompt


def _generic_extraction_prompt(announcement: Announcement) -> str:
    """通用提取提示词（无特定模板时使用）。"""
    return f"""请从以下公告中提取所有可能影响股价的关键信息，返回 JSON 格式。

字段:
{{
    "event_type": "事件类型（如: 财报/回购/减持/合同/并购/处罚 等）",
    "amount": "涉及金额（万元），无则 null",
    "ratio": "涉及比例（%），如 5% 记为 5，无则 null",
    "direction": "对股价的方向性影响: 'positive' / 'negative' / 'neutral'",
    "key_entities": ["涉及的关键主体/人名/公司名"],
    "summary": "一句话核心摘要（50字内）"
}}

公告标题: {announcement.title}

公告正文:
{announcement.full_text or "（无正文）"}
"""


# ── 提取系统提示词 ──────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个专业的中国A股公告分析助手。你的任务是从上市公司公告中准确提取结构化数据。

要求:
1. 始终保持客观、准确
2. 数值需精确提取，不要估算
3. 金额统一换算为"万元"（1亿=10000万）
4. 百分比提取数值部分（如 15% → 15）
5. 未提及的字段严格设为 null
6. 始终返回有效的 JSON"""


class ExtractionStep:
    """管线字段提取步骤。

    使用廉价 LLM（GPT-3.5/Qwen）批量提取公告中的结构化字段。
    """

    def __init__(self, llm_client: LlmClient | None = None) -> None:
        self.llm = llm_client or LlmClient()
        self.model = config.models.extraction.model
        self.min_confidence = config.pipeline.extraction_min_confidence

    def run(self, limit: int = 200) -> int:
        """执行字段提取。返回处理的公告数量。"""
        engine = get_engine()
        count = 0
        failed = 0

        with Session(engine) as session:
            # 获取已分类的公告（带预加载）
            announcements = AnnouncementRepository.get_with_classification(
                session, status="classified", limit=limit
            )

            if not announcements:
                logger.info("No announcements to extract")
                return 0

            logger.info(f"Extracting fields from {len(announcements)} announcements...")

            for ann in announcements:
                classification = ann.classification
                if classification is None:
                    AnnouncementRepository.update_status(
                        session, ann.id, "failed", "No classification"
                    )
                    failed += 1
                    continue

                # 跳过低置信度分类
                if classification.confidence < self.min_confidence:
                    logger.debug(
                        f"Skipping {ann.announcement_id}: confidence={classification.confidence:.2f}"
                    )
                    continue

                try:
                    # 构建提示词
                    user_prompt = _build_extraction_prompt(ann, classification)

                    # 调用 LLM
                    result = self.llm.complete_json(
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        model=self.model,
                        max_tokens=config.models.extraction.max_tokens,
                        temperature=config.models.extraction.temperature,
                        retries=2,
                    )

                    # 转换结果 → extracted_fields 格式
                    fields = self._parse_extraction_result(result, classification.sub_category)
                    ExtractedFieldRepository.bulk_insert(
                        session, ann.id, fields
                    )

                    AnnouncementRepository.update_status(
                        session, ann.id, "extracted"
                    )
                    count += 1

                except Exception as e:
                    logger.warning(f"Extraction failed for {ann.announcement_id}: {e}")
                    AnnouncementRepository.update_status(
                        session, ann.id, "failed", str(e)
                    )
                    failed += 1

            session.commit()

        logger.info(f"Extraction done: {count} extracted, {failed} failed")
        return count

    def _parse_extraction_result(
        self, result: dict, sub_category: str
    ) -> list[dict]:
        """将 LLM 返回的 JSON 解析为 ExtractedField 格式。"""
        sub_def = None
        for cat in event_registry.categories.values():
            if sub_category in cat.subcategories:
                sub_def = cat.subcategories[sub_category]
                break

        fields = []
        for key, value in result.items():
            # 跳过非预期字段
            if sub_def and key not in sub_def.fields:
                if key in ("event_type", "direction", "summary", "key_entities"):
                    pass  # 通用字段保留
                else:
                    continue

            # 判断类型
            if value is None:
                field_type = "text"
                field_value = None
            elif isinstance(value, (int, float)):
                field_type = "numeric"
                field_value = str(value)
            elif isinstance(value, bool):
                field_type = "text"
                field_value = str(value)
            elif isinstance(value, list):
                field_type = "json"
                field_value = str(value)
            else:
                field_type = "text"
                field_value = str(value)

            fields.append({
                "field_name": key,
                "field_value": field_value,
                "field_type": field_type,
                "unit": None,
                "confidence": 0.8,
                "model_used": self.model,
            })

        return fields


class DeepAnalysisStep:
    """深度分析步骤 — 对高影响事件使用 DeepSeek 模型。

    不在标准管线中自动运行；由 Reporter 按需调用。
    """

    def __init__(self, llm_client: LlmClient | None = None) -> None:
        self.llm = llm_client or LlmClient()
        self.model = config.models.analysis.model

    def analyze(self, announcement: Announcement) -> str:
        """对单个公告做深度分析，返回分析文本。"""
        classification = announcement.classification
        score = announcement.score
        fields = ExtractedFieldRepository.to_dict(
            next(get_engine())._dbapi_connection,  # 简化处理
            announcement.id,
        )
        # TODO: 修正 session 管理
        return self._do_analyze(announcement, classification, score, fields)

    def _do_analyze(
        self,
        ann: Announcement,
        classification,
        score,
        fields: dict,
    ) -> str:
        """执行深度分析。"""
        system_prompt = """你是一位资深的A股投资分析师。请基于公告信息撰写一段精炼的分析。

分析要点:
1. 该公告对股价的短期（1-3天）影响方向及逻辑
2. 该公告是否已被市场预期（从公告内容和历史背景判断）
3. 后续需要关注的关键时点或信号
4. 同类事件的统计规律（如有经验可参考）

控制在 200 字以内，直接给出结论。"""

        user_prompt = f"""请分析以下公告:

公司: {ann.company.stock_name if ann.company else "未知"} ({ann.company.stock_code if ann.company else "?"})
日期: {ann.published_date}
分类: {classification.major_category if classification else "?"} - {classification.sub_category if classification else "?"}
标题: {ann.title}

正文摘要: {(ann.full_text or "")[:1500]}

已提取的关键数据:
{fields}

评分: 方向={score.direction if score else "?"}, 强度={score.magnitude if score else "?"}, 综合={score.composite_score if score else "?"}
"""
        return self.llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            max_tokens=config.models.analysis.max_tokens,
            temperature=config.models.analysis.temperature,
        )

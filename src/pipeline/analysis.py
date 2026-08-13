"""深度分析 — 对高影响事件使用 LLM 生成叙事分析。

与提取步骤（extractor）解耦：提取产出结构化字段，深度分析产出散文式
解读，二者是不同形态的 LLM 任务（temp 0.0 vs 0.3，见 config.models.analysis）。
由 Reporter 按需对 Top-N 事件调用，不在标准管线状态机中单独占位。
"""

from sqlalchemy.orm import Session

from src.config import config
from src.database.engine import get_engine
from src.database.models import Announcement
from src.database.repository import ExtractedFieldRepository
from src.ml.llm_client import LlmClient


class DeepAnalysisStep:
    """深度分析步骤 — 对高影响事件使用 DeepSeek 模型。

    不在标准管线中自动运行；由 Reporter 按需调用。
    """

    def __init__(self, llm_client: LlmClient | None = None) -> None:
        self.llm = llm_client or LlmClient()
        self.model = config.models.analysis.model

    def analyze(
        self,
        announcement: Announcement,
        session: Session | None = None,
    ) -> str:
        """对单个公告做深度分析，返回分析文本。

        `session` 传入调用方已打开的 Session（推荐，避免重复建连）；
        省略时自建 Session 查询提取字段。
        """
        classification = announcement.classification
        score = announcement.score
        fields = self._get_fields(announcement.id, session)
        return self._do_analyze(announcement, classification, score, fields)

    @staticmethod
    def _get_fields(
        announcement_id: int, session: Session | None = None
    ) -> dict:
        """读取公告的提取字段。

        原实现误用 `next(get_engine())._dbapi_connection`（原始 DBAPI 连接，
        没有 `.query()`，`to_dict` 一调用即崩溃），已改为规范 Session 访问。
        """
        if session is not None:
            return ExtractedFieldRepository.to_dict(session, announcement_id)
        with Session(get_engine()) as s:
            return ExtractedFieldRepository.to_dict(s, announcement_id)

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

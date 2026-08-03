"""管线报告生成步骤 — 汇总评分结果，生成 Markdown 每日报告。

结构:
  1. 执行摘要（关键统计数据）
  2. 高影响事件表格（|composite_score| > 阈值）
  3. 分类分布统计
  4. 深度分析（GPT-5.5 对 Top-N 事件）
"""

from datetime import date, datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from sqlalchemy.orm import Session

from src.config import config
from src.database.engine import get_engine
from src.database.models import Announcement, Classification, Company, Score
from src.database.repository import (
    AnnouncementRepository,
    DailyReportRepository,
    ExtractedFieldRepository,
    ScoreRepository,
)
from src.ml.llm_client import LlmClient


class Reporter:
    """每日报告生成器。"""

    def __init__(self, llm_client: LlmClient | None = None) -> None:
        self.llm = llm_client or LlmClient()
        self.threshold = config.pipeline.report_high_impact_threshold
        self.top_n = config.pipeline.report_deep_analysis_top_n
        self.output_dir = config.resolve_path("data/reports")

    def run(self, target_date: date | None = None) -> str | None:
        """生成每日报告。返回报告文件路径。"""
        if target_date is None:
            target_date = date.today()

        engine = get_engine()
        with Session(engine) as session:
            # 1. 获取已评分的公告
            announcements = AnnouncementRepository.get_by_date_range(
                session, target_date, target_date, status="scored"
            )

            if not announcements:
                logger.info(f"No scored announcements for {target_date}")
                return None

            logger.info(f"Generating report for {target_date}: {len(announcements)} announcements")

            # 2. 收集评分数据
            scored_items = []
            for ann in announcements:
                if ann.score:
                    scored_items.append({
                        "announcement": ann,
                        "score": ann.score,
                        "classification": ann.classification,
                        "company": ann.company,
                    })

            if not scored_items:
                logger.info("No scored items to report")
                return None

            # 3. 按综合得分绝对值排序
            scored_items.sort(
                key=lambda x: abs(x["score"].composite_score), reverse=True
            )

            # 4. 分离高影响和普通事件
            high_impact = [
                item for item in scored_items
                if abs(item["score"].composite_score) >= self.threshold
            ]
            top_n = high_impact[:self.top_n]

            # 5. 生成 Markdown 报告
            report_content = self._generate_markdown(
                target_date, scored_items, high_impact, top_n
            )

            # 6. 生成摘要
            summary = self._generate_summary(target_date, high_impact, len(scored_items))

            # 7. 保存到数据库
            DailyReportRepository.upsert(
                session,
                report_date=target_date,
                report_title=f"公告事件驱动分析日报 — {target_date}",
                report_content=report_content,
                summary_text=summary,
                high_impact_count=len(high_impact),
                total_announcements=len(scored_items),
            )
            session.commit()

            # 8. 保存到文件
            self.output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"report_{target_date.isoformat()}.md"
            filepath = self.output_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_content)

            logger.info(f"Report saved: {filepath}")
            return str(filepath)

    def _generate_markdown(
        self,
        report_date: date,
        all_items: list[dict],
        high_impact: list[dict],
        top_n: list[dict],
    ) -> str:
        """生成 Markdown 报告全文。"""
        lines = []
        lines.append(f"# 公告事件驱动分析日报")
        lines.append(f"**日期**: {report_date}  |  **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # ── 执行摘要 ──
        lines.append("## 一、执行摘要")
        lines.append("")
        total = len(all_items)
        high_count = len(high_impact)
        bullish = sum(1 for x in all_items if x["score"].composite_score > 0.1)
        bearish = sum(1 for x in all_items if x["score"].composite_score < -0.1)
        neutral = total - bullish - bearish
        lines.append(f"- **公告总数**: {total}")
        lines.append(f"- **高影响事件**: {high_count} (阈值 |score| ≥ {self.threshold})")
        lines.append(f"- **利好信号**: {bullish} | **利空信号**: {bearish} | **中性**: {neutral}")
        lines.append("")

        # ── 高影响事件表格 ──
        lines.append("## 二、高影响事件")
        lines.append("")
        if high_impact:
            lines.append("| 股票 | 公告标题 | 分类 | 方向 | 强度 | 意外度 | 可信度 | 综合评分 |")
            lines.append("|------|---------|------|------|------|--------|--------|----------|")
            for item in high_impact[:20]:  # 最多 20 行
                ann = item["announcement"]
                score = item["score"]
                cls = item["classification"]
                comp = item["company"]
                title_short = (ann.title or "")[:30]
                lines.append(
                    f"| {comp.stock_name if comp else '?'}({comp.stock_code if comp else '?'}) "
                    f"| {title_short} "
                    f"| {cls.sub_category if cls else '?'} "
                    f"| {score.direction:+.2f} "
                    f"| {score.magnitude:.2f} "
                    f"| {score.surprise:.2f} "
                    f"| {score.credibility:.2f} "
                    f"| **{score.composite_score:+.3f}** |"
                )
        else:
            lines.append("*今日无高影响事件*")
        lines.append("")

        # ── 分类分布 ──
        lines.append("## 三、分类分布")
        lines.append("")
        cat_counts = {}
        for item in all_items:
            cls = item["classification"]
            if cls:
                key = f"{cls.major_category}-{cls.sub_category}"
                cat_counts[key] = cat_counts.get(key, 0) + 1
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- **{cat}**: {cnt} 条")
        lines.append("")

        # ── 深度分析 ──
        if top_n:
            lines.append("## 四、深度分析")
            lines.append("")
            for i, item in enumerate(top_n, 1):
                ann = item["announcement"]
                score = item["score"]
                cls = item["classification"]
                comp = item["company"]
                lines.append(f"### {i}. {comp.stock_name if comp else '?'} — {ann.title or '(无标题)'}")
                lines.append(f"")
                lines.append(f"- **股票**: {comp.stock_code} | **分类**: {cls.sub_category if cls else '?'}")
                lines.append(f"- **综合评分**: {score.composite_score:+.3f}")
                lines.append(f"- **方向**: {score.direction:+.2f} | **强度**: {score.magnitude:.2f} | **意外度**: {score.surprise:.2f} | **可信度**: {score.credibility:.2f}")
                lines.append(f"")

                # 尝试用 LLM 做深度分析
                try:
                    analysis = self._deep_analyze(ann, session_for_fields=True)
                    lines.append(f"> {analysis}")
                except Exception as e:
                    logger.warning(f"Deep analysis failed for {ann.announcement_id}: {e}")
                    lines.append(f"> *(AI 分析暂不可用)*")
                lines.append("")

        lines.append("---")
        lines.append(f"*由 DDOS 系统自动生成 · 仅供参考，不构成投资建议*")

        return "\n".join(lines)

    def _generate_summary(
        self,
        report_date: date,
        high_impact: list[dict],
        total_count: int,
    ) -> str:
        """生成一句话摘要。"""
        if not high_impact:
            return f"{report_date} 共处理 {total_count} 条公告，无高影响事件。"
        top = high_impact[0]
        return (
            f"{report_date} 共处理 {total_count} 条公告，其中 {len(high_impact)} 条高影响事件。"
            f" 最高评分: {top['announcement'].title[:30]}..."
            f" (综合: {top['score'].composite_score:+.3f})"
        )

    def _deep_analyze(
        self, ann: Announcement, session_for_fields: bool = False
    ) -> str:
        """对单条公告做深度分析。"""
        score = ann.score
        classification = ann.classification
        company = ann.company

        # 获取提取字段
        fields_str = "N/A"
        if session_for_fields:
            engine = get_engine()
            with Session(engine) as session:
                fields = ExtractedFieldRepository.to_dict(session, ann.id)
                fields_str = str(fields)[:1000]

        system_prompt = """你是资深A股投资分析师。请基于公告信息撰写精炼分析(150字内)：
1. 对股价的短期影响方向及逻辑
2. 是否已被市场预期
3. 后续关注时点"""

        user_prompt = f"""公告分析:

公司: {company.stock_name if company else '?'}({company.stock_code if company else '?'})
分类: {classification.sub_category if classification else '?'}
标题: {ann.title}
正文: {(ann.full_text or '')[:2000]}

评分: 方向={score.direction:+.2f} 强度={score.magnitude:.2f} 意外度={score.surprise:.2f}
提取字段: {fields_str}"""

        return self.llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=config.models.analysis.model,
            max_tokens=512,
            temperature=0.3,
        )

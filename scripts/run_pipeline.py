#!/usr/bin/env python
"""每日管线运行入口。

由 Windows 任务计划程序每日 18:00 调用:
    python scripts/run_pipeline.py --date yesterday

手动运行:
    python scripts/run_pipeline.py --date 2024-04-25
    python scripts/run_pipeline.py --date 2024-04-25 --stages fetch,preprocess
"""

import argparse
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

# 确保 src 在 Python path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.config import config
from src.database.engine import dispose_engine, get_engine, init_db
from src.database.repository import PipelineRunRepository
from src.utils.date_utils import parse_date
from src.utils.logging_config import setup_logging


def parse_date_arg(date_str: str) -> date:
    """解析日期参数，支持 'today', 'yesterday', 'YYYYMMDD', 'YYYY-MM-DD'。"""
    if date_str == "today":
        return date.today()
    elif date_str == "yesterday":
        return date.today() - timedelta(days=1)
    elif "-" in date_str:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    else:
        return parse_date(date_str)


def main():
    parser = argparse.ArgumentParser(description="DDOS 每日公告分析管线")
    parser.add_argument(
        "--date",
        default="yesterday",
        help="目标日期: today / yesterday / YYYYMMDD / YYYY-MM-DD",
    )
    parser.add_argument(
        "--stages",
        default="fetch,preprocess,classify,extract,score,report",
        help="要执行的阶段（逗号分隔）。默认全部。",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="覆盖数据库 URL",
    )
    args = parser.parse_args()

    # ── 日志 ─────────────────────────────────────────
    setup_logging()

    target_date = parse_date_arg(args.date)
    stages = [s.strip() for s in args.stages.split(",")]

    logger.info(f"Starting pipeline for {target_date}, stages: {stages}")

    # ── 数据库 ───────────────────────────────────────
    # 初始化表（如果尚未创建）
    init_db()

    from sqlalchemy.orm import Session

    engine = get_engine(args.db_url)

    # ── 管线运行记录 ─────────────────────────────────
    with Session(engine) as session:
        run_id = str(uuid.uuid4())
        run = PipelineRunRepository.create(
            session,
            run_id=run_id,
            run_type="full" if len(stages) >= 4 else stages[0],
            metadata={
                "target_date": str(target_date),
                "stages": stages,
                "config_snapshot": {
                    "extraction_model": config.models.extraction.model,
                    "analysis_model": config.models.analysis.model,
                },
            },
        )
        session.commit()
        logger.info(f"Pipeline run started: {run_id}")

    # ── 执行各阶段 ───────────────────────────────────
    # TODO: 各阶段将在后续周次中实现
    # 目前仅验证框架连通性

    try:
        from src.pipeline.fetcher import Fetcher
        from src.pipeline.preprocessor import Preprocessor
        # 后续阶段按需导入

        for stage in stages:
            logger.info(f"[{stage}] Starting...")

            if stage == "fetch":
                fetcher = Fetcher()
                count = fetcher.run(target_date)
                logger.info(f"[{stage}] Fetched {count} announcements")

            elif stage == "preprocess":
                preprocessor = Preprocessor()
                count = preprocessor.run()
                logger.info(f"[{stage}] Preprocessed {count} announcements")

            elif stage == "classify":
                logger.info(f"[{stage}] Classifier not yet implemented — skipping")

            elif stage == "extract":
                logger.info(f"[{stage}] Extractor not yet implemented — skipping")

            elif stage == "score":
                logger.info(f"[{stage}] Scorer not yet implemented — skipping")

            elif stage == "report":
                logger.info(f"[{stage}] Reporter not yet implemented — skipping")

            else:
                logger.warning(f"[{stage}] Unknown stage — skipping")

        # ── 标记成功 ──────────────────────────────────
        with Session(engine) as session:
            PipelineRunRepository.complete(session, run_id)
            session.commit()

        logger.info(f"Pipeline {run_id} completed successfully")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        with Session(engine) as session:
            PipelineRunRepository.fail(session, run_id, str(e))
            session.commit()
        sys.exit(1)

    finally:
        dispose_engine()


if __name__ == "__main__":
    main()

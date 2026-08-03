"""管线编排器 — 串行执行各管线步骤，管理状态流转。

每步按 processing_status 筛选待处理数据，出错可安全重跑。
"""

import uuid
from datetime import date, datetime
from typing import Optional

from loguru import logger
from sqlalchemy.orm import Session

from src.config import config
from src.database.engine import get_engine
from src.database.repository import PipelineRunRepository


class PipelineOrchestrator:
    """管线编排器 — 协调 fetcher → preprocessor → classifier → extractor → scorer → reporter。"""

    def __init__(self) -> None:
        self.run_id: str = ""
        self.stages_run: list[str] = []
        self.total_processed = 0
        self.total_failed = 0

    def run_daily_pipeline(
        self,
        target_date: date | None = None,
        stages: list[str] | None = None,
    ) -> dict:
        """执行每日管线。

        Args:
            target_date: 目标日期（默认昨天）
            stages: 要执行的阶段列表（默认全部）

        Returns:
            {"run_id": str, "stages": list, "processed": int, "failed": int, "status": str}
        """
        if target_date is None:
            target_date = date.today()

        if stages is None:
            stages = ["fetch", "preprocess", "classify", "extract", "score", "report"]

        self.run_id = str(uuid.uuid4())
        self.stages_run = []
        self.total_processed = 0
        self.total_failed = 0

        engine = get_engine()

        # 记录管线运行开始
        with Session(engine) as session:
            PipelineRunRepository.create(
                session,
                run_id=self.run_id,
                run_type="full" if len(stages) >= 4 else stages[0],
                metadata={
                    "target_date": str(target_date),
                    "stages": stages,
                },
            )
            session.commit()

        logger.info(f"Pipeline {self.run_id} starting: {stages} for {target_date}")

        try:
            for stage in stages:
                self._run_stage(stage, target_date)

            # 标记成功
            with Session(engine) as session:
                PipelineRunRepository.complete(
                    session,
                    self.run_id,
                    records_processed=self.total_processed,
                    records_failed=self.total_failed,
                )
                session.commit()

            logger.info(f"Pipeline {self.run_id} completed: {self.total_processed} processed, {self.total_failed} failed")
            return {
                "run_id": self.run_id,
                "stages": self.stages_run,
                "processed": self.total_processed,
                "failed": self.total_failed,
                "status": "completed",
            }

        except Exception as e:
            logger.exception(f"Pipeline {self.run_id} failed: {e}")
            with Session(engine) as session:
                PipelineRunRepository.fail(session, self.run_id, str(e))
                session.commit()
            return {
                "run_id": self.run_id,
                "stages": self.stages_run,
                "status": "failed",
                "error": str(e),
            }

    def _run_stage(self, stage: str, target_date: date) -> None:
        """执行单个阶段。"""

        if stage == "fetch":
            from src.pipeline.fetcher import Fetcher
            fetcher = Fetcher()
            count = fetcher.run(target_date=target_date)
            self.total_processed += count

        elif stage == "preprocess":
            from src.pipeline.preprocessor import Preprocessor
            preprocessor = Preprocessor()
            count = preprocessor.run()
            self.total_processed += count

        elif stage == "classify":
            from src.pipeline.classifier import ClassificationStep
            classifier = ClassificationStep()
            count = classifier.run()
            self.total_processed += count

        elif stage == "extract":
            from src.pipeline.extractor import ExtractionStep
            extractor = ExtractionStep()
            count = extractor.run()
            self.total_processed += count

        elif stage == "score":
            from src.pipeline.scorer import ImpactScorer
            scorer = ImpactScorer()
            count = scorer.run()
            self.total_processed += count

        elif stage == "report":
            from src.pipeline.reporter import Reporter
            reporter = Reporter()
            report_path = reporter.run(target_date=target_date)
            if report_path:
                self.total_processed += 1
                logger.info(f"Report: {report_path}")

        else:
            logger.warning(f"Unknown stage: {stage}, skipping")

        self.stages_run.append(stage)
        logger.info(f"  [{stage}] done")

    def get_status(self) -> dict:
        """获取当前管线状态概览。"""
        engine = get_engine()
        with Session(engine) as session:
            from src.database.repository import AnnouncementRepository
            statuses = [
                "fetched", "preprocessed", "classified",
                "extracted", "scored", "reported", "failed",
            ]
            counts = {}
            for s in statuses:
                counts[s] = AnnouncementRepository.count_by_status(session, s)
            return counts

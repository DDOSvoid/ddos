"""日志配置 — 基于 Loguru。"""

import sys
from pathlib import Path

from loguru import logger

from src.config import config


def setup_logging(log_file: str | None = None, level: str | None = None) -> None:
    """配置 Loguru 日志输出到控制台 + 文件。

    调用一次即可（在 scripts/run_pipeline.py 入口处）。
    """
    # 移除默认 handler
    logger.remove()

    # 控制台输出（彩色）
    logger.add(
        sys.stderr,
        level=level or config.logging.level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<level>{message}</level>",
        colorize=True,
    )

    # 文件输出（带 rotation 和 retention）
    log_path = log_file or config.logging.file
    # 解析相对路径
    if not Path(log_path).is_absolute():
        log_path = str(config.resolve_path(log_path))

    # 确保日志目录存在
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
               "{name}:{function}:{line} | {message}",
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        encoding="utf-8",
    )

    logger.info(f"Logging configured: console={level or config.logging.level}, file={log_path}")

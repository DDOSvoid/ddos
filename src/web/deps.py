"""FastAPI 会话依赖 — 测试通过 dependency_overrides 唯一替换点。"""

from collections.abc import Generator

from sqlalchemy.orm import Session

from src.database.engine import get_session


def get_db() -> Generator[Session, None, None]:
    """注入一个已打开的数据库会话（get_session 自带 finally 关闭）。"""
    yield from get_session()

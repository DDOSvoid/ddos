"""交易日历工具。"""

from datetime import date, datetime, timedelta


def is_weekday(d: date) -> bool:
    """检查是否为工作日（周一到周五）。"""
    return d.weekday() < 5


def previous_trading_day(d: date | None = None) -> date:
    """返回最近的交易日（跳过周末）。

    注意：不处理节假日，MVP 阶段仅跳过周末。
    """
    if d is None:
        d = date.today()
    d = d - timedelta(days=1)
    while not is_weekday(d):
        d = d - timedelta(days=1)
    return d


def get_trading_days(start: date, end: date) -> list[date]:
    """返回日期范围内的所有交易日（仅跳过周末）。"""
    days = []
    current = start
    while current <= end:
        if is_weekday(current):
            days.append(current)
        current += timedelta(days=1)
    return days


def format_date(d: date) -> str:
    """格式化为 YYYYMMDD 字符串。"""
    return d.strftime("%Y%m%d")


def parse_date(s: str) -> date:
    """从 YYYYMMDD 解析日期。"""
    return datetime.strptime(s, "%Y%m%d").date()

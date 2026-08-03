#!/usr/bin/env python
"""初始化数据库 — 从 Tushare 拉取股票列表并写入 companies 表。

Usage:
    python scripts/seed_database.py
    python scripts/seed_database.py --token YOUR_TUSHARE_TOKEN
    python scripts/seed_database.py --exchange SSE   # 仅上交所
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

# 确保 src 在 Python path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tushare as ts
from loguru import logger

from src.database.engine import dispose_engine, get_engine, init_db
from src.database.models import Company
from src.utils.logging_config import setup_logging

# 试点行业列表（与 config/tracked_companies.yaml 保持一致）
PILOT_INDUSTRIES = {
    "电气设备",
    "新能源",
    "光伏",
    "风电",
    "电池",
    "电力设备",
    "电网",
}


def seed_companies(token: str, exchange: str = "") -> int:
    """从 Tushare 拉取全部 A 股列表，写入数据库。"""
    ts.set_token(token)
    pro = ts.pro_api()

    logger.info("Fetching stock list from Tushare...")
    df = pro.stock_basic(
        exchange=exchange or "",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,list_date",
    )

    if df.empty:
        logger.warning("No stocks returned from Tushare")
        return 0

    logger.info(f"Got {len(df)} stocks from Tushare")

    engine = get_engine()
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        count = 0
        for _, row in df.iterrows():
            code = row["ts_code"]
            industry = row.get("industry") or ""
            is_pilot = any(p in industry for p in PILOT_INDUSTRIES) or industry in PILOT_INDUSTRIES
            is_tracked = is_pilot  # MVP: 仅跟踪试点行业

            existing = session.query(Company).filter_by(stock_code=code).first()
            if existing:
                existing.stock_name = row["name"]
                existing.industry = industry
                existing.is_tracked = is_tracked
                existing.updated_at = None  # trigger default
            else:
                session.add(
                    Company(
                        stock_code=code,
                        stock_name=row["name"],
                        exchange="SSE" if code.endswith(".SH") else "SZSE",
                        industry=industry if industry else None,
                        is_tracked=is_tracked,
                        tracked_since=date.today() if is_tracked else None,
                    )
                )
                count += 1

            if count % 500 == 0:
                session.flush()
                logger.info(f"  ... {count} new companies inserted")

        session.commit()
        total = session.query(Company).count()

    logger.info(f"Done: {count} new companies added, {total} total in database")
    return count


def main():
    parser = argparse.ArgumentParser(description="Seed database with stock list from Tushare")
    parser.add_argument("--token", help="Tushare API token (or set TUSHARE_TOKEN env var)")
    parser.add_argument("--exchange", default="", choices=["", "SSE", "SZSE"],
                        help="Filter by exchange (default: all)")
    args = parser.parse_args()

    setup_logging()
    token = args.token or os.getenv("TUSHARE_TOKEN")
    if not token:
        logger.error("Tushare token required. Set TUSHARE_TOKEN env var or pass --token")
        sys.exit(1)

    # 初始化数据库表
    logger.info("Initializing database...")
    init_db()

    seed_companies(token, args.exchange)

    dispose_engine()
    logger.info("Seed complete")


if __name__ == "__main__":
    main()

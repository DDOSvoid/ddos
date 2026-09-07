#!/usr/bin/env python
"""Initialize the database from the survivorship-safe Tushare A-share universe.

Usage:
    python scripts/seed_database.py
    python scripts/seed_database.py --token YOUR_TUSHARE_TOKEN
    python scripts/seed_database.py --exchange SSE
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import tushare as ts
import yaml
from loguru import logger

from src.database.engine import dispose_engine, get_engine, init_db
from src.database.models import Company
from src.utils.logging_config import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _tracking_config() -> dict:
    path = PROJECT_ROOT / "config" / "tracked_companies.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _parse_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    return datetime.strptime(text, "%Y%m%d").date()


def _download_universe(pro, exchange: str) -> pd.DataFrame:
    frames = []
    for list_status in ("L", "D", "P"):
        frame = pro.stock_basic(
            exchange=exchange or "",
            list_status=list_status,
            fields=(
                "ts_code,symbol,name,area,industry,market,exchange,"
                "list_status,list_date,delist_date"
            ),
        )
        if not frame.empty:
            frame = frame.copy()
            frame["list_status"] = list_status
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first")


def seed_companies(token: str, exchange: str = "") -> int:
    """Download L/D/P stocks and persist a versioned universe snapshot."""
    ts.set_token(token)
    pro = ts.pro_api()
    tracking = _tracking_config()
    track_all = bool(tracking.get("track_all"))
    pilot_industries = set(str(value) for value in tracking.get("pilot_industries", []))
    tracked_codes = set(str(value) for value in tracking.get("tracked_codes", []))

    logger.info("Fetching L/D/P stock universe from Tushare...")
    frame = _download_universe(pro, exchange)
    if frame.empty:
        logger.warning("No stocks returned from Tushare")
        return 0
    logger.info(f"Got {len(frame)} unique L/D/P stocks from Tushare")

    from sqlalchemy.orm import Session

    universe_as_of = datetime.now(UTC)
    engine = get_engine()
    with Session(engine) as session:
        inserted = 0
        for _, row in frame.iterrows():
            code = str(row["ts_code"])
            industry = str(row.get("industry") or "").strip()
            is_pilot = any(value and value in industry for value in pilot_industries)
            is_tracked = track_all or code in tracked_codes or is_pilot
            exchange_value = str(row.get("exchange") or "").strip()
            if exchange_value not in {"SSE", "SZSE", "BSE"}:
                exchange_value = "SSE" if code.endswith(".SH") else "SZSE"

            values = {
                "stock_name": str(row["name"]),
                "exchange": exchange_value,
                "industry": industry or None,
                "listing_date": _parse_date(row.get("list_date")),
                "delisting_date": _parse_date(row.get("delist_date")),
                "list_status": str(row.get("list_status") or "") or None,
                "universe_source": "tushare_stock_basic_ldp_v1",
                "universe_as_of": universe_as_of,
                "is_tracked": is_tracked,
            }
            existing = session.query(Company).filter_by(stock_code=code).first()
            if existing:
                for key, value in values.items():
                    setattr(existing, key, value)
                if is_tracked and existing.tracked_since is None:
                    existing.tracked_since = date.today()
            else:
                session.add(
                    Company(
                        stock_code=code,
                        tracked_since=date.today() if is_tracked else None,
                        **values,
                    )
                )
                inserted += 1
            if inserted and inserted % 500 == 0:
                session.flush()
                logger.info(f"  ... {inserted} new companies inserted")
        session.commit()
        total = session.query(Company).count()
    logger.info(f"Done: {inserted} new companies added, {total} total in database")
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the point-in-time-capable A-share L/D/P universe from Tushare"
    )
    parser.add_argument("--token", help="Tushare token (prefer TUSHARE_TOKEN in .env)")
    parser.add_argument(
        "--exchange", default="", choices=["", "SSE", "SZSE", "BSE"]
    )
    args = parser.parse_args()

    setup_logging()
    token = args.token or os.getenv("TUSHARE_TOKEN")
    if not token:
        logger.error("Tushare token required. Set TUSHARE_TOKEN in local .env")
        sys.exit(1)
    init_db()
    seed_companies(token, args.exchange)
    dispose_engine()


if __name__ == "__main__":
    main()

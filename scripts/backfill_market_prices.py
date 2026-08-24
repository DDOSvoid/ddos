#!/usr/bin/env python
"""Backfill causal stock/index daily bars for announcement outcome evaluation."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.engine import get_engine, init_db
from src.database.models import Announcement, Company, DailyPrice
from src.pipeline.fetcher import TushareClient


def _optional_float(value) -> float | None:
    return None if pd.isna(value) else float(value)


def upsert_bars(
    session: Session,
    frame: pd.DataFrame,
    *,
    instrument_type: str,
    fetched_at: datetime,
) -> int:
    if frame.empty:
        return 0
    rows = []
    for item in frame.to_dict("records"):
        rows.append(
            {
                "instrument_code": str(item["ts_code"]),
                "instrument_type": instrument_type,
                "trade_date": date.fromisoformat(
                    f"{item['trade_date'][:4]}-{item['trade_date'][4:6]}-"
                    f"{item['trade_date'][6:8]}"
                ),
                "open_price": float(item["open"]),
                "high_price": float(item["high"]),
                "low_price": float(item["low"]),
                "close_price": float(item["close"]),
                "pre_close": _optional_float(item.get("pre_close")),
                "pct_change": _optional_float(item.get("pct_chg")),
                "volume": _optional_float(item.get("vol")),
                "amount": _optional_float(item.get("amount")),
                "source": "tushare",
                "fetched_at": fetched_at,
            }
        )
    statement = sqlite_insert(DailyPrice).values(rows)
    statement = statement.on_conflict_do_update(
        index_elements=["instrument_code", "trade_date"],
        set_={
            "instrument_type": statement.excluded.instrument_type,
            "open_price": statement.excluded.open_price,
            "high_price": statement.excluded.high_price,
            "low_price": statement.excluded.low_price,
            "close_price": statement.excluded.close_price,
            "pre_close": statement.excluded.pre_close,
            "pct_change": statement.excluded.pct_change,
            "volume": statement.excluded.volume,
            "amount": statement.excluded.amount,
            "source": statement.excluded.source,
            "fetched_at": statement.excluded.fetched_at,
        },
    )
    session.execute(statement)
    return len(rows)


def backfill(
    *,
    start_date: date | None,
    end_date: date | None,
    benchmark_code: str,
    limit: int,
) -> dict:
    init_db()
    engine = get_engine()
    with Session(engine) as session:
        minimum = session.query(Announcement.published_date).order_by(
            Announcement.published_date.asc()
        ).first()
        maximum = session.query(Announcement.published_date).order_by(
            Announcement.published_date.desc()
        ).first()
        if not minimum or not maximum:
            raise ValueError("no announcements available")
        start = start_date or minimum[0] - timedelta(days=10)
        end = end_date or min(date.today(), maximum[0] + timedelta(days=10))
        codes = list(
            session.scalars(
                select(Company.stock_code)
                .join(Announcement, Announcement.company_id == Company.id)
                .distinct()
                .order_by(Company.stock_code)
            )
        )
    if limit > 0:
        codes = codes[:limit]

    client = TushareClient()
    fetched_at = datetime.now(UTC)
    stock_rows = 0
    empty_codes = []
    for index, code in enumerate(codes, 1):
        frame = client.get_daily(
            ts_code=code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        with Session(engine) as session:
            stock_rows += upsert_bars(
                session,
                frame,
                instrument_type="stock",
                fetched_at=fetched_at,
            )
            session.commit()
        if frame.empty:
            empty_codes.append(code)
        if index % 20 == 0 or index == len(codes):
            print(
                f"stock progress {index}/{len(codes)}; "
                f"rows={stock_rows}; empty={len(empty_codes)}",
                flush=True,
            )

    benchmark = client.get_index_daily(
        benchmark_code,
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
    )
    with Session(engine) as session:
        benchmark_rows = upsert_bars(
            session,
            benchmark,
            instrument_type="index",
            fetched_at=fetched_at,
        )
        session.commit()
    result = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "stock_codes": len(codes),
        "stock_rows": stock_rows,
        "empty_codes": empty_codes,
        "benchmark_code": benchmark_code,
        "benchmark_rows": benchmark_rows,
    }
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="回填公告预测所需的日线行情")
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    backfill(
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark_code=args.benchmark,
        limit=max(args.limit, 0),
    )


if __name__ == "__main__":
    main()

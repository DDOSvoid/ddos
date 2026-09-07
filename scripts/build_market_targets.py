#!/usr/bin/env python
"""Build real forward excess-return targets from cached daily bars."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.engine import get_engine, init_db
from src.database.models import (
    Announcement,
    AnnouncementMarketTarget,
    Company,
    DailyPrice,
)
from src.prediction.splits import load_prediction_split_contract

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _available_at(exit_date: date) -> datetime:
    return datetime.combine(exit_date, time(15, 0), tzinfo=SHANGHAI).astimezone(UTC)


def build_targets(
    *,
    benchmark_code: str,
    horizons: tuple[int, ...],
    output: Path,
    announcement_ids: frozenset[str] | None = None,
    announcement_plan_sha256: str | None = None,
) -> dict:
    init_db()
    split = load_prediction_split_contract()
    engine = get_engine()
    with Session(engine) as session:
        announcement_query = (
            session.query(
                Announcement.id,
                Announcement.announcement_id,
                Announcement.published_date,
                Company.stock_code,
            )
            .join(Company, Company.id == Announcement.company_id)
            .filter(
                Announcement.published_date >= split.train.start,
                Announcement.published_date <= split.train.end,
            )
        )
        if announcement_ids is not None:
            if not announcement_ids:
                raise ValueError("announcement plan is empty")
            announcement_query = announcement_query.filter(
                Announcement.announcement_id.in_(announcement_ids)
            )
        announcement_rows = announcement_query.order_by(
            Announcement.published_date, Announcement.id
        ).all()
        if announcement_ids is not None and len(announcement_rows) != len(announcement_ids):
            raise ValueError(
                "announcement plan does not resolve one unique train row per external ID"
            )
        price_rows = (
            session.query(DailyPrice)
            .filter(DailyPrice.trade_date <= split.train.end)
            .order_by(DailyPrice.instrument_code, DailyPrice.trade_date)
            .all()
        )

    prices: dict[str, list[DailyPrice]] = defaultdict(list)
    for row in price_rows:
        prices[row.instrument_code].append(row)
    price_dates = {code: [row.trade_date for row in rows] for code, rows in prices.items()}
    benchmark = {row.trade_date: row for row in prices.get(benchmark_code, [])}
    if not benchmark:
        raise ValueError(f"benchmark prices missing: {benchmark_code}")

    target_rows = []
    pending = Counter()
    excluded = Counter()
    role_counts: dict[str, Counter] = defaultdict(Counter)
    for announcement_id, external_id, published_date, stock_code in announcement_rows:
        stock_prices = prices.get(stock_code, [])
        start_index = bisect.bisect_right(price_dates.get(stock_code, []), published_date)
        future_bars = stock_prices[start_index:]
        for horizon in horizons:
            if len(future_bars) < horizon:
                pending[horizon] += 1
                continue
            entry = future_bars[0]
            exit_bar = future_bars[horizon - 1]
            benchmark_entry = benchmark.get(entry.trade_date)
            benchmark_exit = benchmark.get(exit_bar.trade_date)
            if benchmark_entry is None or benchmark_exit is None:
                pending[horizon] += 1
                continue
            stock_return = exit_bar.close_price / entry.open_price - 1
            benchmark_return = benchmark_exit.close_price / benchmark_entry.open_price - 1
            excess_return = stock_return - benchmark_return
            actual_direction = 1 if excess_return > 0 else -1 if excess_return < 0 else 0
            outcome_available_at = _available_at(exit_bar.trade_date)
            dataset_role = split.role_for(published_date, outcome_available_at)
            if dataset_role != "train":
                excluded[dataset_role] += 1
                continue
            evidence = {
                "announcement_id": external_id,
                "stock_code": stock_code,
                "entry": {
                    "date": entry.trade_date.isoformat(),
                    "stock_open": entry.open_price,
                    "benchmark_open": benchmark_entry.open_price,
                },
                "exit": {
                    "date": exit_bar.trade_date.isoformat(),
                    "stock_close": exit_bar.close_price,
                    "benchmark_close": benchmark_exit.close_price,
                },
            }
            target_rows.append(
                {
                    "announcement_id": announcement_id,
                    "horizon_sessions": horizon,
                    "entry_date": entry.trade_date,
                    "exit_date": exit_bar.trade_date,
                    "entry_price": entry.open_price,
                    "exit_price": exit_bar.close_price,
                    "benchmark_code": benchmark_code,
                    "stock_return": stock_return,
                    "benchmark_return": benchmark_return,
                    "excess_return": excess_return,
                    "actual_direction": actual_direction,
                    "outcome_available_at": outcome_available_at,
                    "prices_sha256": _sha256(evidence),
                    "dataset_role": dataset_role,
                    "split_contract": split.contract,
                    "split_source_sha256": split.source_sha256,
                    "created_at": datetime.now(UTC),
                }
            )
            role_counts[dataset_role][horizon] += 1

    with Session(engine) as session:
        for offset in range(0, len(target_rows), 500):
            batch = target_rows[offset : offset + 500]
            statement = sqlite_insert(AnnouncementMarketTarget).values(batch)
            statement = statement.on_conflict_do_update(
                index_elements=["announcement_id", "horizon_sessions"],
                set_={
                    "entry_date": statement.excluded.entry_date,
                    "exit_date": statement.excluded.exit_date,
                    "entry_price": statement.excluded.entry_price,
                    "exit_price": statement.excluded.exit_price,
                    "benchmark_code": statement.excluded.benchmark_code,
                    "stock_return": statement.excluded.stock_return,
                    "benchmark_return": statement.excluded.benchmark_return,
                    "excess_return": statement.excluded.excess_return,
                    "actual_direction": statement.excluded.actual_direction,
                    "outcome_available_at": statement.excluded.outcome_available_at,
                    "prices_sha256": statement.excluded.prices_sha256,
                    "dataset_role": statement.excluded.dataset_role,
                    "split_contract": statement.excluded.split_contract,
                    "split_source_sha256": statement.excluded.split_source_sha256,
                    "created_at": statement.excluded.created_at,
                },
            )
            session.execute(statement)
        session.commit()

    report = {
        "contract": "causal-market-target-v2",
        "split_contract": split.contract,
        "split_source_sha256": split.source_sha256,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_code": benchmark_code,
        "entry_rule": "first stock trading session strictly after publication; entry at open",
        "exit_rule": "horizon-th stock trading session close",
        "target_rule": "stock return minus benchmark return; sign gives actual direction",
        "warning": "retrospective training targets, not historical predictions or accuracy",
        "dataset_role": "train",
        "official_test_queried": False,
        "forward_validation_queried": False,
        "announcement_selection": (
            {
                "mode": "explicit_train_only_plan",
                "announcement_ids": len(announcement_ids),
                "announcement_plan_sha256": announcement_plan_sha256,
            }
            if announcement_ids is not None
            else {"mode": "all_train_announcements"}
        ),
        "announcements": len(announcement_rows),
        "targets": len(target_rows),
        "by_horizon": {
            str(horizon): {
                "matured": sum(counts[horizon] for counts in role_counts.values()),
                "pending": pending[horizon],
            }
            for horizon in horizons
        },
        "by_dataset_role": {
            role: {str(horizon): counts[horizon] for horizon in horizons}
            for role, counts in sorted(role_counts.items())
        },
        "excluded_after_role_check": dict(sorted(excluded.items())),
        "sealed_labels": ["test", "quarantine", "forward_validation"],
        "label_visibility": (
            "report exposes row counts only; direction labels remain sealed until "
            "the pre-registered one-time evaluation"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="从真实日线生成公告时间外收益目标")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/market_targets_report.json"),
    )
    parser.add_argument(
        "--announcement-plan",
        type=Path,
        help="Optional train-only plan; limits target creation to its announcement_ids",
    )
    args = parser.parse_args()
    horizons = tuple(sorted({int(value) for value in args.horizons.split(",")}))
    if not horizons or not set(horizons) <= {1, 3, 5}:
        raise ValueError("horizons must contain only 1, 3, 5")
    announcement_ids = None
    announcement_plan_sha256 = None
    if args.announcement_plan:
        plan_path = args.announcement_plan.resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("dataset_role") != "train":
            parser.error("announcement plan must be train only")
        if plan.get("official_test_queried") is not False:
            parser.error("announcement plan must prove official_test_queried=false")
        if plan.get("market_targets_queried") is not False:
            parser.error("announcement plan must prove market_targets_queried=false")
        if plan.get("forward_validation_queried") is not False:
            parser.error("announcement plan must prove forward_validation_queried=false")
        announcement_ids = frozenset(str(value) for value in plan["announcement_ids"])
        announcement_plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    build_targets(
        benchmark_code=args.benchmark,
        horizons=horizons,
        output=args.output,
        announcement_ids=announcement_ids,
        announcement_plan_sha256=announcement_plan_sha256,
    )


if __name__ == "__main__":
    main()

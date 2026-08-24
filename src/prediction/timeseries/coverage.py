"""Read-only train-role coverage audit for daily causal sequences."""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from src.config import PROJECT_ROOT
from src.prediction.timeseries.contract import (
    TimeseriesModelContract,
    load_timeseries_model_contract,
)


def _read_only_connection(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)


def summarize_train_coverage(
    sample_rows: list[tuple[str, str, int]],
    price_rows: list[tuple[str, str]],
    *,
    contract: TimeseriesModelContract,
) -> dict[str, object]:
    """Summarize availability without reading any outcome value or sealed role."""
    dates_by_code: dict[str, list[date]] = defaultdict(list)
    for instrument_code, raw_day in price_rows:
        dates_by_code[str(instrument_code)].append(date.fromisoformat(str(raw_day)))
    for dates in dates_by_code.values():
        dates.sort()
    benchmark = dates_by_code.get("000300.SH", [])
    if not benchmark:
        raise ValueError("CSI300 daily bars are required for coverage audit")

    totals: Counter = Counter()
    halves: dict[str, Counter] = defaultdict(Counter)
    for stock_code, raw_day, horizon in sample_rows:
        published = date.fromisoformat(str(raw_day))
        if not contract.training_start <= published <= contract.training_end:
            raise ValueError(f"non-train sample reached train audit: {published}")
        if int(horizon) not in contract.horizons_sessions:
            raise ValueError(f"unexpected horizon in train audit: {horizon}")
        benchmark_end = bisect.bisect_left(benchmark, published)
        calendar = benchmark[
            max(0, benchmark_end - contract.maximum_sequence_sessions) : benchmark_end
        ]
        stock_dates = dates_by_code.get(stock_code, [])
        valid_stock_sessions = 0
        for trading_day in calendar:
            stock_index = bisect.bisect_left(stock_dates, trading_day)
            valid_stock_sessions += int(
                stock_index < len(stock_dates)
                and stock_dates[stock_index] == trading_day
            )
        minimum_eligible = (
            len(calendar) >= contract.minimum_history_sessions
            and valid_stock_sessions >= contract.minimum_valid_stock_sessions
        )
        full_sequence = (
            len(calendar) >= contract.maximum_sequence_sessions
            and valid_stock_sessions >= contract.maximum_sequence_sessions
        )
        half = f"{published.year}-H{1 if published.month <= 6 else 2}"
        for bucket in (totals, halves[half]):
            bucket["samples"] += 1
            bucket["eligible_minimum"] += int(minimum_eligible)
            bucket["full_sequence"] += int(full_sequence)
            bucket["missing_stock_code"] += int(stock_code not in dates_by_code)

    def serialize(counter: Counter) -> dict[str, object]:
        samples = int(counter["samples"])
        eligible = int(counter["eligible_minimum"])
        full = int(counter["full_sequence"])
        return {
            "samples": samples,
            "eligible_minimum": eligible,
            "eligible_minimum_rate": eligible / samples if samples else 0.0,
            "full_sequence": full,
            "full_sequence_rate": full / samples if samples else 0.0,
            "missing_stock_code": int(counter["missing_stock_code"]),
        }

    return {
        "totals": serialize(totals),
        "by_half_year": {
            half: serialize(values) for half, values in sorted(halves.items())
        },
    }


def audit_train_coverage(
    database_path: Path,
    *,
    contract: TimeseriesModelContract,
) -> dict[str, object]:
    """Read train-role identities and pre-2025 price dates; never select outcomes."""
    with _read_only_connection(database_path) as connection:
        sample_rows = connection.execute(
            """
            SELECT c.stock_code, a.published_date, t.horizon_sessions
            FROM announcement_market_targets AS t
            JOIN announcements AS a ON a.id = t.announcement_id
            JOIN companies AS c ON c.id = a.company_id
            JOIN classifications AS x ON x.announcement_id = a.id
            WHERE t.dataset_role = ? AND x.needs_review = 0
            GROUP BY c.stock_code, a.published_date, t.horizon_sessions
            ORDER BY a.published_date, c.stock_code, t.horizon_sessions
            """,
            ("train",),
        ).fetchall()
        price_exclusive_end = contract.training_end + timedelta(days=1)
        price_rows = connection.execute(
            """
            SELECT instrument_code, trade_date
            FROM daily_prices
            WHERE trade_date < ?
            ORDER BY instrument_code, trade_date
            """,
            (price_exclusive_end.isoformat(),),
        ).fetchall()
    coverage = summarize_train_coverage(
        sample_rows,
        price_rows,
        contract=contract,
    )
    return {
        "audit": "causal-timeseries-daily-v1-train-coverage",
        "created_at": datetime.now(UTC).isoformat(),
        "contract": contract.contract,
        "contract_sha256": contract.source_sha256,
        "dataset_role_read": "train",
        "outcome_values_read": False,
        "official_test_read": False,
        "quarantine_read": False,
        "forward_validation_read": False,
        "sample_unit": "company-publication-day-horizon",
        "minimum_history_sessions": contract.minimum_history_sessions,
        "minimum_valid_stock_sessions": contract.minimum_valid_stock_sessions,
        "maximum_sequence_sessions": contract.maximum_sequence_sessions,
        **coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "ddos.db",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    contract = load_timeseries_model_contract()
    report = audit_train_coverage(args.database, contract=contract)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to(contract.data_directory):
            raise ValueError("coverage output must remain under data/timeseries")
        if output.exists() and not args.force:
            raise FileExistsError(f"coverage report already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

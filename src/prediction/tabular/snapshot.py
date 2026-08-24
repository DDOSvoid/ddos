"""Create a physically train-only SQLite snapshot for tabular development."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from src.prediction.splits import PredictionSplitContract

SNAPSHOT_CONTRACT = "tabular-train-snapshot-v1"

ANNOUNCEMENT_QUERY = """
    SELECT
        a.id,
        a.company_id,
        c.stock_code,
        a.published_date,
        a.title,
        cl.major_category,
        cl.sub_category
    FROM announcements a
    JOIN companies c ON c.id = a.company_id
    JOIN classifications cl ON cl.announcement_id = a.id
    WHERE a.published_date BETWEEN ? AND ?
      AND cl.needs_review = 0
      AND cl.relevance = 'core_event'
    ORDER BY a.published_date, c.stock_code, a.id
"""

TARGET_QUERY = """
    SELECT
        t.announcement_id,
        t.horizon_sessions,
        t.entry_date,
        t.exit_date,
        t.stock_return,
        t.benchmark_return,
        t.excess_return,
        t.actual_direction,
        t.outcome_available_at,
        t.prices_sha256
    FROM announcement_market_targets t
    JOIN announcements a ON a.id = t.announcement_id
    JOIN classifications cl ON cl.announcement_id = a.id
    WHERE t.dataset_role = 'train'
      AND t.split_contract = ?
      AND t.split_source_sha256 = ?
      AND a.published_date BETWEEN ? AND ?
      AND cl.needs_review = 0
      AND cl.relevance = 'core_event'
    ORDER BY t.horizon_sessions, a.published_date, t.announcement_id
"""

DAILY_PRICE_QUERY = """
    SELECT
        p.instrument_code,
        p.instrument_type,
        p.trade_date,
        p.open_price,
        p.high_price,
        p.low_price,
        p.close_price,
        p.pre_close,
        p.pct_change,
        p.volume,
        p.amount,
        p.source
    FROM daily_prices p
    WHERE p.trade_date <= ?
      AND (
          p.instrument_code = '000300.SH'
          OR p.instrument_code IN (
              SELECT DISTINCT c.stock_code
              FROM announcements a
              JOIN companies c ON c.id = a.company_id
              JOIN classifications cl ON cl.announcement_id = a.id
              WHERE a.published_date BETWEEN ? AND ?
                AND cl.needs_review = 0
                AND cl.relevance = 'core_event'
          )
      )
    ORDER BY p.instrument_code, p.trade_date
"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _query_sha256(query: str) -> str:
    return hashlib.sha256(" ".join(query.split()).encode()).hexdigest()


def _copy_query(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    *,
    query: str,
    parameters: tuple,
    insert_sql: str,
) -> int:
    cursor = source.execute(query, parameters)
    count = 0
    while rows := cursor.fetchmany(2000):
        target.executemany(insert_sql, rows)
        count += len(rows)
    return count


def build_train_snapshot(
    *,
    source_path: Path,
    destination_path: Path,
    split: PredictionSplitContract,
) -> dict:
    """Materialize only accepted train events, train labels, and pre-cutoff prices."""
    source_path = source_path.resolve()
    destination_path = destination_path.resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    source_uri = f"file:{source_path.as_posix()}?mode=ro"
    source = sqlite3.connect(source_uri, uri=True, timeout=30)
    target = sqlite3.connect(temporary)
    try:
        source.execute("BEGIN")
        target.executescript(
            """
            PRAGMA journal_mode = DELETE;
            PRAGMA synchronous = FULL;
            CREATE TABLE announcements_train (
                announcement_id INTEGER PRIMARY KEY,
                company_id INTEGER NOT NULL,
                stock_code TEXT NOT NULL,
                published_date TEXT NOT NULL,
                title TEXT NOT NULL,
                major_category TEXT NOT NULL,
                sub_category TEXT NOT NULL
            );
            CREATE TABLE targets_train (
                announcement_id INTEGER NOT NULL,
                horizon_sessions INTEGER NOT NULL,
                entry_date TEXT NOT NULL,
                exit_date TEXT NOT NULL,
                stock_return REAL NOT NULL,
                benchmark_return REAL NOT NULL,
                excess_return REAL NOT NULL,
                actual_direction INTEGER NOT NULL,
                outcome_available_at TEXT NOT NULL,
                prices_sha256 TEXT NOT NULL,
                PRIMARY KEY (announcement_id, horizon_sessions)
            );
            CREATE TABLE daily_prices_train (
                instrument_code TEXT NOT NULL,
                instrument_type TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                pre_close REAL,
                pct_change REAL,
                volume REAL,
                amount REAL,
                source TEXT NOT NULL,
                PRIMARY KEY (instrument_code, trade_date)
            );
            CREATE TABLE snapshot_metadata (
                name TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        date_parameters = (split.train.start.isoformat(), split.train.end.isoformat())
        announcements = _copy_query(
            source,
            target,
            query=ANNOUNCEMENT_QUERY,
            parameters=date_parameters,
            insert_sql="INSERT INTO announcements_train VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        targets = _copy_query(
            source,
            target,
            query=TARGET_QUERY,
            parameters=(
                split.contract,
                split.source_sha256,
                *date_parameters,
            ),
            insert_sql="INSERT INTO targets_train VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        daily_prices = _copy_query(
            source,
            target,
            query=DAILY_PRICE_QUERY,
            parameters=(split.train.end.isoformat(), *date_parameters),
            insert_sql="INSERT INTO daily_prices_train VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        created_at = datetime.now(UTC).isoformat()
        metadata = {
            "contract": SNAPSHOT_CONTRACT,
            "created_at_utc": created_at,
            "train_start": split.train.start.isoformat(),
            "train_end": split.train.end.isoformat(),
            "split_contract": split.contract,
            "split_source_sha256": split.source_sha256,
            "official_test_queried": "false",
            "announcement_query_sha256": _query_sha256(ANNOUNCEMENT_QUERY),
            "target_query_sha256": _query_sha256(TARGET_QUERY),
            "daily_price_query_sha256": _query_sha256(DAILY_PRICE_QUERY),
        }
        target.executemany(
            "INSERT INTO snapshot_metadata(name, value) VALUES (?, ?)",
            sorted(metadata.items()),
        )
        target.executescript(
            """
            CREATE INDEX ix_announcements_train_stock_date
                ON announcements_train(stock_code, published_date);
            CREATE INDEX ix_targets_train_horizon
                ON targets_train(horizon_sessions, announcement_id);
            CREATE INDEX ix_daily_prices_train_code_date
                ON daily_prices_train(instrument_code, trade_date);
            """
        )
        target.commit()
        source.commit()
        integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"train snapshot integrity check failed: {integrity}")
    except Exception:
        target.close()
        source.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        target.close()
        source.close()

    temporary.replace(destination_path)
    report = {
        "contract": SNAPSHOT_CONTRACT,
        "created_at_utc": created_at,
        "source_path": str(source_path),
        "snapshot_path": str(destination_path),
        "snapshot_sha256": _sha256_file(destination_path),
        "split_contract": split.contract,
        "split_source_sha256": split.source_sha256,
        "train_range": {
            "start": split.train.start.isoformat(),
            "end": split.train.end.isoformat(),
        },
        "official_test_queried": False,
        "rows": {
            "announcements_train": announcements,
            "targets_train": targets,
            "daily_prices_train": daily_prices,
        },
        "query_sha256": {
            "announcements": _query_sha256(ANNOUNCEMENT_QUERY),
            "targets": _query_sha256(TARGET_QUERY),
            "daily_prices": _query_sha256(DAILY_PRICE_QUERY),
        },
    }
    return report

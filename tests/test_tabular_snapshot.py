"""Train-only SQLite snapshot tests."""

import sqlite3

from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.snapshot import build_train_snapshot


def _source_database(path, split):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE companies (id INTEGER PRIMARY KEY, stock_code TEXT);
        CREATE TABLE announcements (
            id INTEGER PRIMARY KEY, company_id INTEGER, published_date TEXT, title TEXT
        );
        CREATE TABLE classifications (
            announcement_id INTEGER, major_category TEXT, sub_category TEXT,
            needs_review INTEGER, relevance TEXT
        );
        CREATE TABLE announcement_market_targets (
            announcement_id INTEGER, horizon_sessions INTEGER, entry_date TEXT,
            exit_date TEXT, stock_return REAL, benchmark_return REAL,
            excess_return REAL, actual_direction INTEGER, outcome_available_at TEXT,
            prices_sha256 TEXT, dataset_role TEXT, split_contract TEXT,
            split_source_sha256 TEXT
        );
        CREATE TABLE daily_prices (
            instrument_code TEXT, instrument_type TEXT, trade_date TEXT,
            open_price REAL, high_price REAL, low_price REAL, close_price REAL,
            pre_close REAL, pct_change REAL, volume REAL, amount REAL, source TEXT
        );
        INSERT INTO companies VALUES (1, '300750.SZ');
        INSERT INTO announcements VALUES
            (1, 1, '2024-04-24', 'train event'),
            (2, 1, '2025-02-01', 'sealed event');
        INSERT INTO classifications VALUES
            (1, 'D', 'contract', 0, 'core_event'),
            (2, 'D', 'contract', 0, 'core_event');
        INSERT INTO daily_prices VALUES
            ('300750.SZ', 'stock', '2024-04-23', 10, 11, 9, 10.5, 10, 5, 100, 1000, 'test'),
            ('300750.SZ', 'stock', '2025-01-02', 11, 12, 10, 11.5, 11, 4, 100, 1000, 'test'),
            ('000300.SH', 'index', '2024-04-23', 10, 11, 9, 10.5, 10, 5, 100, 1000, 'test');
        """
    )
    connection.execute(
        "INSERT INTO announcement_market_targets VALUES "
        "(?, 1, '2024-04-25', '2024-04-25', 0.1, 0.01, 0.09, 1, "
        "'2024-04-25 07:00:00', ?, 'train', ?, ?)",
        (1, "a" * 64, split.contract, split.source_sha256),
    )
    connection.execute(
        "INSERT INTO announcement_market_targets VALUES "
        "(?, 1, '2025-02-03', '2025-02-03', 0.1, 0.01, 0.09, 1, "
        "'2025-02-03 07:00:00', ?, 'test', ?, ?)",
        (2, "b" * 64, split.contract, split.source_sha256),
    )
    connection.commit()
    connection.close()


def test_snapshot_physically_excludes_sealed_events_and_post_train_prices(tmp_path):
    split = load_prediction_split_contract()
    source = tmp_path / "source.sqlite"
    output = tmp_path / "train.sqlite"
    _source_database(source, split)
    report = build_train_snapshot(
        source_path=source, destination_path=output, split=split
    )

    connection = sqlite3.connect(output)
    assert connection.execute("SELECT COUNT(*) FROM announcements_train").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM targets_train").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM daily_prices_train").fetchone()[0] == 2
    assert connection.execute(
        "SELECT value FROM snapshot_metadata WHERE name='official_test_queried'"
    ).fetchone()[0] == "false"
    connection.close()
    assert report["official_test_queried"] is False
    assert report["rows"] == {
        "announcements_train": 1,
        "targets_train": 1,
        "daily_prices_train": 2,
    }

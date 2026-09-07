import hashlib
import json
import sqlite3
from datetime import date

import pytest

from src.prediction.data_download import (
    build_download_readiness,
    build_event_train_universe,
    load_data_download_contract,
    load_train_universe_state,
)


def test_event_ranking_download_contract_includes_delisted_stocks(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    contract = load_data_download_contract()
    assert set(contract.list_statuses) == {"L", "D", "P"}
    assert contract.prehistory_start < contract.train_start
    report = build_download_readiness(contract)
    assert report["credentials"]["tushare_token_present"] is False
    assert report["official_test_labels_read"] is False


def test_train_universe_is_bound_to_completed_metadata(tmp_path):
    database = tmp_path / "test.db"
    metadata = tmp_path / "metadata.json"
    output = tmp_path / "universe.json"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE companies (id INTEGER PRIMARY KEY, stock_code TEXT);"
            "CREATE TABLE announcements ("
            "id INTEGER PRIMARY KEY, company_id INTEGER, published_date DATE);"
            "INSERT INTO companies VALUES (1, '000001.SZ'), (2, '600000.SH');"
            "INSERT INTO announcements VALUES "
            "(1, 1, '2023-02-01'), (2, 2, '2025-02-01');"
        )
    metadata.write_text(
        json.dumps(
            {
                "progress": {"completed": 2, "total": 2},
                "finished_at": "2026-09-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    state = build_event_train_universe(
        database_path=database,
        metadata_state_path=metadata,
        output_path=output,
        train_start=date(2023, 1, 1),
        train_end=date(2024, 12, 31),
    )
    assert state["codes"] == ["000001.SZ"]
    assert load_train_universe_state(output) == ["000001.SZ"]
    assert state["official_test_queried"] is False


def test_train_universe_rejects_tampered_code_list(tmp_path):
    path = tmp_path / "universe.json"
    path.write_text(
        json.dumps(
            {
                "selection": {
                    "dataset_role": "train",
                    "codes": 1,
                    "codes_sha256": hashlib.sha256(b"000001.SZ").hexdigest(),
                },
                "codes": ["600000.SH"],
                "official_test_queried": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash"):
        load_train_universe_state(path)

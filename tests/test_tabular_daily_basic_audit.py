"""End-to-end audit tests for train-only daily_basic partitions."""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime

import pandas as pd

from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.contract import load_tabular_model_contract
from src.prediction.tabular.daily_basic import (
    DAILY_BASIC_CONTRACT,
    DAILY_BASIC_FIELDS,
    normalize_daily_basic_frame,
)
from src.prediction.tabular.daily_basic_audit import audit_daily_basic_dataset
from src.prediction.tabular.snapshot import SNAPSHOT_CONTRACT


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(tmp_path):
    split = load_prediction_split_contract()
    tabular = load_tabular_model_contract(split=split)
    root = tmp_path / "daily_basic"
    relative = "raw/tushare_daily_basic/300750.SZ.parquet"
    path = root / relative
    path.parent.mkdir(parents=True)
    raw = pd.DataFrame(
        {
            **{name: [1.0] for name in DAILY_BASIC_FIELDS[2:]},
            "ts_code": ["300750.SZ"],
            "trade_date": ["20240424"],
            "close": [190.0],
            "total_share": [240000.0],
            "float_share": [220000.0],
            "free_share": [200000.0],
            "total_mv": [45600000.0],
            "circ_mv": [40000000.0],
        }
    )
    normalized = normalize_daily_basic_frame(
        raw,
        stock_code="300750.SZ",
        start=split.train.start,
        end=split.train.end,
        fetched_at=datetime(2026, 8, 24, tzinfo=UTC),
    )
    normalized.to_parquet(path, index=False)
    codes = ["300750.SZ", "688999.SH"]
    codes_sha256 = hashlib.sha256("\n".join(codes).encode()).hexdigest()
    state = {
        "contract": DAILY_BASIC_CONTRACT,
        "tabular_contract_sha256": tabular.source_sha256,
        "split_contract": split.contract,
        "split_source_sha256": split.source_sha256,
        "official_test_queried": False,
        "selection": {
            "dataset_role": "train",
            "start": split.train.start.isoformat(),
            "end": split.train.end.isoformat(),
            "source": "tushare_daily_basic",
            "codes": 2,
            "codes_sha256": codes_sha256,
            "fields": list(DAILY_BASIC_FIELDS),
            "availability_rule": "trade_date 18:00 Asia/Shanghai",
        },
        "completed": {
            "300750.SZ:2023-01-01:2024-12-31": {
                "status": "downloaded",
                "rows": 1,
                "path": relative,
                "sha256": _sha256(path),
            },
            "688999.SH:2023-01-01:2024-12-31": {
                "status": "empty",
                "rows": 0,
                "path": None,
                "sha256": None,
            },
        },
        "failures": {},
    }
    state_path = root / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    snapshot = tmp_path / "snapshot.sqlite"
    connection = sqlite3.connect(snapshot)
    connection.execute(
        "CREATE TABLE daily_prices_train "
        "(instrument_code TEXT, instrument_type TEXT, trade_date TEXT, close_price REAL)"
    )
    connection.execute(
        "INSERT INTO daily_prices_train VALUES (?, ?, ?, ?)",
        ("300750.SZ", "stock", "2024-04-24", 190.0),
    )
    connection.commit()
    connection.close()
    snapshot_manifest = tmp_path / "snapshot.manifest.json"
    snapshot_manifest.write_text(
        json.dumps(
            {
                "contract": SNAPSHOT_CONTRACT,
                "official_test_queried": False,
                "split_contract": split.contract,
                "split_source_sha256": split.source_sha256,
                "train_range": {
                    "start": split.train.start.isoformat(),
                    "end": split.train.end.isoformat(),
                },
                "snapshot_sha256": _sha256(snapshot),
            }
        ),
        encoding="utf-8",
    )
    return root, state_path, snapshot, snapshot_manifest, tabular, split, path


def test_daily_basic_audit_accepts_hashes_times_and_empty_unlisted_code(tmp_path):
    root, state, snapshot, manifest, tabular, split, _ = _dataset(tmp_path)
    report = audit_daily_basic_dataset(
        root=root,
        state_path=state,
        snapshot_path=snapshot,
        snapshot_manifest_path=manifest,
        tabular=tabular,
        split=split,
    )
    assert report["passed"] is True
    assert report["stats"]["downloaded_partitions"] == 1
    assert report["stats"]["empty_partitions"] == 1
    assert report["stats"]["close_mismatches"] == 0
    assert report["snapshot_daily_basic_coverage"] == 1.0


def test_daily_basic_audit_rejects_changed_file_even_if_state_is_unchanged(tmp_path):
    root, state, snapshot, manifest, tabular, split, path = _dataset(tmp_path)
    changed = pd.read_parquet(path)
    changed.loc[0, "pb"] = 999.0
    changed.to_parquet(path, index=False)
    report = audit_daily_basic_dataset(
        root=root,
        state_path=state,
        snapshot_path=snapshot,
        snapshot_manifest_path=manifest,
        tabular=tabular,
        split=split,
    )
    assert report["passed"] is False
    assert any("SHA-256" in error for error in report["errors"])

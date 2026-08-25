"""Full integrity and train-boundary audit for Tushare daily_basic partitions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.prediction.splits import PredictionSplitContract
from src.prediction.tabular.contract import TabularModelContract
from src.prediction.tabular.daily_basic import DAILY_BASIC_CONTRACT, DAILY_BASIC_FIELDS
from src.prediction.tabular.snapshot import SNAPSHOT_CONTRACT

DAILY_BASIC_AUDIT_CONTRACT = "tabular-daily-basic-audit-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_inside(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"daily_basic path escapes dataset root: {relative}") from exc
    return path


def _snapshot_prices(
    *, snapshot_path: Path, snapshot_manifest_path: Path, split: PredictionSplitContract
) -> tuple[pd.DataFrame, str, str]:
    manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != SNAPSHOT_CONTRACT:
        raise ValueError("unexpected train snapshot contract")
    if manifest.get("official_test_queried") is not False:
        raise PermissionError("train snapshot does not prove official-test isolation")
    if manifest.get("split_contract") != split.contract:
        raise ValueError("train snapshot split contract mismatch")
    if manifest.get("train_range") != {
        "start": split.train.start.isoformat(),
        "end": split.train.end.isoformat(),
    }:
        raise ValueError("train snapshot semantic range mismatch")
    snapshot_sha256 = _sha256_file(snapshot_path)
    if snapshot_sha256 != manifest.get("snapshot_sha256"):
        raise ValueError("train snapshot file hash mismatch")
    connection = sqlite3.connect(
        f"file:{snapshot_path.resolve().as_posix()}?mode=ro", uri=True, timeout=30
    )
    try:
        prices = pd.read_sql_query(
            """
            SELECT instrument_code, trade_date, close_price
            FROM daily_prices_train
            WHERE instrument_type = 'stock'
            """,
            connection,
        )
    finally:
        connection.close()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"], errors="raise").dt.date
    return prices, snapshot_sha256, str(manifest.get("split_source_sha256", ""))


def audit_daily_basic_dataset(
    *,
    root: Path,
    state_path: Path,
    snapshot_path: Path,
    snapshot_manifest_path: Path,
    tabular: TabularModelContract,
    split: PredictionSplitContract,
) -> dict:
    """Re-read every partition and compare all available closes to train prices."""
    state = json.loads(state_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    selection = state.get("selection", {})
    if state.get("contract") != DAILY_BASIC_CONTRACT:
        errors.append("unexpected daily_basic state contract")
    if state.get("tabular_contract_sha256") != tabular.source_sha256:
        errors.append("daily_basic tabular contract hash mismatch")
    if state.get("split_contract") != split.contract:
        errors.append("daily_basic split contract name mismatch")
    if state.get("official_test_queried") is not False:
        errors.append("daily_basic state does not prove official-test isolation")
    expected_selection = {
        "dataset_role": "train",
        "start": split.train.start.isoformat(),
        "end": split.train.end.isoformat(),
        "source": "tushare_daily_basic",
        "fields": list(DAILY_BASIC_FIELDS),
        "availability_rule": "trade_date 18:00 Asia/Shanghai",
    }
    for name, expected in expected_selection.items():
        if selection.get(name) != expected:
            errors.append(f"daily_basic selection {name} does not match contract")
    completed = state.get("completed", {})
    if state.get("failures"):
        errors.append("daily_basic state contains failures")
    codes = sorted(str(key).split(":", 1)[0] for key in completed)
    if len(codes) != len(set(codes)) or len(codes) != selection.get("codes"):
        errors.append("daily_basic completed stock universe is invalid")
    codes_sha256 = hashlib.sha256("\n".join(sorted(set(codes))).encode()).hexdigest()
    if codes_sha256 != selection.get("codes_sha256"):
        errors.append("daily_basic stock universe hash mismatch")

    try:
        snapshot_prices, snapshot_sha256, snapshot_split_sha256 = _snapshot_prices(
            snapshot_path=snapshot_path,
            snapshot_manifest_path=snapshot_manifest_path,
            split=split,
        )
    except Exception as exc:
        errors.append(f"snapshot audit failed: {type(exc).__name__}: {exc}")
        snapshot_prices = pd.DataFrame(
            columns=["instrument_code", "trade_date", "close_price"]
        )
        snapshot_sha256 = ""
        snapshot_split_sha256 = ""

    artifact_split_sha256 = str(state.get("split_source_sha256", ""))
    if artifact_split_sha256 != snapshot_split_sha256:
        errors.append("daily_basic and train snapshot split hashes differ")
    split_hash_status = (
        "exact"
        if artifact_split_sha256 == split.source_sha256
        else "semantic_requalification"
    )

    stats = {
        "requested_stocks": selection.get("codes", 0),
        "downloaded_partitions": 0,
        "empty_partitions": 0,
        "raw_rows": 0,
        "snapshot_price_rows_for_universe": 0,
        "snapshot_rows_covered": 0,
        "snapshot_rows_without_daily_basic": 0,
        "close_mismatches": 0,
    }
    frames: list[pd.DataFrame] = []
    empty_codes: list[str] = []
    for key, item in sorted(completed.items()):
        stock_code = str(key).split(":", 1)[0]
        status = item.get("status")
        if status == "empty":
            stats["empty_partitions"] += 1
            empty_codes.append(stock_code)
            if item.get("rows") != 0 or item.get("path") is not None:
                errors.append(f"{key}: invalid empty partition record")
            continue
        if status != "downloaded":
            errors.append(f"{key}: unexpected partition status {status}")
            continue
        stats["downloaded_partitions"] += 1
        try:
            path = _resolve_inside(root, item["path"])
            if _sha256_file(path) != item.get("sha256"):
                raise ValueError("file SHA-256 mismatch")
            frame = pd.read_parquet(path)
            if len(frame) != item.get("rows"):
                raise ValueError("row count mismatch")
            required = {
                *DAILY_BASIC_FIELDS,
                "available_at_utc",
                "source",
                "fetched_at_utc",
                "unit_market_value",
                "unit_share",
            }
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"missing columns: {sorted(missing)}")
            if set(frame["ts_code"].astype(str).unique()) != {stock_code}:
                raise ValueError("partition contains another stock code")
            frame["trade_date"] = pd.to_datetime(
                frame["trade_date"], errors="raise"
            ).dt.date
            if frame["trade_date"].duplicated().any():
                raise ValueError("duplicate trade date")
            if frame["trade_date"].min() < split.train.start or frame[
                "trade_date"
            ].max() > split.train.end:
                raise ValueError("partition leaves train boundary")
            expected_available = (
                pd.to_datetime(frame["trade_date"].astype(str) + " 18:00:00")
                .dt.tz_localize("Asia/Shanghai")
                .dt.tz_convert("UTC")
            )
            actual_available = pd.to_datetime(
                frame["available_at_utc"], errors="raise", utc=True
            )
            if not actual_available.equals(expected_available):
                raise ValueError("availability timestamp mismatch")
            if set(frame["source"].astype(str).unique()) != {"tushare_daily_basic"}:
                raise ValueError("source marker mismatch")
            if set(frame["unit_market_value"].astype(str).unique()) != {"CNY_10K"}:
                raise ValueError("market-value unit mismatch")
            if set(frame["unit_share"].astype(str).unique()) != {"SHARES_10K"}:
                raise ValueError("share unit mismatch")
            for column in ("close", "total_share", "total_mv", "circ_mv"):
                values = pd.to_numeric(frame[column], errors="coerce")
                if values.isna().any() or (values <= 0).any():
                    raise ValueError(f"invalid required numeric column {column}")
            stats["raw_rows"] += len(frame)
            frames.append(frame)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty and combined.duplicated(["ts_code", "trade_date"]).any():
        errors.append("duplicate stock-date across daily_basic partitions")
    selected_prices = snapshot_prices.loc[
        snapshot_prices["instrument_code"].isin(set(codes))
    ].copy()
    stats["snapshot_price_rows_for_universe"] = len(selected_prices)
    if not combined.empty:
        comparison = selected_prices.merge(
            combined.loc[:, ["ts_code", "trade_date", "close"]],
            left_on=["instrument_code", "trade_date"],
            right_on=["ts_code", "trade_date"],
            how="left",
            validate="one_to_one",
        )
        covered = comparison["close"].notna()
        stats["snapshot_rows_covered"] = int(covered.sum())
        stats["snapshot_rows_without_daily_basic"] = int((~covered).sum())
        close_diff = (
            pd.to_numeric(comparison.loc[covered, "close"], errors="coerce")
            - pd.to_numeric(
                comparison.loc[covered, "close_price"], errors="coerce"
            )
        ).abs()
        stats["close_mismatches"] = int((close_diff > 1e-10).sum())
        if stats["close_mismatches"]:
            errors.append("daily_basic close differs from train daily-price close")
        extra = combined.merge(
            selected_prices,
            left_on=["ts_code", "trade_date"],
            right_on=["instrument_code", "trade_date"],
            how="left",
            indicator=True,
        )
        if (extra["_merge"] != "both").any():
            errors.append("daily_basic contains stock-date rows absent from train snapshot")

    empty_with_prices = sorted(
        set(empty_codes) & set(selected_prices["instrument_code"].astype(str))
    )
    if empty_with_prices:
        errors.append(
            "empty daily_basic partitions have train-period price rows: "
            + ",".join(empty_with_prices)
        )
    report = {
        "contract": DAILY_BASIC_AUDIT_CONTRACT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "passed": not errors,
        "dataset_role": "train",
        "official_test_queried": False,
        "train_range": {
            "start": split.train.start.isoformat(),
            "end": split.train.end.isoformat(),
        },
        "tabular_contract_sha256": tabular.source_sha256,
        "split_source_sha256": split.source_sha256,
        "artifact_split_source_sha256": artifact_split_sha256,
        "split_hash_status": split_hash_status,
        "split_semantic_requalification": {
            "applied": split_hash_status == "semantic_requalification",
            "reason": (
                "Legacy train artifacts share one split hash and contract name; their "
                "recorded train range exactly matches the current parsed contract. No "
                "role boundary was changed and no sealed row was read."
            ),
        },
        "state_sha256": _sha256_file(state_path),
        "snapshot_sha256": snapshot_sha256,
        "stats": stats,
        "empty_stock_codes": sorted(empty_codes),
        "empty_partition_explanation": (
            "Every empty code has zero train-period daily-price rows in the audited "
            "train-only snapshot; it therefore had no eligible stock-day input."
        ),
        "errors": errors,
    }
    if stats["snapshot_price_rows_for_universe"]:
        report["snapshot_daily_basic_coverage"] = float(
            stats["snapshot_rows_covered"]
            / stats["snapshot_price_rows_for_universe"]
        )
    else:
        report["snapshot_daily_basic_coverage"] = 0.0
    return report

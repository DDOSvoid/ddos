"""Auditable integrity checks for the train-only intraday feature dataset."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.splits import PredictionSplitContract
from src.prediction.tabular.aggregates import (
    AGGREGATION_VERSION,
    EXPECTED_BAR_LABELS,
    INTRADAY_FEATURE_COLUMNS,
    aggregate_intraday_days,
    intraday_feature_schema_sha256,
)
from src.prediction.tabular.contract import TabularModelContract
from src.prediction.tabular.intraday import yearly_chunks

AUDIT_CONTRACT = "tabular-intraday-audit-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_manifest_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"manifest path escapes dataset root: {relative}") from exc
    return path


def _expected_keys(raw_state: dict, errors: list[str]) -> set[str]:
    selection = raw_state.get("selection", {})
    completed = raw_state.get("completed", {})
    codes = sorted({str(key).split(":", 1)[0] for key in completed})
    canonical_codes = "\n".join(codes)
    if len(codes) != selection.get("codes"):
        errors.append("manifest stock count does not match selection")
    if hashlib.sha256(canonical_codes.encode()).hexdigest() != selection.get(
        "codes_sha256"
    ):
        errors.append("manifest stock universe hash does not match selection")

    try:
        start = datetime.fromisoformat(selection["start"]).date()
        end = datetime.fromisoformat(selection["end"]).date()
    except (KeyError, TypeError, ValueError):
        errors.append("manifest selection contains invalid dates")
        return set()
    return {
        chunk.key
        for code in codes
        for chunk in yearly_chunks(code, start=start, end=end)
    }


def audit_intraday_dataset(
    *,
    raw_root: Path,
    raw_state_path: Path,
    aggregate_root: Path,
    aggregate_state_path: Path,
    tabular: TabularModelContract,
    split: PredictionSplitContract,
) -> dict:
    """Recompute and compare every completed train aggregate against raw bars."""
    raw_state = json.loads(raw_state_path.read_text(encoding="utf-8"))
    aggregate_state = json.loads(aggregate_state_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    raw_selection = raw_state.get("selection", {})

    expected_raw_contract = {
        "dataset_role": "train",
        "start": split.train.start.isoformat(),
        "end": split.train.end.isoformat(),
        "frequency": tabular.raw_intraday_frequency,
        "usage": tabular.raw_intraday_usage,
        "source": "amazingdata_query_kline",
        "bar_timestamp_semantics": "bar_start",
    }
    for name, expected in expected_raw_contract.items():
        if raw_selection.get(name) != expected:
            errors.append(f"raw selection {name} does not match contract")
    if raw_state.get("official_test_queried") is not False:
        errors.append("raw state does not prove official-test isolation")
    if aggregate_state.get("official_test_queried") is not False:
        errors.append("aggregate state does not prove official-test isolation")
    if raw_state.get("tabular_contract_sha256") != tabular.source_sha256:
        errors.append("raw state tabular contract hash mismatch")
    if aggregate_state.get("tabular_contract_sha256") != tabular.source_sha256:
        errors.append("aggregate state tabular contract hash mismatch")
    if raw_state.get("split_source_sha256") != split.source_sha256:
        errors.append("raw state split contract hash mismatch")
    if aggregate_state.get("split_source_sha256") != split.source_sha256:
        errors.append("aggregate state split contract hash mismatch")

    raw_completed = raw_state.get("completed", {})
    aggregate_completed = aggregate_state.get("completed", {})
    expected_keys = _expected_keys(raw_state, errors)
    if set(raw_completed) != expected_keys:
        errors.append("raw manifest is incomplete or contains unexpected chunks")
    if set(aggregate_completed) != expected_keys:
        errors.append("aggregate manifest is incomplete or contains unexpected chunks")
    if raw_state.get("failures"):
        errors.append("raw manifest contains failures")
    if aggregate_state.get("failures"):
        errors.append("aggregate manifest contains failures")

    aggregate_selection = aggregate_state.get("selection", {})
    if aggregate_selection.get("aggregation_version") != AGGREGATION_VERSION:
        errors.append("aggregate version mismatch")
    if aggregate_selection.get(
        "feature_schema_sha256"
    ) != intraday_feature_schema_sha256():
        errors.append("aggregate feature schema hash mismatch")
    if aggregate_selection.get("raw_codes_sha256") != raw_selection.get(
        "codes_sha256"
    ):
        errors.append("raw and aggregate stock universe hashes differ")

    stats = {
        "stocks": raw_selection.get("codes", 0),
        "expected_chunks": len(expected_keys),
        "downloaded_chunks": 0,
        "empty_chunks": 0,
        "raw_rows": 0,
        "aggregate_rows": 0,
    }
    all_stock_dates: set[tuple[str, str]] = set()
    for key in sorted(expected_keys):
        raw_item = raw_completed.get(key)
        aggregate_item = aggregate_completed.get(key)
        if raw_item is None or aggregate_item is None:
            continue
        if raw_item.get("status") == "empty":
            stats["empty_chunks"] += 1
            if raw_item.get("rows") != 0 or raw_item.get("path") is not None:
                errors.append(f"{key}: invalid empty raw record")
            if aggregate_item.get("status") != "empty_source":
                errors.append(f"{key}: empty raw chunk has non-empty aggregate")
            continue
        if raw_item.get("status") != "downloaded":
            errors.append(f"{key}: unexpected raw status")
            continue

        stats["downloaded_chunks"] += 1
        try:
            raw_path = _resolve_manifest_path(raw_root, raw_item["path"])
            raw_sha256 = _sha256_file(raw_path)
            if raw_sha256 != raw_item.get("sha256"):
                raise ValueError("raw file SHA-256 mismatch")
            raw_frame = pd.read_parquet(raw_path)
            if len(raw_frame) != raw_item.get("rows"):
                raise ValueError("raw row count mismatch")
            stats["raw_rows"] += len(raw_frame)

            day_labels = raw_frame.assign(
                bar_label=pd.to_datetime(raw_frame["trade_time"]).dt.strftime("%H:%M")
            ).groupby("trade_date", sort=False)["bar_label"].agg(tuple)
            if not day_labels.map(lambda value: value == EXPECTED_BAR_LABELS).all():
                raise ValueError("raw file contains an incomplete trading day")

            expected_aggregate = aggregate_intraday_days(
                raw_frame,
                source_file_sha256=raw_sha256,
                tabular=tabular,
                split=split,
            )
            if aggregate_item.get("status") != "aggregated":
                raise ValueError("downloaded raw chunk has non-aggregated output")
            if aggregate_item.get("source_file_sha256") != raw_sha256:
                raise ValueError("aggregate provenance hash mismatch")
            aggregate_path = _resolve_manifest_path(
                aggregate_root, aggregate_item["path"]
            )
            if _sha256_file(aggregate_path) != aggregate_item.get("sha256"):
                raise ValueError("aggregate file SHA-256 mismatch")
            actual_aggregate = pd.read_parquet(aggregate_path)
            if len(actual_aggregate) != aggregate_item.get("rows"):
                raise ValueError("aggregate row count mismatch")
            pd.testing.assert_frame_equal(
                actual_aggregate.reset_index(drop=True),
                expected_aggregate.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
            feature_values = actual_aggregate.loc[:, INTRADAY_FEATURE_COLUMNS].to_numpy(
                dtype=float
            )
            if not np.isfinite(feature_values).all():
                raise ValueError("aggregate contains non-finite feature values")
            for row in actual_aggregate.loc[:, ["stock_code", "observation_date"]].itertuples(
                index=False
            ):
                identity = (str(row.stock_code), str(row.observation_date))
                if identity in all_stock_dates:
                    raise ValueError("duplicate stock-date across aggregate files")
                all_stock_dates.add(identity)
            stats["aggregate_rows"] += len(actual_aggregate)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")

    report = {
        "contract": AUDIT_CONTRACT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "passed": not errors,
        "official_test_queried": False,
        "train_range": {
            "start": split.train.start.isoformat(),
            "end": split.train.end.isoformat(),
        },
        "tabular_contract_sha256": tabular.source_sha256,
        "split_source_sha256": split.source_sha256,
        "feature_schema_sha256": intraday_feature_schema_sha256(),
        "stats": stats,
        "errors": errors,
    }
    return report

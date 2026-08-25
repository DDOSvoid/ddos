"""Build label-free, causal direct-sequence artifacts for LSTM development."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.prediction.development_contract import CausalDevelopmentContract
from src.prediction.development_contract import load_causal_development_contract
from src.prediction.timeseries.alignment import (
    next_trading_preopen_prediction_as_of,
    require_daily_v1_bar_usable,
)
from src.prediction.timeseries.contract import TimeseriesModelContract
from src.prediction.timeseries.contract import load_timeseries_model_contract
from src.prediction.timeseries.coverage import _read_only_connection
from src.prediction.timeseries.features import (
    PRICE_COLUMNS,
    build_daily_sequence_panel,
)
from src.prediction.timeseries.lstm_contract import (
    LstmTimeseriesContract,
    load_lstm_timeseries_contract,
)
from src.prediction.timeseries.sequence_identity import canonical_company_day_id

IDENTITY_COLUMNS = (
    "company_day_id",
    "stock_code",
    "published_date",
    "dataset_role",
)


@dataclass(frozen=True)
class SequenceArtifactBundle:
    sequences: np.ndarray
    time_masks: np.ndarray
    metadata: pd.DataFrame


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_company_days(company_days: pd.DataFrame) -> pd.DataFrame:
    missing = set(IDENTITY_COLUMNS) - set(company_days.columns)
    if missing:
        raise ValueError(f"company-day identities missing columns: {sorted(missing)}")
    working = company_days.loc[:, IDENTITY_COLUMNS].copy()
    roles = {str(value) for value in working["dataset_role"].dropna().unique()}
    if roles != {"train"}:
        raise PermissionError(
            "LSTM sequence artifacts accept train identities only; "
            f"found {sorted(roles)}"
        )
    working["stock_code"] = working["stock_code"].astype(str).str.strip().str.upper()
    working["published_date"] = pd.to_datetime(
        working["published_date"], errors="raise"
    ).dt.date
    if working.duplicated(["stock_code", "published_date"]).any():
        raise ValueError("duplicate stock/publication date in company-day identities")
    expected = [
        canonical_company_day_id(row.stock_code, row.published_date)
        for row in working.itertuples(index=False)
    ]
    if expected != working["company_day_id"].astype(str).tolist():
        raise ValueError("company_day_id does not match the canonical algorithm")
    return working.sort_values(
        ["published_date", "stock_code"], kind="mergesort"
    ).reset_index(drop=True)


def build_direct_sequence_bundle(
    company_days: pd.DataFrame,
    *,
    stock_bars_by_code: dict[str, pd.DataFrame],
    benchmark_bars: pd.DataFrame,
    parent_contract: TimeseriesModelContract,
    contract: LstmTimeseriesContract,
    development: CausalDevelopmentContract,
) -> SequenceArtifactBundle:
    """Materialize only pre-publication sequences; never accept outcome columns."""
    forbidden = [
        name for name in company_days.columns if not contract.feature_name_allowed(name)
    ]
    if forbidden:
        raise ValueError(f"outcome columns rejected from sequence identities: {forbidden}")
    identities = _validate_company_days(company_days)
    benchmark_dates = pd.to_datetime(
        benchmark_bars["trade_date"], errors="raise"
    ).dt.date.tolist()
    maximum = contract.maximum_sequence_sessions
    feature_count = len(contract.sequence_columns)
    sequence_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    panels: dict[str, pd.DataFrame] = {}
    panel_dates: dict[str, list[date]] = {}

    for identity in identities.itertuples(index=False):
        metadata: dict[str, object] = {
            "company_day_id": identity.company_day_id,
            "stock_code": identity.stock_code,
            "published_date": identity.published_date,
            "dataset_role": "train",
            "prediction_as_of": None,
            "sequence_available": False,
            "unavailable_reason": None,
            "sequence_row": None,
            "history_sessions": 0,
            "valid_stock_sessions": 0,
            "last_bar_trade_date": None,
            "last_bar_available_at": None,
            "data_as_of": None,
            "evidence_sha256": None,
        }
        try:
            metadata["prediction_as_of"] = next_trading_preopen_prediction_as_of(
                identity.published_date,
                trading_days=benchmark_dates,
                development=development,
            )
        except ValueError as error:
            metadata["unavailable_reason"] = str(error)
            metadata_rows.append(metadata)
            continue
        stock_bars = stock_bars_by_code.get(identity.stock_code)
        if stock_bars is None or stock_bars.empty:
            metadata["unavailable_reason"] = "missing_stock_bars"
            metadata_rows.append(metadata)
            continue
        try:
            if identity.stock_code not in panels:
                panels[identity.stock_code] = build_daily_sequence_panel(
                    stock_bars=stock_bars,
                    benchmark_bars=benchmark_bars,
                    contract=parent_contract,
                    development=development,
                )
                panel_dates[identity.stock_code] = panels[
                    identity.stock_code
                ]["trade_date"].tolist()
            panel = panels[identity.stock_code]
            dates = panel_dates[identity.stock_code]
            endpoint = bisect.bisect_left(dates, identity.published_date)
            start = max(0, endpoint - maximum)
            frame = panel.iloc[start:endpoint].copy().reset_index(drop=True)
            if len(frame) < parent_contract.minimum_history_sessions:
                raise ValueError(
                    "insufficient sequence history before prediction: "
                    f"{len(frame)} < {parent_contract.minimum_history_sessions}"
                )
            valid_stock_sessions = int(frame["stock_tradable_mask"].sum())
            if valid_stock_sessions < contract.minimum_valid_stock_sessions:
                raise ValueError(
                    "insufficient valid stock sessions before prediction: "
                    f"{valid_stock_sessions} < {contract.minimum_valid_stock_sessions}"
                )
            last = frame.iloc[-1]
            require_daily_v1_bar_usable(
                trade_date=last["trade_date"],
                published_date=identity.published_date,
                prediction_as_of=metadata["prediction_as_of"],
                bar_available_at=last["bar_available_at"],
                development=development,
            )
        except ValueError as error:
            metadata["unavailable_reason"] = str(error)
            metadata_rows.append(metadata)
            continue

        values = frame.loc[:, contract.sequence_columns].to_numpy(dtype=np.float32)
        if len(values) > maximum:
            raise ValueError("direct sequence exceeds the registered maximum lookback")
        padded = np.full((maximum, feature_count), np.nan, dtype=np.float32)
        time_mask = np.zeros(maximum, dtype=bool)
        padded[: len(values)] = values
        time_mask[: len(values)] = True
        sequence_row = len(sequence_rows)
        sequence_rows.append(padded)
        mask_rows.append(time_mask)
        metadata.update(
            {
                "sequence_available": True,
                "sequence_row": sequence_row,
                "history_sessions": int(len(frame)),
                "valid_stock_sessions": int(frame["stock_tradable_mask"].sum()),
                "last_bar_trade_date": last["trade_date"],
                "last_bar_available_at": last["bar_available_at"],
                "data_as_of": last["bar_available_at"],
                "evidence_sha256": _frame_sha256(frame),
            }
        )
        metadata_rows.append(metadata)

    sequences = (
        np.stack(sequence_rows)
        if sequence_rows
        else np.empty((0, maximum, feature_count), dtype=np.float32)
    )
    time_masks = (
        np.stack(mask_rows)
        if mask_rows
        else np.empty((0, maximum), dtype=bool)
    )
    metadata = pd.DataFrame.from_records(metadata_rows)
    return SequenceArtifactBundle(
        sequences=sequences,
        time_masks=time_masks,
        metadata=metadata,
    )


def _write_npy_atomic(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, array, allow_pickle=False)
    temporary.replace(path)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_direct_sequence_bundle(
    bundle: SequenceArtifactBundle,
    output_directory: Path,
    *,
    contract: LstmTimeseriesContract,
    identity_source_sha256: str,
    price_source_sha256: str,
    source_audit: dict[str, object] | None = None,
    force: bool = False,
) -> dict[str, object]:
    output = output_directory.resolve()
    if not output.is_relative_to(contract.data_directory):
        raise ValueError("LSTM artifacts must remain under the registered data boundary")
    sequence_path = output / "sequences" / "train_sequences.npy"
    mask_path = output / "sequences" / "train_time_masks.npy"
    metadata_path = output / "sequences" / "train_metadata.parquet"
    manifest_path = output / "manifests" / "train_manifest.json"
    targets = (sequence_path, mask_path, metadata_path, manifest_path)
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(f"LSTM sequence artifact already exists: {existing[0]}")

    _write_npy_atomic(bundle.sequences, sequence_path)
    _write_npy_atomic(bundle.time_masks, mask_path)
    _write_parquet_atomic(bundle.metadata, metadata_path)
    available = bundle.metadata["sequence_available"].astype(bool)
    manifest: dict[str, object] = {
        "artifact": "causal-timeseries-lstm-daily-v1-train-sequences",
        "created_at": datetime.now(UTC).isoformat(),
        "contract": contract.contract,
        "contract_sha256": contract.source_sha256,
        "parent_contract": contract.parent_contract,
        "parent_contract_sha256": contract.parent_contract_sha256,
        "split_contract": contract.split_contract,
        "split_contract_sha256": contract.split_contract_sha256,
        "development_contract": contract.development_contract,
        "development_contract_sha256": contract.development_contract_sha256,
        "dataset_role_read": "train",
        "outcome_values_read": False,
        "official_test_read": False,
        "quarantine_read": False,
        "forward_validation_read": False,
        "company_days": int(len(bundle.metadata)),
        "available_sequences": int(available.sum()),
        "unavailable_sequences": int((~available).sum()),
        "sequence_shape": list(bundle.sequences.shape),
        "sequence_columns": list(contract.sequence_columns),
        "identity_source_sha256": identity_source_sha256,
        "price_source_sha256": price_source_sha256,
        "source_audit": source_audit or {},
        "sequences_file": sequence_path.relative_to(output).as_posix(),
        "sequences_sha256": _file_sha256(sequence_path),
        "time_masks_file": mask_path.relative_to(output).as_posix(),
        "time_masks_sha256": _file_sha256(mask_path),
        "metadata_file": metadata_path.relative_to(output).as_posix(),
        "metadata_sha256": _file_sha256(metadata_path),
    }
    _write_json_atomic(manifest, manifest_path)
    return manifest


def _stable_rows_sha256(rows: list[tuple]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def load_label_free_train_sources(
    connection: sqlite3.Connection,
    *,
    contract: LstmTimeseriesContract,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, dict[str, object]]:
    """Read accepted train identities and prices without touching target tables."""
    announcement_rows = connection.execute(
        """
        SELECT
            a.id,
            c.stock_code,
            a.published_date
        FROM announcements AS a
        JOIN companies AS c ON c.id = a.company_id
        JOIN classifications AS cl ON cl.announcement_id = a.id
        WHERE a.published_date BETWEEN ? AND ?
          AND cl.needs_review = 0
          AND cl.relevance = 'core_event'
        ORDER BY a.published_date, c.stock_code, a.id
        """,
        (contract.training_start.isoformat(), contract.training_end.isoformat()),
    ).fetchall()
    if not announcement_rows:
        raise ValueError("no accepted train core-event announcements")
    identity_records = []
    for stock_code, published_date in sorted(
        {(str(row[1]), date.fromisoformat(str(row[2]))) for row in announcement_rows},
        key=lambda item: (item[1], item[0]),
    ):
        identity_records.append(
            {
                "company_day_id": canonical_company_day_id(
                    stock_code, published_date
                ),
                "stock_code": stock_code,
                "published_date": published_date,
                "dataset_role": "train",
            }
        )
    company_days = pd.DataFrame.from_records(identity_records)
    stock_codes = sorted(set(company_days["stock_code"].astype(str)))
    instrument_codes = ["000300.SH", *stock_codes]
    placeholders = ",".join("?" for _ in instrument_codes)
    price_rows = connection.execute(
        f"""
        SELECT
            instrument_code,
            trade_date,
            open_price,
            high_price,
            low_price,
            close_price,
            pre_close,
            volume,
            amount
        FROM daily_prices
        WHERE instrument_code IN ({placeholders})
          AND trade_date <= ?
        ORDER BY instrument_code, trade_date
        """,
        [*instrument_codes, contract.training_end.isoformat()],
    ).fetchall()
    grouped: dict[str, list[tuple]] = defaultdict(list)
    for row in price_rows:
        grouped[str(row[0])].append(tuple(row[1:]))
    price_frames = {
        code: pd.DataFrame(rows, columns=PRICE_COLUMNS)
        for code, rows in grouped.items()
    }
    benchmark = price_frames.pop("000300.SH", None)
    if benchmark is None or benchmark.empty:
        raise ValueError("CSI300 calendar is unavailable for LSTM sequence artifacts")
    minimum_dates = [
        date.fromisoformat(str(row[1])) for row in price_rows if row[1] is not None
    ]
    source_audit: dict[str, object] = {
        "announcement_rows": len(announcement_rows),
        "company_days": len(company_days),
        "stock_codes": len(stock_codes),
        "price_rows": len(price_rows),
        "minimum_price_date": min(minimum_dates).isoformat() if minimum_dates else None,
        "maximum_price_date": max(minimum_dates).isoformat() if minimum_dates else None,
        "required_prehistory_start": contract.prehistory_required_start.isoformat(),
        "prehistory_requirement_met": bool(
            minimum_dates and min(minimum_dates) <= contract.prehistory_required_start
        ),
        "target_table_queried": False,
        "official_test_queried": False,
    }
    return company_days, price_frames, benchmark, {
        "identity_source_sha256": _stable_rows_sha256(announcement_rows),
        "price_source_sha256": _stable_rows_sha256(price_rows),
        "source_audit": source_audit,
    }


def build_label_free_train_artifacts(
    database_path: Path,
    output_directory: Path,
    *,
    contract: LstmTimeseriesContract,
    parent_contract: TimeseriesModelContract,
    development: CausalDevelopmentContract,
    force: bool = False,
) -> dict[str, object]:
    with _read_only_connection(database_path) as connection:
        company_days, stock_frames, benchmark, source = load_label_free_train_sources(
            connection,
            contract=contract,
        )
    bundle = build_direct_sequence_bundle(
        company_days,
        stock_bars_by_code=stock_frames,
        benchmark_bars=benchmark,
        parent_contract=parent_contract,
        contract=contract,
        development=development,
    )
    return write_direct_sequence_bundle(
        bundle,
        output_directory,
        contract=contract,
        identity_source_sha256=str(source["identity_source_sha256"]),
        price_source_sha256=str(source["price_source_sha256"]),
        source_audit=dict(source["source_audit"]),
        force=force,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "ddos.db",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "timeseries" / "lstm_daily_v1",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    development = load_causal_development_contract()
    parent = load_timeseries_model_contract(development=development)
    contract = load_lstm_timeseries_contract(
        parent=parent,
        development=development,
    )
    manifest = build_label_free_train_artifacts(
        args.database,
        args.output_dir,
        contract=contract,
        parent_contract=parent,
        development=development,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

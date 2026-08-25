"""Audit LSTM train artifacts without reading any sealed partition."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.prediction.timeseries.lstm_contract import (
    LstmTimeseriesContract,
    load_lstm_timeseries_contract,
)
from src.prediction.timeseries.sequence_artifacts import _file_sha256
from src.prediction.timeseries.sequence_identity import canonical_sample_id


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest must be an object: {path}")
    return value


def _require_manifest_contract(
    manifest: dict,
    *,
    contract: LstmTimeseriesContract,
) -> None:
    if manifest.get("contract") != contract.contract:
        raise ValueError("artifact contract name does not match current LSTM contract")
    if manifest.get("contract_sha256") != contract.source_sha256:
        raise ValueError("artifact contract hash does not match current LSTM contract")
    if manifest.get("split_contract_sha256") != contract.split_contract_sha256:
        raise ValueError("artifact split hash does not match current LSTM contract")
    if manifest.get("official_test_read") is not False:
        raise PermissionError("LSTM train artifact does not prove official-test isolation")


def _half_year(value: object) -> str:
    timestamp = pd.Timestamp(value)
    return f"{timestamp.year}-H{1 if timestamp.month <= 6 else 2}"


def audit_lstm_train_artifacts(
    artifact_directory: Path,
    *,
    contract: LstmTimeseriesContract,
    tabular_labels_path: Path | None = None,
) -> dict[str, object]:
    root = artifact_directory.resolve()
    if not root.is_relative_to(contract.data_directory):
        raise ValueError("LSTM audit input leaves the registered data boundary")
    sequence_manifest_path = root / "manifests" / "train_manifest.json"
    label_manifest_path = root / "manifests" / "train_labels_manifest.json"
    sequence_manifest = _load_json(sequence_manifest_path)
    label_manifest = _load_json(label_manifest_path)
    _require_manifest_contract(sequence_manifest, contract=contract)
    _require_manifest_contract(label_manifest, contract=contract)

    sequence_path = root / str(sequence_manifest["sequences_file"])
    mask_path = root / str(sequence_manifest["time_masks_file"])
    metadata_path = root / str(sequence_manifest["metadata_file"])
    label_path = root / str(label_manifest["labels_file"])
    hashes = {
        "sequences": _file_sha256(sequence_path),
        "time_masks": _file_sha256(mask_path),
        "metadata": _file_sha256(metadata_path),
        "labels": _file_sha256(label_path),
    }
    expected_hashes = {
        "sequences": sequence_manifest["sequences_sha256"],
        "time_masks": sequence_manifest["time_masks_sha256"],
        "metadata": sequence_manifest["metadata_sha256"],
        "labels": label_manifest["labels_sha256"],
    }
    if hashes != expected_hashes:
        raise ValueError("LSTM artifact content hash verification failed")

    sequences = np.load(sequence_path, mmap_mode="r", allow_pickle=False)
    masks = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    metadata = pd.read_parquet(metadata_path)
    labels = pd.read_parquet(label_path)
    if sequences.shape[:2] != masks.shape:
        raise ValueError("LSTM sequence and mask shapes differ")
    available = metadata.loc[metadata["sequence_available"].astype(bool)].copy()
    if len(available) != len(sequences):
        raise ValueError("available metadata count differs from sequence tensor")
    sequence_rows = available["sequence_row"].astype(int).sort_values().tolist()
    if sequence_rows != list(range(len(sequences))):
        raise ValueError("sequence_row is not a complete contiguous index")
    if set(metadata["dataset_role"].astype(str).unique()) != {"train"}:
        raise PermissionError("sequence metadata contains a sealed role")
    if set(labels["dataset_role"].astype(str).unique()) != {"train"}:
        raise PermissionError("sequence labels contain a sealed role")
    published = pd.to_datetime(metadata["published_date"], errors="raise")
    prediction = pd.to_datetime(metadata["prediction_as_of"], utc=True, errors="coerce")
    if (
        prediction.loc[prediction.notna()].dt.date
        <= published.loc[prediction.notna()].dt.date
    ).any():
        raise ValueError("LSTM prediction_as_of is not after publication date")
    last_bar = pd.to_datetime(
        available["last_bar_trade_date"], errors="raise"
    ).dt.date
    available_published = pd.to_datetime(
        available["published_date"], errors="raise"
    ).dt.date
    if any(
        bar >= event
        for bar, event in zip(last_bar, available_published, strict=True)
    ):
        raise ValueError("same-day or future daily bar found in LSTM metadata")
    data_as_of = pd.to_datetime(available["data_as_of"], utc=True, errors="raise")
    available_prediction = pd.to_datetime(
        available["prediction_as_of"], utc=True, errors="raise"
    )
    if (data_as_of > available_prediction).any():
        raise ValueError("LSTM sequence data is available after prediction_as_of")
    expected_sample_ids = [
        canonical_sample_id(row.company_day_id, int(row.horizon_sessions))
        for row in labels.itertuples(index=False)
    ]
    if expected_sample_ids != labels["sample_id"].astype(str).tolist():
        raise ValueError("LSTM label identity audit failed")
    if labels["sample_id"].duplicated().any():
        raise ValueError("duplicate canonical LSTM sample_id")

    metadata["half_year"] = metadata["published_date"].map(_half_year)
    by_half_year = {}
    for half_year, group in metadata.groupby("half_year", sort=True):
        by_half_year[str(half_year)] = {
            "company_days": int(len(group)),
            "available_20d": int((group["history_sessions"] >= 20).sum()),
            "full_60d": int((group["history_sessions"] >= 60).sum()),
        }
    source_audit = dict(sequence_manifest.get("source_audit") or {})
    prehistory_met = bool(source_audit.get("prehistory_requirement_met"))
    lstm_report_path = root / "reports" / "train_lstm_candidates.json"
    lstm_oof_audit: dict[str, object]
    if lstm_report_path.exists():
        lstm_report = _load_json(lstm_report_path)
        if lstm_report.get("contract") != contract.contract:
            raise ValueError("LSTM OOF contract name changed")
        if lstm_report.get("contract_sha256") != contract.source_sha256:
            raise ValueError("LSTM OOF contract hash changed")
        if lstm_report.get("official_test_read") is not False:
            raise PermissionError("official test reached LSTM OOF research")
        lstm_oof_path = root / str(lstm_report["oof_file"])
        if _file_sha256(lstm_oof_path) != lstm_report["oof_sha256"]:
            raise ValueError("LSTM OOF hash verification failed")
        eligible = list(lstm_report.get("eligible_for_table_increment_review") or [])
        lstm_oof_audit = {
            "run": True,
            "decision": lstm_report.get("decision"),
            "oof_rows": int(lstm_report["oof_rows"]),
            "oof_sha256": lstm_report["oof_sha256"],
            "candidate_count": len(dict(lstm_report.get("candidates") or {})),
            "eligible_for_table_increment_review": eligible,
            "training_gate_passed": bool(eligible),
            "official_test_read": lstm_report["official_test_read"],
        }
    else:
        lstm_oof_audit = {
            "run": False,
            "training_gate_passed": False,
            "eligible_for_table_increment_review": [],
        }
    comparison: dict[str, object] | None = None
    if tabular_labels_path is not None and tabular_labels_path.exists():
        table = pd.read_parquet(tabular_labels_path)
        if set(table["dataset_role"].astype(str).unique()) != {"train"}:
            raise PermissionError("tabular comparison labels are not train-only")
        joined = labels.merge(
            table,
            on="sample_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_lstm", "_tabular"),
        )
        comparison = {
            "comparison_only_not_lstm_source": True,
            "lstm_rows": len(labels),
            "tabular_rows": len(table),
            "matched_rows": len(joined),
            "identity_sets_equal": set(labels["sample_id"]) == set(table["sample_id"]),
            "directions_equal": bool(
                (
                    joined["actual_direction_lstm"]
                    == joined["actual_direction_tabular"]
                ).all()
            ),
            "maximum_absolute_excess_return_difference": float(
                (
                    joined["future_excess_return_lstm"]
                    - joined["future_excess_return_tabular"]
                ).abs().max()
            ),
        }

    return {
        "audit": "causal-timeseries-lstm-daily-v1-train-artifacts",
        "created_at": datetime.now(UTC).isoformat(),
        "contract": contract.contract,
        "contract_sha256": contract.source_sha256,
        "dataset_role_read": "train",
        "official_test_read": False,
        "quarantine_read": False,
        "forward_validation_read": False,
        "causal_integrity_pass": True,
        "research_20d_ready": bool((metadata["history_sessions"] >= 20).any()),
        "full_universe_60d_ready": prehistory_met,
        "release_eligible": False,
        "release_blockers": (
            []
            if prehistory_met
            else ["daily prehistory does not reach registered 2022-09-01 start"]
        )
        + (
            []
            if lstm_oof_audit["training_gate_passed"]
            else [
                "LSTM expanding-window OOF has not been run"
                if not lstm_oof_audit["run"]
                else "LSTM expanding-window OOF ran but did not pass"
            ]
        ),
        "company_days": len(metadata),
        "available_sequences": len(available),
        "available_20d": int((metadata["history_sessions"] >= 20).sum()),
        "full_60d": int((metadata["history_sessions"] >= 60).sum()),
        "labels": len(labels),
        "labels_by_horizon": {
            str(key): int(value)
            for key, value in labels.groupby("horizon_sessions").size().items()
        },
        "unavailable_reasons": dict(
            Counter(
                metadata.loc[
                    ~metadata["sequence_available"].astype(bool),
                    "unavailable_reason",
                ].astype(str)
            ).most_common()
        ),
        "by_half_year": by_half_year,
        "artifact_hashes": hashes,
        "source_audit": source_audit,
        "lstm_oof_audit": lstm_oof_audit,
        "tabular_train_label_comparison": comparison,
    }


def _write_json_atomic(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "timeseries" / "lstm_daily_v1",
    )
    parser.add_argument(
        "--tabular-labels",
        type=Path,
        default=PROJECT_ROOT / "data" / "tabular" / "train_v1" / "labels.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "timeseries"
            / "lstm_daily_v1"
            / "reports"
            / "train_artifact_audit.json"
        ),
    )
    args = parser.parse_args()
    contract = load_lstm_timeseries_contract()
    report = audit_lstm_train_artifacts(
        args.artifact_dir,
        contract=contract,
        tabular_labels_path=args.tabular_labels,
    )
    _write_json_atomic(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

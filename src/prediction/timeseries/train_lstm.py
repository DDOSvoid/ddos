"""Deterministic train-only expanding-window LSTM candidate research."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import PROJECT_ROOT
from src.prediction.development_contract import (
    CausalDevelopmentContract,
    load_causal_development_contract,
)
from src.prediction.timeseries.calibrate_lstm import (
    IntervalCalibration,
    TemperatureCalibration,
    fit_interval_expansion,
    fit_temperature_scaling,
)
from src.prediction.timeseries.evaluate_lstm import evaluate_lstm_candidate
from src.prediction.timeseries.lstm_contract import (
    LstmTimeseriesContract,
    load_lstm_timeseries_contract,
)
from src.prediction.timeseries.lstm_model import (
    SmallLstmMultiHead,
    multi_head_lstm_loss,
)
from src.prediction.timeseries.sequence_artifacts import (
    _file_sha256,
    _write_json_atomic,
)
from src.prediction.timeseries.sequence_dataset import (
    DirectSequenceFrames,
    assemble_direct_sequence_frames,
)
from src.prediction.timeseries.sequence_preprocessing import SequencePreprocessor
from src.prediction.timeseries.walk_forward import iter_timeseries_expanding_frames


@dataclass(frozen=True)
class InnerTemporalSplit:
    model_fit_indices: np.ndarray
    early_stop_indices: np.ndarray
    calibration_indices: np.ndarray
    audit: dict[str, object]


@dataclass(frozen=True)
class TrainedFoldModel:
    model: SmallLstmMultiHead
    preprocessor: SequencePreprocessor
    history: list[dict[str, float | int]]
    best_epoch: int
    best_early_stop_loss: float
    stopped_epoch: int
    parameter_count: int


def _segment_cutoff(day: object, zone: ZoneInfo) -> pd.Timestamp:
    local = datetime.combine(
        pd.Timestamp(day).date(),
        time(23, 59, 59),
        tzinfo=zone,
    )
    return pd.Timestamp(local.astimezone(UTC))


def build_inner_temporal_split(
    outer_train: pd.DataFrame,
    *,
    contract: LstmTimeseriesContract,
    development: CausalDevelopmentContract,
) -> InnerTemporalSplit:
    required = {"__row_index", "published_date", "outcome_available_at"}
    missing = required - set(outer_train.columns)
    if missing:
        raise ValueError(f"inner LSTM split missing columns: {sorted(missing)}")
    working = outer_train.copy()
    working["__published_date"] = pd.to_datetime(
        working["published_date"], errors="raise"
    ).dt.date
    working["__outcome_available_at"] = pd.to_datetime(
        working["outcome_available_at"], utc=True, errors="raise"
    )
    unique_dates = sorted(working["__published_date"].unique())
    if len(unique_dates) < 10:
        raise ValueError("inner LSTM temporal split has too few unique dates")
    fit_fraction, early_fraction, _ = contract.inner_temporal_partitions
    fit_date_count = max(1, int(np.floor(len(unique_dates) * fit_fraction)))
    early_end = max(
        fit_date_count + 1,
        int(np.floor(len(unique_dates) * (fit_fraction + early_fraction))),
    )
    if early_end >= len(unique_dates):
        raise ValueError("inner LSTM temporal split leaves no calibration dates")
    date_segments = {
        "model_fit": unique_dates[:fit_date_count],
        "early_stop": unique_dates[fit_date_count:early_end],
        "calibration": unique_dates[early_end:],
    }
    indices: dict[str, np.ndarray] = {}
    audit_segments: dict[str, object] = {}
    previous_end = None
    for name, dates in date_segments.items():
        if not dates:
            raise ValueError(f"empty inner LSTM segment: {name}")
        start_date = dates[0]
        end_date = dates[-1]
        if previous_end is not None and start_date <= previous_end:
            raise ValueError("inner LSTM temporal segments overlap")
        previous_end = end_date
        in_dates = working["__published_date"].isin(dates)
        before_maturity = int(in_dates.sum())
        cutoff = _segment_cutoff(end_date, development.zone)
        eligible = in_dates & (working["__outcome_available_at"] <= cutoff)
        selected = working.loc[eligible, "__row_index"].to_numpy(dtype=int)
        if len(selected) == 0:
            raise ValueError(f"inner LSTM segment has no mature labels: {name}")
        indices[name] = selected
        audit_segments[name] = {
            "published_date_start": start_date.isoformat(),
            "published_date_end": end_date.isoformat(),
            "unique_dates": len(dates),
            "samples_before_maturity_filter": before_maturity,
            "mature_samples": int(len(selected)),
            "outcome_maturity_cutoff": cutoff.isoformat(),
        }
    index_sets = [set(value.tolist()) for value in indices.values()]
    if any(index_sets[left] & index_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("inner LSTM sample partitions overlap")
    return InnerTemporalSplit(
        model_fit_indices=indices["model_fit"],
        early_stop_indices=indices["early_stop"],
        calibration_indices=indices["calibration"],
        audit={
            "split_unit": contract.inner_split_unit,
            "outcome_maturity_required": contract.inner_outcome_maturity_required,
            "segments": audit_segments,
        },
    )


def _set_deterministic(seed: int, contract: LstmTimeseriesContract) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(contract.deterministic_algorithms)


def _tensor_dataset(
    frames: DirectSequenceFrames,
    indices: np.ndarray,
    *,
    preprocessor: SequencePreprocessor,
) -> TensorDataset:
    sequences = preprocessor.transform(
        frames.sequences[indices],
        frames.time_masks[indices],
    )
    metadata = frames.metadata.iloc[indices]
    return TensorDataset(
        torch.as_tensor(sequences, dtype=torch.float32),
        torch.as_tensor(frames.time_masks[indices], dtype=torch.bool),
        torch.as_tensor(metadata["target"].to_numpy(dtype=np.float32)),
        torch.as_tensor(metadata["excess_return"].to_numpy(dtype=np.float32)),
    )


def _loss_kwargs(contract: LstmTimeseriesContract) -> dict[str, float]:
    return {
        "lower_quantile": contract.interval_quantiles[0],
        "upper_quantile": contract.interval_quantiles[1],
        "huber_delta": contract.huber_delta,
        "direction_weight": contract.direction_loss_weight,
        "return_weight": contract.return_loss_weight,
        "interval_weight": contract.interval_loss_weight,
    }


def _evaluate_loss(
    model: SmallLstmMultiHead,
    loader: DataLoader,
    *,
    contract: LstmTimeseriesContract,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "direction": 0.0, "return": 0.0, "interval": 0.0}
    samples = 0
    with torch.no_grad():
        for sequence, mask, target, excess_return in loader:
            output = model(sequence, mask)
            loss = multi_head_lstm_loss(
                output,
                direction_target=target,
                excess_return_target=excess_return,
                **_loss_kwargs(contract),
            )
            count = len(sequence)
            samples += count
            totals["total"] += float(loss.total.item()) * count
            totals["direction"] += float(loss.direction.item()) * count
            totals["return"] += float(loss.expected_return.item()) * count
            totals["interval"] += float(loss.interval.item()) * count
    if samples == 0:
        raise ValueError("LSTM early-stop loader is empty")
    return {name: value / samples for name, value in totals.items()}


def train_fold_lstm(
    frames: DirectSequenceFrames,
    split: InnerTemporalSplit,
    *,
    hidden_size: int,
    seed: int,
    contract: LstmTimeseriesContract,
) -> TrainedFoldModel:
    _set_deterministic(seed, contract)
    preprocessor = SequencePreprocessor.fit(
        frames.sequences[split.model_fit_indices],
        frames.time_masks[split.model_fit_indices],
    )
    fit_dataset = _tensor_dataset(
        frames,
        split.model_fit_indices,
        preprocessor=preprocessor,
    )
    early_dataset = _tensor_dataset(
        frames,
        split.early_stop_indices,
        preprocessor=preprocessor,
    )
    generator = torch.Generator().manual_seed(seed)
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=contract.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    early_loader = DataLoader(
        early_dataset,
        batch_size=contract.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = SmallLstmMultiHead(
        input_size=len(contract.sequence_columns),
        hidden_size=hidden_size,
        external_dropout=contract.external_dropout_candidates[0],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=contract.learning_rate,
        weight_decay=contract.weight_decay,
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    stopped_epoch = contract.maximum_epochs
    for epoch in range(1, contract.maximum_epochs + 1):
        model.train()
        train_total = 0.0
        train_samples = 0
        for sequence, mask, target, excess_return in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(sequence, mask)
            loss = multi_head_lstm_loss(
                output,
                direction_target=target,
                excess_return_target=excess_return,
                **_loss_kwargs(contract),
            )
            if not torch.isfinite(loss.total):
                raise FloatingPointError("non-finite LSTM training loss")
            loss.total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), contract.gradient_clip_norm)
            optimizer.step()
            train_total += float(loss.total.item()) * len(sequence)
            train_samples += len(sequence)
        early = _evaluate_loss(model, early_loader, contract=contract)
        history.append(
            {
                "epoch": epoch,
                "model_fit_weighted_loss": train_total / train_samples,
                "early_stop_weighted_loss": early["total"],
                "early_stop_direction_loss": early["direction"],
                "early_stop_return_loss": early["return"],
                "early_stop_interval_loss": early["interval"],
            }
        )
        if early["total"] < best_loss - 1e-8:
            best_loss = early["total"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (
            epoch >= contract.minimum_epochs
            and epochs_without_improvement >= contract.early_stopping_patience
        ):
            stopped_epoch = epoch
            break
    if best_state is None:
        raise RuntimeError("LSTM early stopping did not retain a checkpoint")
    model.load_state_dict(best_state)
    return TrainedFoldModel(
        model=model,
        preprocessor=preprocessor,
        history=history,
        best_epoch=best_epoch,
        best_early_stop_loss=best_loss,
        stopped_epoch=stopped_epoch,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def predict_fold_lstm(
    trained: TrainedFoldModel,
    frames: DirectSequenceFrames,
    indices: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    dataset = _tensor_dataset(
        frames,
        indices,
        preprocessor=trained.preprocessor,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    collected: dict[str, list[np.ndarray]] = {
        "direction_logit": [],
        "bullish_probability": [],
        "expected_excess_return": [],
        "return_interval_lower": [],
        "return_interval_upper": [],
    }
    trained.model.eval()
    with torch.no_grad():
        for sequence, mask, _, _ in loader:
            output = trained.model(sequence, mask)
            for name in collected:
                collected[name].append(output[name].detach().cpu().numpy())
    return {
        name: np.concatenate(parts).astype(float, copy=False)
        for name, parts in collected.items()
    }


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest must be an object: {path}")
    return value


def _save_checkpoint(
    path: Path,
    *,
    trained: TrainedFoldModel,
    contract: LstmTimeseriesContract,
    candidate_key: str,
    fold_name: str,
    seed: int,
    inner_audit: dict[str, object],
    temperature: TemperatureCalibration,
    interval: IntervalCalibration,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "artifact": "causal-timeseries-lstm-daily-v1-research-checkpoint",
            "contract": contract.contract,
            "contract_sha256": contract.source_sha256,
            "candidate": candidate_key,
            "fold": fold_name,
            "seed": seed,
            "model_state_dict": trained.model.state_dict(),
            "preprocessor": {
                "medians": trained.preprocessor.medians,
                "means": trained.preprocessor.means,
                "scales": trained.preprocessor.scales,
                "continuous_feature_count": trained.preprocessor.continuous_feature_count,
            },
            "temperature_calibration": temperature.as_dict(),
            "interval_calibration": interval.as_dict(),
            "best_epoch": trained.best_epoch,
            "stopped_epoch": trained.stopped_epoch,
            "history": trained.history,
            "inner_temporal_split": inner_audit,
        },
        temporary,
    )
    temporary.replace(path)
    return _file_sha256(path)


def _row_evidence_sha256(
    *,
    sample_id: str,
    sequence_evidence_sha256: str,
    checkpoint_sha256: str,
) -> str:
    return hashlib.sha256(
        "|".join(
            ("lstm_oof", sample_id, sequence_evidence_sha256, checkpoint_sha256)
        ).encode("utf-8")
    ).hexdigest()


def _candidate_fold_output(
    validation: pd.DataFrame,
    raw: dict[str, np.ndarray],
    *,
    candidate_key: str,
    fold_name: str,
    lookback: int,
    hidden_size: int,
    temperature: TemperatureCalibration,
    interval: IntervalCalibration,
    checkpoint_sha256: str,
) -> pd.DataFrame:
    output = validation.loc[
        :,
        [
            "sample_id",
            "company_day_id",
            "stock_code",
            "published_date",
            "prediction_as_of",
            "outcome_available_at",
            "horizon_sessions",
            "target",
            "future_excess_return",
            "data_as_of",
            "evidence_sha256",
        ],
    ].copy()
    output = output.rename(columns={"evidence_sha256": "sequence_evidence_sha256"})
    calibrated_probability = temperature.predict_probability(raw["direction_logit"])
    calibrated_lower, calibrated_upper = interval.transform(
        raw["return_interval_lower"],
        raw["return_interval_upper"],
    )
    output["fold"] = fold_name
    output["lookback_sessions"] = lookback
    output["hidden_size"] = hidden_size
    output["model_family"] = "small_single_layer_lstm"
    output["model_version"] = candidate_key
    output["raw_direction_logit"] = raw["direction_logit"]
    output["raw_bullish_probability"] = raw["bullish_probability"]
    output["bullish_probability"] = calibrated_probability
    output["expected_excess_return"] = raw["expected_excess_return"]
    output["raw_interval_lower"] = raw["return_interval_lower"]
    output["raw_interval_upper"] = raw["return_interval_upper"]
    output["return_interval_lower"] = calibrated_lower
    output["return_interval_upper"] = calibrated_upper
    output["risk_scale"] = (calibrated_upper - calibrated_lower) / 2.0
    output["sequence_available"] = True
    output["unavailable_reason"] = None
    output["checkpoint_sha256"] = checkpoint_sha256
    output["evidence_sha256"] = [
        _row_evidence_sha256(
            sample_id=str(row.sample_id),
            sequence_evidence_sha256=str(row.sequence_evidence_sha256),
            checkpoint_sha256=checkpoint_sha256,
        )
        for row in output.itertuples(index=False)
    ]
    return output


def run_lstm_candidate_oof(
    frames: DirectSequenceFrames,
    simple_baseline_oof: pd.DataFrame,
    *,
    lookback: int,
    hidden_size: int,
    horizon: int,
    contract: LstmTimeseriesContract,
    development: CausalDevelopmentContract,
    model_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    indexed_metadata = frames.metadata.copy()
    indexed_metadata["__row_index"] = np.arange(len(indexed_metadata), dtype=int)
    fold_outputs = []
    training_reports = []
    candidate_key = f"lstm_lb{lookback}_h{hidden_size}_d{horizon}"
    for fold_index, fold in enumerate(
        iter_timeseries_expanding_frames(indexed_metadata, development=development),
        1,
    ):
        print(f"training {candidate_key} {fold.name}", flush=True)
        inner = build_inner_temporal_split(
            fold.train,
            contract=contract,
            development=development,
        )
        seed = contract.seed + horizon * 10_000 + lookback * 100 + hidden_size * 10 + fold_index
        trained = train_fold_lstm(
            frames,
            inner,
            hidden_size=hidden_size,
            seed=seed,
            contract=contract,
        )
        calibration_raw = predict_fold_lstm(
            trained,
            frames,
            inner.calibration_indices,
            batch_size=contract.batch_size,
        )
        calibration_metadata = frames.metadata.iloc[inner.calibration_indices]
        temperature = fit_temperature_scaling(
            calibration_raw["direction_logit"],
            calibration_metadata["target"].to_numpy(dtype=float),
            bounds=contract.probability_temperature_bounds,
            identity_fallback_if_brier_worsens=contract.calibration_identity_fallback,
        )
        interval = fit_interval_expansion(
            calibration_raw["return_interval_lower"],
            calibration_raw["return_interval_upper"],
            calibration_metadata["excess_return"].to_numpy(dtype=float),
            target_coverage=contract.interval_target_coverage,
        )
        validation_indices = fold.validation["__row_index"].to_numpy(dtype=int)
        validation_raw = predict_fold_lstm(
            trained,
            frames,
            validation_indices,
            batch_size=contract.batch_size,
        )
        checkpoint_path = model_root / candidate_key / f"{fold.name}.pt"
        checkpoint_sha256 = _save_checkpoint(
            checkpoint_path,
            trained=trained,
            contract=contract,
            candidate_key=candidate_key,
            fold_name=fold.name,
            seed=seed,
            inner_audit=inner.audit,
            temperature=temperature,
            interval=interval,
        )
        fold_output = _candidate_fold_output(
            fold.validation,
            validation_raw,
            candidate_key=candidate_key,
            fold_name=fold.name,
            lookback=lookback,
            hidden_size=hidden_size,
            temperature=temperature,
            interval=interval,
            checkpoint_sha256=checkpoint_sha256,
        )
        fold_outputs.append(fold_output)
        training_reports.append(
            {
                "fold": fold.name,
                "seed": seed,
                "outer_train_samples": int(len(fold.train)),
                "outer_validation_samples": int(len(fold.validation)),
                "inner_temporal_split": inner.audit,
                "best_epoch": trained.best_epoch,
                "stopped_epoch": trained.stopped_epoch,
                "best_early_stop_weighted_loss": trained.best_early_stop_loss,
                "parameter_count": trained.parameter_count,
                "history": trained.history,
                "temperature_calibration": temperature.as_dict(),
                "interval_calibration": interval.as_dict(),
                "checkpoint_file": checkpoint_path.relative_to(
                    contract.model_directory
                ).as_posix(),
                "checkpoint_sha256": checkpoint_sha256,
            }
        )
    candidate_oof = pd.concat(fold_outputs, ignore_index=True)
    evaluation = evaluate_lstm_candidate(
        candidate_oof,
        simple_baseline_oof,
        lookback_sessions=lookback,
        horizon_sessions=horizon,
        contract=contract,
        development=development,
    )
    return evaluation.oof_predictions, {
        "candidate": candidate_key,
        "lookback_sessions": lookback,
        "hidden_size": hidden_size,
        "horizon_sessions": horizon,
        "architecture": {
            "family": contract.model_family,
            "num_layers": contract.num_layers,
            "bidirectional": contract.bidirectional,
            "external_dropout": contract.external_dropout_candidates[0],
        },
        "fold_training": training_reports,
        "evaluation": evaluation.report,
    }


def run_train_lstm(
    artifact_directory: Path,
    *,
    contract: LstmTimeseriesContract,
    development: CausalDevelopmentContract,
    force: bool = False,
) -> dict[str, object]:
    root = artifact_directory.resolve()
    if not root.is_relative_to(contract.data_directory):
        raise ValueError("LSTM training input leaves the registered data boundary")
    sequence_manifest_path = root / "manifests" / "train_manifest.json"
    label_manifest_path = root / "manifests" / "train_labels_manifest.json"
    baseline_report_path = root / "reports" / "train_simple_sequence_baselines.json"
    sequence_manifest = _load_json(sequence_manifest_path)
    label_manifest = _load_json(label_manifest_path)
    baseline_report = _load_json(baseline_report_path)
    for manifest in (sequence_manifest, label_manifest, baseline_report):
        if manifest.get("contract_sha256") != contract.source_sha256:
            raise ValueError("LSTM training dependency contract hash changed")
        if manifest.get("official_test_read") is not False:
            raise PermissionError("official test reached LSTM training dependency")
    sequence_path = root / str(sequence_manifest["sequences_file"])
    mask_path = root / str(sequence_manifest["time_masks_file"])
    metadata_path = root / str(sequence_manifest["metadata_file"])
    label_path = root / str(label_manifest["labels_file"])
    baseline_oof_path = root / str(baseline_report["oof_file"])
    for path, expected in (
        (sequence_path, sequence_manifest["sequences_sha256"]),
        (mask_path, sequence_manifest["time_masks_sha256"]),
        (metadata_path, sequence_manifest["metadata_sha256"]),
        (label_path, label_manifest["labels_sha256"]),
        (baseline_oof_path, baseline_report["oof_sha256"]),
    ):
        if _file_sha256(path) != expected:
            raise ValueError(f"LSTM training dependency hash changed: {path.name}")

    oof_path = root / "oof" / "train_lstm_candidates.parquet"
    report_path = root / "reports" / "train_lstm_candidates.json"
    existing = [path for path in (oof_path, report_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(f"LSTM research output already exists: {existing[0]}")
    model_root = contract.model_directory / "research"
    if not model_root.is_relative_to(contract.model_directory):
        raise ValueError("LSTM checkpoint output leaves the model boundary")
    torch.set_num_threads(contract.cpu_threads)
    sequences = np.load(sequence_path, mmap_mode="r", allow_pickle=False)
    masks = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    metadata = pd.read_parquet(metadata_path)
    labels = pd.read_parquet(label_path)
    simple_baseline_oof = pd.read_parquet(baseline_oof_path)
    candidate_predictions = []
    candidate_reports: dict[str, object] = {}
    for lookback in contract.lookback_candidates_sessions:
        for horizon in contract.horizons_sessions:
            frames = assemble_direct_sequence_frames(
                sequences,
                masks,
                metadata,
                labels,
                horizon_sessions=horizon,
                lookback_sessions=lookback,
                contract=contract,
            )
            for hidden_size in contract.hidden_size_candidates:
                oof, candidate_report = run_lstm_candidate_oof(
                    frames,
                    simple_baseline_oof,
                    lookback=lookback,
                    hidden_size=hidden_size,
                    horizon=horizon,
                    contract=contract,
                    development=development,
                    model_root=model_root,
                )
                candidate_key = str(candidate_report["candidate"])
                candidate_reports[candidate_key] = candidate_report
                candidate_predictions.append(oof)

    all_oof = pd.concat(candidate_predictions, ignore_index=True).sort_values(
        [
            "lookback_sessions",
            "hidden_size",
            "horizon_sessions",
            "published_date",
            "sample_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = oof_path.with_suffix(oof_path.suffix + ".tmp")
    all_oof.to_parquet(temporary, index=False)
    temporary.replace(oof_path)
    eligible = [
        key
        for key, report in candidate_reports.items()
        if report["evaluation"]["train_gate_passed_before_table_increment"]
    ]
    report: dict[str, object] = {
        "experiment": "causal-timeseries-lstm-daily-v1-expanding-window-oof",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "KEEP_RESEARCH_ONLY_DO_NOT_FREEZE",
        "decision": (
            "PROCEED_TO_TABLE_INCREMENT_REVIEW"
            if eligible
            else "KEEP_RESEARCH_ONLY_DO_NOT_FREEZE"
        ),
        "contract": contract.contract,
        "contract_sha256": contract.source_sha256,
        "development_contract_sha256": contract.development_contract_sha256,
        "dataset_role_read": "train",
        "official_test_read": False,
        "quarantine_read": False,
        "forward_validation_read": False,
        "torch_version": torch.__version__,
        "device": contract.training_device,
        "cpu_threads": contract.cpu_threads,
        "deterministic_algorithms": contract.deterministic_algorithms,
        "sequence_manifest_sha256": _file_sha256(sequence_manifest_path),
        "label_manifest_sha256": _file_sha256(label_manifest_path),
        "simple_baseline_report_sha256": _file_sha256(baseline_report_path),
        "simple_baseline_oof_sha256": _file_sha256(baseline_oof_path),
        "producer_code_sha256": _file_sha256(Path(__file__)),
        "model_code_sha256": _file_sha256(
            PROJECT_ROOT / "src" / "prediction" / "timeseries" / "lstm_model.py"
        ),
        "calibration_code_sha256": _file_sha256(
            PROJECT_ROOT
            / "src"
            / "prediction"
            / "timeseries"
            / "calibrate_lstm.py"
        ),
        "evaluation_code_sha256": _file_sha256(
            PROJECT_ROOT
            / "src"
            / "prediction"
            / "timeseries"
            / "evaluate_lstm.py"
        ),
        "oof_file": oof_path.relative_to(root).as_posix(),
        "oof_sha256": _file_sha256(oof_path),
        "oof_rows": int(len(all_oof)),
        "eligible_for_table_increment_review": eligible,
        "release_eligible": False,
        "official_test_eligible": False,
        "candidates": candidate_reports,
    }
    _write_json_atomic(report, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "timeseries" / "lstm_daily_v1",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    development = load_causal_development_contract()
    contract = load_lstm_timeseries_contract(development=development)
    report = run_train_lstm(
        args.artifact_dir,
        contract=contract,
        development=development,
        force=args.force,
    )
    summary = {
        "experiment": report["experiment"],
        "status": report["status"],
        "decision": report["decision"],
        "contract_sha256": report["contract_sha256"],
        "oof_rows": report["oof_rows"],
        "oof_sha256": report["oof_sha256"],
        "eligible_for_table_increment_review": report[
            "eligible_for_table_increment_review"
        ],
        "official_test_read": report["official_test_read"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

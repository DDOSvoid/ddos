"""Compare matched daily and lagged complete-day 15-minute OOF baselines."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.prediction.development_contract import load_causal_development_contract
from src.prediction.timeseries.artifacts import (
    _file_sha256,
    _write_csv_atomic,
    _write_json_atomic,
)
from src.prediction.timeseries.baselines import (
    LogisticBaselineResult,
    run_logistic_baseline_oof,
)
from src.prediction.timeseries.dataset import assemble_train_frame
from src.prediction.timeseries.intraday_contract import (
    IntradayAggregateTimeseriesContract,
    load_intraday_aggregate_timeseries_contract,
)


def _read_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        parse_dates=["published_date", "prediction_as_of", "last_bar_available_at"],
    )
    frame["published_date"] = frame["published_date"].dt.date
    frame["last_bar_trade_date"] = pd.to_datetime(
        frame["last_bar_trade_date"], errors="raise"
    ).dt.date
    return frame


def _read_labels(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["outcome_available_at"])


def _report_without_oof(result: LogisticBaselineResult) -> dict[str, object]:
    return result.report


def _fold_brier_by_name(horizon_report: dict[str, object]) -> dict[str, float]:
    return {
        str(item["fold"]): float(item["brier_score"])
        for item in horizon_report["folds"]
    }


def _brier_increment_comparison(
    daily_report: dict[str, object],
    combined_report: dict[str, object],
) -> tuple[float, float, dict[str, float], bool]:
    daily_brier = float(daily_report["pooled_metrics"]["brier_score"])
    combined_brier = float(combined_report["pooled_metrics"]["brier_score"])
    daily_folds = _fold_brier_by_name(daily_report)
    combined_folds = _fold_brier_by_name(combined_report)
    if set(daily_folds) != set(combined_folds):
        raise ValueError("daily and combined fold identities differ")
    fold_improvements = {
        fold: daily_folds[fold] - combined_folds[fold] for fold in daily_folds
    }
    stable_increment = (
        combined_brier < daily_brier
        and sum(value > 0 for value in fold_improvements.values()) >= 2
    )
    return daily_brier, combined_brier, fold_improvements, stable_increment


def _general_precheck(
    horizon_report: dict[str, object],
    *,
    development,
) -> dict[str, object]:
    gate = development.module_gates["predictive_general"]
    pooled = horizon_report["pooled_metrics"]
    folds = horizon_report["folds"]
    fold_hit_rates = [
        float(item["bullish_hit_rate"])
        if item["bullish_hit_rate"] is not None
        else 0.0
        for item in folds
    ]
    checks = {
        "minimum_oof_bullish_signals": (
            int(pooled["bullish_signals"])
            >= int(gate["minimum_oof_bullish_signals_per_horizon"])
        ),
        "bullish_wilson_lower": (
            float(pooled["bullish_hit_rate_wilson_95pct"][0])
            > float(gate["bullish_wilson_lower_must_exceed"])
        ),
        "folds_above_half": (
            sum(value > 0.5 for value in fold_hit_rates)
            >= int(gate["minimum_folds_with_bullish_hit_rate_above_half"])
        ),
        "worst_fold_hit_rate": (
            min(fold_hit_rates) >= float(gate["minimum_worst_fold_bullish_hit_rate"])
        ),
        "brier_beats_constant": (
            float(pooled["brier_score"])
            < float(pooled["constant_base_rate_brier_score"])
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def run_intraday_matched_baseline(
    artifact_directory: Path,
    *,
    parent_artifact_directory: Path,
    contract: IntradayAggregateTimeseriesContract,
    force: bool = False,
) -> dict[str, object]:
    root = artifact_directory.resolve()
    parent_root = parent_artifact_directory.resolve()
    for path, label in ((root, "intraday"), (parent_root, "parent daily")):
        if not path.is_relative_to(contract.data_directory):
            raise ValueError(f"{label} artifact directory left data/timeseries")
    manifest_path = root / "manifests" / "train_intraday_aggregates_v1_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["contract_sha256"] != contract.source_sha256:
        raise ValueError("intraday artifact contract hash changed")
    if manifest["parent_contract_sha256"] != contract.parent_contract_sha256:
        raise ValueError("intraday artifact parent contract hash changed")
    for role in ("official_test", "quarantine", "forward_validation"):
        if manifest[f"{role}_read"]:
            raise PermissionError(f"sealed role reached intraday artifact: {role}")
    if manifest["publication_day_intraday_used"]:
        raise PermissionError("publication-day intraday data reached train artifact")
    intraday_feature_path = root / manifest["features_file"]
    label_path = root / manifest["labels_file"]
    if _file_sha256(intraday_feature_path) != manifest["features_sha256"]:
        raise ValueError("intraday feature artifact hash changed")
    if _file_sha256(label_path) != manifest["labels_sha256"]:
        raise ValueError("intraday label artifact hash changed")

    parent_manifest_path = parent_root / "manifests" / "train_daily_v1_manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if _file_sha256(parent_manifest_path) != manifest["parent_manifest_sha256"]:
        raise ValueError("parent train manifest hash changed")
    parent_feature_path = parent_root / parent_manifest["features_file"]
    if _file_sha256(parent_feature_path) != manifest["parent_features_sha256"]:
        raise ValueError("parent daily feature artifact hash changed")

    labels = _read_labels(label_path)
    intraday_features = _read_features(intraday_feature_path)
    included_ids = set(intraday_features["sample_id"])
    parent_features = _read_features(parent_feature_path)
    parent_features = parent_features.loc[
        parent_features["sample_id"].isin(included_ids)
    ].copy()
    if set(parent_features["sample_id"]) != included_ids:
        raise ValueError("matched daily and intraday sample identities differ")

    daily_columns = tuple(parent_manifest["feature_columns"])
    intraday_columns = tuple(manifest["feature_columns"])
    daily_frame = assemble_train_frame(
        parent_features,
        labels,
        feature_columns=daily_columns,
        contract=contract,
    )
    intraday_frame = assemble_train_frame(
        intraday_features,
        labels,
        feature_columns=intraday_columns,
        contract=contract,
    )
    combined_features = intraday_features.merge(
        parent_features.loc[:, ["sample_id", *daily_columns]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    combined_columns = (*daily_columns, *intraday_columns)
    combined_frame = assemble_train_frame(
        combined_features,
        labels,
        feature_columns=combined_columns,
        contract=contract,
    )
    frames = {
        "daily_only_matched": (daily_frame, daily_columns),
        "intraday_only": (intraday_frame, intraday_columns),
        "daily_plus_intraday": (combined_frame, combined_columns),
    }
    if tuple(frames) != contract.feature_set_candidates:
        raise ValueError("feature-set execution order differs from contract")

    development = load_causal_development_contract()
    results = {
        name: run_logistic_baseline_oof(
            frame,
            feature_columns=columns,
            contract=contract,
            development=development,
        )
        for name, (frame, columns) in frames.items()
    }
    reference_ids = set(results["daily_only_matched"].oof_predictions["sample_id"])
    for name, result in results.items():
        if set(result.oof_predictions["sample_id"]) != reference_ids:
            raise ValueError(f"OOF sample identities differ for {name}")

    comparisons = {}
    tcn_eligible_horizons = []
    for horizon in contract.horizons_sessions:
        key = str(horizon)
        daily = results["daily_only_matched"].report["horizons"][key]
        intraday = results["intraday_only"].report["horizons"][key]
        combined = results["daily_plus_intraday"].report["horizons"][key]
        intraday_brier = float(intraday["pooled_metrics"]["brier_score"])
        (
            daily_brier,
            combined_brier,
            fold_improvements,
            stable_increment,
        ) = _brier_increment_comparison(
            daily,
            combined,
        )
        general_precheck = _general_precheck(
            combined,
            development=development,
        )
        tcn_eligible = stable_increment and bool(general_precheck["passed"])
        if tcn_eligible:
            tcn_eligible_horizons.append(horizon)
        selected_name = min(
            results,
            key=lambda name: float(
                results[name].report["horizons"][key]["pooled_metrics"][
                    "brier_score"
                ]
            ),
        )
        comparisons[key] = {
            "identical_oof_samples": True,
            "oof_samples": int(daily["pooled_metrics"]["samples"]),
            "daily_only_matched_brier": daily_brier,
            "intraday_only_brier": intraday_brier,
            "daily_plus_intraday_brier": combined_brier,
            "intraday_only_brier_improvement_over_daily": (
                daily_brier - intraday_brier
            ),
            "daily_plus_intraday_brier_improvement_over_daily": (
                daily_brier - combined_brier
            ),
            "daily_plus_intraday_fold_brier_improvements": fold_improvements,
            "stable_intraday_increment": stable_increment,
            "combined_general_predictive_precheck": general_precheck,
            "selected_feature_set_by_brier": selected_name,
            "tcn_eligible": tcn_eligible,
        }

    oof_frames = []
    for name, result in results.items():
        item = result.oof_predictions.copy()
        item["feature_set"] = name
        oof_frames.append(item)
    oof = pd.concat(oof_frames, ignore_index=True).sort_values(
        ["feature_set", "horizon_sessions", "published_date", "sample_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    oof_path = root / "oof" / "train_intraday_aggregates_v1_matched_oof.csv"
    report_path = root / "reports" / "train_intraday_aggregates_v1_matched_oof.json"
    existing = [path for path in (oof_path, report_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(f"intraday baseline output exists: {existing[0]}")
    _write_csv_atomic(oof, oof_path)
    report = {
        "experiment": "causal-timeseries-intraday-aggregates-v1-matched-baseline",
        "status": (
            "train_gate_pass_pending_cost_and_table_increment"
            if tcn_eligible_horizons
            else "train_gate_fail_tcn_blocked"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "contract": contract.contract,
        "contract_sha256": contract.source_sha256,
        "dataset_role_read": "train",
        "official_test_read": False,
        "quarantine_read": False,
        "forward_validation_read": False,
        "publication_day_intraday_used": False,
        "matched_samples": len(intraday_features),
        "fixed_logistic_c": contract.logistic_c_values[0],
        "fixed_logistic_class_weight": contract.logistic_class_weights[0],
        "stability_rule": contract.stability_rule,
        "feature_set_results": {
            name: _report_without_oof(result) for name, result in results.items()
        },
        "comparisons": comparisons,
        "tcn_eligible_horizons": tcn_eligible_horizons,
        "oof_file": oof_path.relative_to(root).as_posix(),
        "oof_sha256": _file_sha256(oof_path),
        "train_manifest_sha256": _file_sha256(manifest_path),
        "limitations": [
            "uses complete pre-publication trading-day aggregates only",
            "does not use publication-day or minute-level decision sequences",
            "cost-adjusted confidence interval remains a later gate",
            "table-model incremental comparison remains a later gate",
        ],
    }
    _write_json_atomic(report, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "timeseries" / "intraday_aggregates_v1"
        ),
    )
    parser.add_argument(
        "--parent-artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "timeseries" / "daily_v1",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = run_intraday_matched_baseline(
        args.artifact_dir,
        parent_artifact_directory=args.parent_artifact_dir,
        contract=load_intraday_aggregate_timeseries_contract(),
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

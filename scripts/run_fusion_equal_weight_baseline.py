#!/usr/bin/env python
"""Run the first non-trainable fusion baseline and optional ranking evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prediction.fusion.baselines import run_equal_weight_baseline
from src.prediction.fusion.contract import load_expert_fusion_contract
from src.prediction.fusion.ranking import evaluate_daily_ranking
from src.prediction.fusion.schema import validate_development_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-oof", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    contract = load_expert_fusion_contract()
    joint = pd.read_parquet(args.joint_oof)
    predictions = run_equal_weight_baseline(joint, contract=contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    predictions.to_parquet(temporary, index=False)
    temporary.replace(args.output)

    result: dict[str, object] = {
        "contract": "equal-weight-fusion-baseline-run-v1",
        "prediction_rows": len(predictions),
        "available_rows": int(predictions["fusion_available"].sum()),
        "official_test_read": False,
        "quarantine_read": False,
        "strict_oos_2026_read": False,
    }
    if args.labels:
        labels = validate_development_labels(pd.read_parquet(args.labels), contract=contract)
        evaluated = predictions.merge(
            labels[["sample_id", "future_excess_return"]],
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )
        result["ranking"] = evaluate_daily_ranking(
            evaluated,
            top_k=contract.top_k,
            round_trip_cost_bps=contract.round_trip_cost_bps,
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

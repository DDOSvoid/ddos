#!/usr/bin/env python
"""Build a deterministic train-only financial revision evidence plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.prediction.development_contract import load_causal_development_contract
from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.contract import load_tabular_model_contract
from src.prediction.tabular.financial_revision_evidence import (
    build_financial_revision_plan,
    load_financial_revision_evidence_contract,
)
from src.prediction.tabular.point_in_time_sources import (
    load_point_in_time_source_contract,
)


def main() -> None:
    default_root = PROJECT_ROOT / "data" / "tabular" / "point_in_time_sources"
    default_output = (
        PROJECT_ROOT
        / "data"
        / "tabular"
        / "financial_revision_evidence_v1"
        / "plan.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--financial-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    contract = load_financial_revision_evidence_contract()
    source = load_point_in_time_source_contract()
    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
    tabular = load_tabular_model_contract(split=split)
    plan = build_financial_revision_plan(
        financial_root=args.financial_root,
        output_path=args.output,
        contract=contract,
        source_contract=source,
        split_contract=split.contract,
        split_source_sha256=split.source_sha256,
        development_contract=development.contract,
        development_source_sha256=development.source_sha256,
        tabular_contract=tabular.contract,
        tabular_source_sha256=tabular.source_sha256,
    )
    print(
        json.dumps(
            {
                "eligible": plan["eligible_revision_groups_by_endpoint"],
                "matched": plan["matched_revision_groups_by_endpoint"],
                "selected": plan["selected_revision_groups_by_endpoint"],
                "announcement_ids": len(plan["announcement_ids"]),
                "official_test_queried": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

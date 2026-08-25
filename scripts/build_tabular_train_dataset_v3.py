#!/usr/bin/env python
"""Build the train-only v3 table from passed industry and financial audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.prediction.development_contract import load_causal_development_contract
from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.catalog_v3 import load_tabular_feature_catalog_v3
from src.prediction.tabular.contract import load_tabular_model_contract
from src.prediction.tabular.dataset_v3 import build_tabular_train_dataset_v3
from src.prediction.tabular.point_in_time_sources import save_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = PROJECT_ROOT / "data" / "tabular"
    parser.add_argument("--v2-root", type=Path, default=root / "train_v2")
    parser.add_argument("--output-root", type=Path, default=root / "train_v3")
    parser.add_argument(
        "--industry-root", type=Path, default=root / "industry_supplement_v1"
    )
    parser.add_argument(
        "--existing-industry-root",
        type=Path,
        default=root / "point_in_time_sources",
    )
    parser.add_argument(
        "--financial-root", type=Path, default=root / "point_in_time_sources"
    )
    parser.add_argument(
        "--financial-evidence-root",
        type=Path,
        default=root / "financial_revision_evidence_v1",
    )
    args = parser.parse_args()
    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
    tabular = load_tabular_model_contract(split=split, development=development)
    catalog = load_tabular_feature_catalog_v3(
        tabular=tabular, split=split, development=development
    )
    features, labels, manifest = build_tabular_train_dataset_v3(
        v2_root=args.v2_root,
        industry_root=args.industry_root,
        financial_root=args.financial_root,
        industry_audit_path=args.industry_root / "audit.report.json",
        financial_audit_path=args.financial_evidence_root / "audit.report.json",
        catalog=catalog,
        existing_industry_root=args.existing_industry_root,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output_root / "features.parquet", index=False)
    labels.to_parquet(args.output_root / "labels.parquet", index=False)
    manifest["artifacts"] = {
        "features": "features.parquet",
        "labels": "labels.parquet",
    }
    save_json_atomic(args.output_root / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "contract": manifest["contract"],
                "rows": manifest["rows"],
                "features": len(manifest["feature_columns"]),
                "official_test_queried": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

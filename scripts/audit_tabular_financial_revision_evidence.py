#!/usr/bin/env python
"""Audit archived financial revision evidence and changed values."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.prediction.tabular.financial_revision_evidence import (
    audit_financial_revision_evidence,
    load_financial_revision_evidence_contract,
)
from src.prediction.tabular.point_in_time_sources import (
    load_point_in_time_source_contract,
    save_json_atomic,
)


def main() -> None:
    root = PROJECT_ROOT / "data" / "tabular" / "financial_revision_evidence_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--plan", type=Path, default=root / "plan.json")
    parser.add_argument("--output", type=Path, default=root / "audit.report.json")
    args = parser.parse_args()
    report = audit_financial_revision_evidence(
        plan_path=args.plan,
        root=args.root,
        state_path=args.root / "state.json",
        contract=load_financial_revision_evidence_contract(),
        source_contract=load_point_in_time_source_contract(),
    )
    save_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

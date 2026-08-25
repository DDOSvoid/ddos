#!/usr/bin/env python
"""Archive PDFs for the deterministic financial revision evidence plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.prediction.tabular.financial_revision_evidence import (
    archive_financial_revision_evidence,
    load_financial_revision_evidence_contract,
)


def main() -> None:
    root = PROJECT_ROOT / "data" / "tabular" / "financial_revision_evidence_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--plan", type=Path, default=root / "plan.json")
    parser.add_argument("--rate-limit", type=int, default=30)
    args = parser.parse_args()
    state = archive_financial_revision_evidence(
        plan_path=args.plan,
        output_root=args.root,
        state_path=args.root / "state.json",
        contract=load_financial_revision_evidence_contract(),
        rate_limit_per_minute=max(args.rate_limit, 1),
    )
    print(
        json.dumps(
            {
                "completed": len(state.get("completed", {})),
                "failures": len(state.get("failures", {})),
                "official_test_queried": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Apply frozen model gates to train-only tabular OOF results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.prediction.tabular.experiment import review_internal_experiment


def main() -> None:
    root = PROJECT_ROOT / "models" / "tabular" / "research_v1"
    parser = argparse.ArgumentParser(description="Review train-only tabular OOF gates")
    parser.add_argument("--report", type=Path, default=root / "report.json")
    parser.add_argument("--output", type=Path, default=root / "selection_review.json")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    review = review_internal_experiment(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(review, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Freeze the train-only stock universe after announcement metadata completes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.prediction.data_download import build_event_train_universe


def main() -> None:
    root = PROJECT_ROOT / "data" / "backfills" / "event_ranking_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "data" / "ddos.db")
    parser.add_argument(
        "--metadata-state",
        type=Path,
        default=root / "full_announcements_train_2023_2024.json",
    )
    parser.add_argument("--output", type=Path, default=root / "train_universe.json")
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 12, 31))
    args = parser.parse_args()
    state = build_event_train_universe(
        database_path=args.database.resolve(),
        metadata_state_path=args.metadata_state.resolve(),
        output_path=args.output.resolve(),
        train_start=args.start_date,
        train_end=args.end_date,
    )
    print(
        json.dumps(
            {
                "codes": state["selection"]["codes"],
                "announcement_count": state["announcement_count"],
                "official_test_queried": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

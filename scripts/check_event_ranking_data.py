#!/usr/bin/env python
"""Check data-source readiness without printing or storing credential values."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prediction.data_download import (
    build_download_readiness,
    load_data_download_contract,
    write_download_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write the resumable download manifest under data/backfills/event_ranking_v1",
    )
    args = parser.parse_args()
    contract = load_data_download_contract()
    report = build_download_readiness(contract)
    if args.write_manifest:
        write_download_manifest(contract.manifest_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

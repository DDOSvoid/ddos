#!/usr/bin/env python
"""Build the company-day union of available specialist OOF predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prediction.fusion.contract import load_expert_fusion_contract
from src.prediction.fusion.schema import build_joint_expert_frame


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("text", "tabular", "timeseries"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: getattr(args, name)
        for name in ("text", "tabular", "timeseries")
        if getattr(args, name) is not None
    }
    if not paths:
        parser.error("at least one expert OOF path is required")
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    contract = load_expert_fusion_contract()
    output = build_joint_expert_frame(frames, contract=contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    output.to_parquet(temporary, index=False)
    temporary.replace(args.output)
    manifest = {
        "contract": "joint-expert-component-oof-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "fusion_contract_sha256": contract.source_sha256,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
        },
        "rows": len(output),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "official_test_read": False,
        "quarantine_read": False,
        "strict_oos_2026_read": False,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

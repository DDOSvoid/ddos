#!/usr/bin/env python
"""Audit the 2026 seal without selecting realized outcome values."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.prediction.strict_oos import audit_current_database, load_strict_oos_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "data" / "ddos.db")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "governance" / "strict_oos_2026.audit.json",
    )
    args = parser.parse_args()
    contract = load_strict_oos_contract()
    store = audit_current_database(args.database.resolve())
    report = {
        "contract": contract.contract,
        "contract_sha256": contract.source_sha256,
        "audited_at": datetime.now(UTC).isoformat(),
        "audit_reads_outcome_values": False,
        "store": store,
        "legacy_repository_pristine": False,
        "eligible_as_new_architecture_strict_outcome_holdout": bool(
            store["current_architecture_store_clean"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

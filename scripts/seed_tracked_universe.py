#!/usr/bin/env python
"""Seed a small public-source smoke universe without requiring credentials."""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime
from pathlib import Path

import yaml
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.database.engine import get_engine, init_db
from src.database.repository import CompanyRepository


def main() -> None:
    path = PROJECT_ROOT / "config" / "tracked_companies.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    codes = [str(value) for value in raw.get("tracked_codes", [])]
    names = {
        str(code): str(name)
        for code, name in (raw.get("tracked_company_metadata") or {}).items()
    }
    init_db()
    with Session(get_engine()) as session:
        for code in codes:
            CompanyRepository.upsert(
                session,
                code,
                stock_name=names.get(code, code),
                exchange=(
                    "SSE"
                    if code.endswith(".SH")
                    else "BSE"
                    if code.endswith(".BJ")
                    else "SZSE"
                ),
                is_tracked=True,
                tracked_since=date.today(),
                universe_source="config_smoke_only_v1",
                universe_as_of=datetime.now(UTC),
            )
        session.commit()
    print(f"seeded_or_updated={len(codes)} source=config_smoke_only_v1")


if __name__ == "__main__":
    main()

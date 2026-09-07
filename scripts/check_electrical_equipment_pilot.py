#!/usr/bin/env python
"""Report read-only progress for the electrical-equipment ranking pilot."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT

PILOT_ROOT = PROJECT_ROOT / "data" / "backfills" / "event_ranking_pilot_electrical_v1"
VALIDATION_ROOT = PROJECT_ROOT / "data" / "validation" / "event_ranking_pilot_electrical_v1"
TABULAR_ROOT = PROJECT_ROOT / "data" / "tabular" / "event_ranking_pilot_electrical_v1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _state_summary(path: Path, finished_key: str) -> dict[str, object]:
    state = _load(path)
    return {
        "exists": path.exists(),
        "completed": len(state.get("completed") or {}),
        "failures": len(state.get("failures") or {}),
        "finished": bool(state.get(finished_key)),
        "updated_at": state.get("updated_at") or state.get("updated_at_utc"),
    }


def _archive_count(
    connection: sqlite3.Connection, announcement_ids: list[str], pdf_only: bool = False
) -> int:
    if not announcement_ids:
        return 0
    placeholders = ",".join("?" for _ in announcement_ids)
    pdf_clause = "AND s.pdf_fetch_status = 'archived'" if pdf_only else ""
    return int(
        connection.execute(
            "SELECT COUNT(DISTINCT s.announcement_id) "
            "FROM announcement_source_archives s "
            "JOIN announcements a ON a.id = s.announcement_id "
            f"WHERE a.announcement_id IN ({placeholders}) "
            "AND s.dataset_role = 'train' AND s.content_complete = 1 "
            f"{pdf_clause}",
            announcement_ids,
        ).fetchone()[0]
    )


def build_status() -> dict[str, object]:
    body_plan = _load(PILOT_ROOT / "body_plan.json")
    pdf_plan = _load(PILOT_ROOT / "pdf_plan.json")
    body_ids = [str(value) for value in body_plan.get("announcement_ids") or []]
    pdf_ids = [str(value) for value in pdf_plan.get("announcement_ids") or []]
    database = PROJECT_ROOT / "data" / "ddos.db"
    with sqlite3.connect(database) as connection:
        body_archived = _archive_count(connection, body_ids)
        pdf_archived = _archive_count(connection, pdf_ids, pdf_only=True)
        target_count = 0
        if body_ids:
            placeholders = ",".join("?" for _ in body_ids)
            target_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM announcement_market_targets t "
                    "JOIN announcements a ON a.id = t.announcement_id "
                    f"WHERE a.announcement_id IN ({placeholders}) "
                    "AND t.dataset_role = 'train'",
                    body_ids,
                ).fetchone()[0]
            )
        prices_2026 = int(
            connection.execute(
                "SELECT COUNT(*) FROM daily_prices WHERE trade_date >= '2026-01-01'"
            ).fetchone()[0]
        )
        targets_2026 = int(
            connection.execute(
                "SELECT COUNT(*) FROM announcement_market_targets t "
                "JOIN announcements a ON a.id = t.announcement_id "
                "WHERE a.published_date >= '2026-01-01'"
            ).fetchone()[0]
        )
    queue = _load(PILOT_ROOT / "data_queue.state.json")
    body_state = _load(PILOT_ROOT / "announcement_bodies.state.json")
    usage = shutil.disk_usage(PROJECT_ROOT.drive + "\\")
    return {
        "checked_at": datetime.now(UTC).isoformat(),
        "contract": body_plan.get("pilot_contract"),
        "queue_status": queue.get("status", "not_started"),
        "queue_stage": queue.get("stages") or {},
        "body": {
            "planned": len(body_ids),
            "archived": body_archived,
            "remaining": len(body_ids) - body_archived,
            "coverage": round(body_archived / len(body_ids), 6) if body_ids else 0.0,
            "cursor_processed": int(body_state.get("processed") or 0),
            "cursor_failures": len(body_state.get("failures") or []),
            "updated_at": body_state.get("updated_at"),
        },
        "pdf_audit": {
            "planned": len(pdf_ids),
            "archived": pdf_archived,
            "remaining": len(pdf_ids) - pdf_archived,
        },
        "daily_prices": _state_summary(
            PILOT_ROOT / "market_prices.state.json", "finished_at"
        ),
        "train_targets": {"expected_maximum": len(body_ids) * 3, "rows": target_count},
        "daily_basic": _state_summary(
            TABULAR_ROOT / "daily_basic" / "state.json", "finished_at_utc"
        ),
        "point_in_time_company_industry": _state_summary(
            TABULAR_ROOT / "point_in_time_sources" / "company_industry.state.json",
            "finished_at_utc",
        ),
        "point_in_time_fundamentals": _state_summary(
            TABULAR_ROOT / "point_in_time_sources" / "fundamentals.state.json",
            "finished_at_utc",
        ),
        "strict_oos_2026": {
            "daily_price_rows": prices_2026,
            "market_target_rows": targets_2026,
            "clean": prices_2026 == 0 and targets_2026 == 0,
        },
        "disk_free_gib": round(usage.free / (1024**3), 2),
    }


def main() -> None:
    print(json.dumps(build_status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

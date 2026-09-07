"""Auditable readiness and state manifest for event-ranking data acquisition."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT

CONTRACT = "causal-event-ranking-data-download-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class DataDownloadContract:
    source_path: Path
    source_sha256: str
    contract: str
    universe_mode: str
    list_statuses: tuple[str, ...]
    train_start: date
    train_end: date
    prehistory_start: date
    download_order: tuple[str, ...]
    manifest_path: Path


def load_data_download_contract(path: Path | None = None) -> DataDownloadContract:
    source = path or PROJECT_ROOT / "config" / "event_ranking_data_v1.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    universe = raw.get("universe") or {}
    dates = raw.get("dates") or {}
    artifacts = raw.get("artifacts") or {}
    contract = DataDownloadContract(
        source_path=source,
        source_sha256=_sha256(source),
        contract=str(raw.get("contract") or ""),
        universe_mode=str(universe.get("mode") or ""),
        list_statuses=tuple(str(value) for value in universe.get("include_list_status", [])),
        train_start=date.fromisoformat(str(dates.get("train_start"))),
        train_end=date.fromisoformat(str(dates.get("train_end"))),
        prehistory_start=date.fromisoformat(str(dates.get("market_prehistory_start"))),
        download_order=tuple(str(value) for value in raw.get("download_order", [])),
        manifest_path=PROJECT_ROOT / str(artifacts.get("manifest")),
    )
    if contract.contract != CONTRACT:
        raise ValueError(f"unexpected data download contract: {contract.contract}")
    if contract.universe_mode != "all_a_share_stocks_with_eligible_announcements":
        raise ValueError("event-ranking data must cover the A-share announcement universe")
    if set(contract.list_statuses) != {"L", "D", "P"}:
        raise ValueError("stock universe must include L/D/P to avoid survivorship bias")
    if not contract.prehistory_start < contract.train_start <= contract.train_end:
        raise ValueError("download date boundaries are invalid")
    return contract


def build_download_readiness(contract: DataDownloadContract) -> dict[str, object]:
    token_ready = bool(os.getenv("TUSHARE_TOKEN"))
    database_path = PROJECT_ROOT / "data" / "ddos.db"
    company_count = None
    full_universe_company_count = None
    announcement_count = None
    daily_price_row_count = None
    if database_path.exists():
        import sqlite3

        with sqlite3.connect(database_path) as connection:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "companies" in tables:
                company_count = int(
                    connection.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
                )
                columns = {row[1] for row in connection.execute("PRAGMA table_info(companies)")}
                if "universe_source" in columns:
                    full_universe_company_count = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM companies "
                            "WHERE universe_source = 'tushare_stock_basic_ldp_v1'"
                        ).fetchone()[0]
                    )
            if "announcements" in tables:
                announcement_count = int(
                    connection.execute("SELECT COUNT(*) FROM announcements").fetchone()[0]
                )
            if "daily_prices" in tables:
                daily_price_row_count = int(
                    connection.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
                )
    universe_ready = bool(full_universe_company_count)
    announcement_state_path = (
        contract.manifest_path.parent / "full_announcements_train_2023_2024.json"
    )
    announcement_progress = None
    announcement_ready = False
    if announcement_state_path.exists():
        state = json.loads(announcement_state_path.read_text(encoding="utf-8"))
        progress = state.get("progress") or {}
        completed = int(progress.get("completed") or 0)
        total = int(progress.get("total") or 0)
        announcement_ready = total > 0 and completed == total and bool(state.get("finished_at"))
        announcement_progress = {
            "completed_segments": completed,
            "total_segments": total,
            "api_items": int((state.get("totals") or {}).get("api_items") or 0),
            "matched_items": int((state.get("totals") or {}).get("matched_items") or 0),
            "unmatched_items": int((state.get("totals") or {}).get("unmatched_items") or 0),
        }
    stages = [
        {
            "name": "stock_universe",
            "ready": universe_ready,
            "blocker": (
                None
                if universe_ready
                else "missing_local_TUSHARE_TOKEN"
                if not token_ready
                else "stock_universe_not_downloaded"
            ),
        },
        {
            "name": "announcement_metadata",
            "ready": announcement_ready,
            "blocker": (
                None
                if announcement_ready
                else "download_in_progress"
                if announcement_progress
                else "download_not_started"
            ),
            "progress": announcement_progress,
        },
        {
            "name": "stock_and_benchmark_daily_prices",
            "ready": token_ready and bool(company_count),
            "blocker": (
                None
                if token_ready and company_count
                else "requires_local_TUSHARE_TOKEN_and_seeded_universe"
            ),
        },
    ]
    return {
        "contract": CONTRACT,
        "contract_sha256": contract.source_sha256,
        "checked_at": datetime.now(UTC).isoformat(),
        "credentials": {"tushare_token_present": token_ready},
        "local_state": {
            "database_exists": database_path.exists(),
            "company_count": company_count,
            "full_universe_company_count": full_universe_company_count,
            "announcement_count": announcement_count,
            "daily_price_row_count": daily_price_row_count,
        },
        "stages": stages,
        "download_order": list(contract.download_order),
        "official_test_labels_read": False,
    }


def write_download_manifest(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_train_universe_state(path: Path) -> list[str]:
    """Load a frozen train-only code universe shared by all download stages."""
    state = json.loads(path.read_text(encoding="utf-8"))
    selection = state.get("selection") or {}
    if selection.get("dataset_role") != "train":
        raise PermissionError("event universe is not train-only")
    if state.get("official_test_queried") is not False:
        raise PermissionError("event universe does not prove official-test isolation")
    codes = sorted(str(value) for value in state.get("codes", []))
    if not codes:
        codes = sorted({str(key).split(":", 1)[0] for key in state.get("completed", {})})
    expected_hash = hashlib.sha256("\n".join(codes).encode()).hexdigest()
    if len(codes) != selection.get("codes"):
        raise ValueError("event universe count differs from its selection")
    if expected_hash != selection.get("codes_sha256"):
        raise ValueError("event universe hash differs from its selection")
    return codes


def build_event_train_universe(
    *,
    database_path: Path,
    metadata_state_path: Path,
    output_path: Path,
    train_start: date,
    train_end: date,
    require_metadata_complete: bool = True,
) -> dict[str, object]:
    """Freeze stocks that actually have announcements in the training period."""
    metadata = json.loads(metadata_state_path.read_text(encoding="utf-8"))
    progress = metadata.get("progress") or {}
    if require_metadata_complete and (
        int(progress.get("completed") or 0) != int(progress.get("total") or 0)
        or not metadata.get("finished_at")
    ):
        raise ValueError("announcement metadata download is not complete")
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT DISTINCT c.stock_code "
            "FROM companies c JOIN announcements a ON a.company_id = c.id "
            "WHERE a.published_date BETWEEN ? AND ? ORDER BY c.stock_code",
            (train_start.isoformat(), train_end.isoformat()),
        ).fetchall()
        announcement_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM announcements WHERE published_date BETWEEN ? AND ?",
                (train_start.isoformat(), train_end.isoformat()),
            ).fetchone()[0]
        )
    codes = [str(row[0]) for row in rows]
    codes_sha256 = hashlib.sha256("\n".join(codes).encode()).hexdigest()
    metadata_sha256 = _sha256(metadata_state_path)
    state: dict[str, object] = {
        "contract": "event-ranking-train-universe-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "official_test_queried": False,
        "selection": {
            "dataset_role": "train",
            "start": train_start.isoformat(),
            "end": train_end.isoformat(),
            "codes": len(codes),
            "codes_sha256": codes_sha256,
            "source": "train_announcement_company_days",
        },
        "codes": codes,
        "announcement_count": announcement_count,
        "announcement_metadata_state": str(metadata_state_path),
        "announcement_metadata_state_sha256": metadata_sha256,
        "completed": {
            f"{code}:{train_start.isoformat()}:{train_end.isoformat()}": {
                "status": "eligible_train_announcement_company"
            }
            for code in codes
        },
    }
    write_download_manifest(output_path, state)
    return state

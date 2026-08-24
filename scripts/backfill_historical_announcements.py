#!/usr/bin/env python
"""Backfill real historical announcement metadata in auditable monthly batches."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, config
from src.database.engine import get_engine
from src.database.models import Company
from src.database.repository import AnnouncementRepository, CompanyRepository
from src.pipeline.fetcher import EastmoneyClient, Fetcher

CONTRACT = "historical-announcement-backfill-v1"


def month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    if end < start:
        raise ValueError("end date is earlier than start date")
    ranges = []
    current = start
    while current <= end:
        next_month = (
            date(current.year + 1, 1, 1)
            if current.month == 12
            else date(current.year, current.month + 1, 1)
        )
        period_end = min(end, next_month - timedelta(days=1))
        ranges.append((current, period_end))
        current = period_end + timedelta(days=1)
    return ranges


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _load_state(path: Path, start: date, end: date) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as file:
            state = json.load(file)
        if state.get("contract") != CONTRACT:
            raise ValueError(f"unknown state contract: {state.get('contract')}")
        if state.get("start_date") != start.isoformat() or state.get(
            "end_date"
        ) != end.isoformat():
            raise ValueError("state date range does not match command")
        return state
    return {
        "contract": CONTRACT,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "created_at": datetime.now(UTC).isoformat(),
        "metadata_only": True,
        "completed_segments": {},
        "totals": {
            "api_items": 0,
            "matched_items": 0,
            "unmatched_items": 0,
            "new_announcements": 0,
        },
    }


def _records_from_items(
    items: list[dict],
    company_by_short_code: dict[str, Company],
) -> tuple[list[dict], int]:
    records = []
    unmatched = 0
    seen = set()
    for item in items:
        announcement_id = str(item.get("art_code") or "").strip()
        if not announcement_id or announcement_id in seen:
            continue
        seen.add(announcement_id)
        company = None
        for code_item in item.get("codes") or []:
            company = company_by_short_code.get(str(code_item.get("stock_code") or ""))
            if company is not None:
                break
        if company is None:
            unmatched += 1
            continue
        published_date = Fetcher._parse_em_date(
            str(item.get("notice_date") or item.get("display_time") or "")
        )
        if published_date is None:
            unmatched += 1
            continue
        records.append(
            {
                "company_id": company.id,
                "announcement_id": announcement_id,
                "title": str(item.get("title") or item.get("title_ch") or "").strip(),
                "full_text": str(item.get("notice_content") or "").strip(),
                "published_date": published_date,
                "source_url": announcement_id,
                "raw_response": json.dumps(item, ensure_ascii=False),
                "processing_status": "fetched",
            }
        )
    return records, unmatched


def backfill(
    *,
    start: date,
    end: date,
    batch_size: int,
    max_pages: int,
    rate_limit: int,
    state_path: Path,
    limit_companies: int,
) -> dict:
    config.eastmoney.rate_limit_per_minute = rate_limit
    client = EastmoneyClient()
    engine = get_engine()
    state = _load_state(state_path, start, end)
    completed = state["completed_segments"]

    with Session(engine) as session:
        companies = CompanyRepository.get_tracked(session)
        companies.sort(key=lambda item: item.stock_code)
        if limit_companies > 0:
            companies = companies[:limit_companies]
        company_by_short_code = {
            company.stock_code.split(".")[0]: company for company in companies
        }

        batches = [
            companies[offset : offset + batch_size]
            for offset in range(0, len(companies), batch_size)
        ]
        periods = month_ranges(start, end)
        total_segments = len(batches) * len(periods)
        segment_number = 0
        for period_start, period_end in periods:
            for batch_index, batch in enumerate(batches):
                segment_number += 1
                segment_key = (
                    f"{period_start.isoformat()}_{period_end.isoformat()}_"
                    f"batch-{batch_index:03d}"
                )
                if segment_key in completed:
                    continue
                short_codes = [company.stock_code.split(".")[0] for company in batch]
                items = client.fetch_all_announcements(
                    stock_code=",".join(short_codes),
                    start_date=period_start.isoformat(),
                    end_date=period_end.isoformat(),
                    max_pages=max_pages,
                )
                records, unmatched = _records_from_items(items, company_by_short_code)
                new_count = AnnouncementRepository.bulk_upsert(session, records)
                session.commit()

                segment = {
                    "completed_at": datetime.now(UTC).isoformat(),
                    "company_count": len(batch),
                    "api_items": len(items),
                    "matched_items": len(records),
                    "unmatched_items": unmatched,
                    "new_announcements": new_count,
                }
                completed[segment_key] = segment
                for key in state["totals"]:
                    state["totals"][key] += segment.get(key, 0)
                state["updated_at"] = datetime.now(UTC).isoformat()
                state["progress"] = {
                    "completed": len(completed),
                    "total": total_segments,
                }
                _write_state(state_path, state)
                print(
                    f"segment {segment_number}/{total_segments} "
                    f"{period_start}..{period_end} batch={batch_index} "
                    f"items={len(items)} new={new_count}",
                    flush=True,
                )

    state["finished_at"] = datetime.now(UTC).isoformat()
    _write_state(state_path, state)
    print(json.dumps(state["totals"], ensure_ascii=False, indent=2))
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="分月批量回填真实历史公告元数据")
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--rate-limit", type=int, default=120)
    parser.add_argument("--limit-companies", type=int, default=0)
    parser.add_argument(
        "--state",
        type=Path,
        default=PROJECT_ROOT / "data" / "backfills" / "historical_announcements.json",
    )
    args = parser.parse_args()
    backfill(
        start=args.start_date,
        end=args.end_date,
        batch_size=max(args.batch_size, 1),
        max_pages=max(args.max_pages, 1),
        rate_limit=max(args.rate_limit, 1),
        state_path=args.state.resolve(),
        limit_companies=max(args.limit_companies, 0),
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Triangulate candidate announcement times without reading market outcomes."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session

from src.config import PROJECT_ROOT
from src.database.engine import get_engine
from src.database.models import Announcement
from src.pipeline.fetcher import EastmoneyClient
from src.prediction.splits import load_prediction_split_contract

SHANGHAI = ZoneInfo("Asia/Shanghai")
CONTENT_URL = "https://np-cnotice-stock.eastmoney.com/api/content/ann"


def _parse_time(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _attachment_epoch(value: object) -> datetime | None:
    match = re.search(r"[?&](\d{13})(?:\.pdf)?(?:$|&)", str(value or ""))
    if not match:
        return None
    return datetime.fromtimestamp(int(match.group(1)) / 1000, tz=UTC).astimezone(
        SHANGHAI
    ).replace(tzinfo=None)


def _art_code_date(value: object):
    match = re.match(r"AN(\d{4})(\d{2})(\d{2})", str(value or ""))
    if not match:
        return None
    return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3))).date()


def _select_evenly(rows: list, limit: int) -> list:
    if limit <= 0 or not rows:
        return []
    if len(rows) <= limit:
        return rows
    indices = [round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)]
    return [rows[index] for index in indices]


def audit(*, limit: int, output: Path) -> dict:
    split = load_prediction_split_contract()
    with Session(get_engine()) as session:
        rows = (
            session.query(
                Announcement.announcement_id,
                Announcement.published_date,
                Announcement.title,
                Announcement.raw_response,
            )
            .filter(Announcement.published_date.between(split.train.start, split.train.end))
            .filter(Announcement.announcement_id.isnot(None))
            .filter(Announcement.raw_response.isnot(None))
            .order_by(Announcement.published_date, Announcement.id)
            .all()
        )
    selected = _select_evenly(rows, limit)
    client = EastmoneyClient()
    counts = Counter()
    attachment_deltas = []
    samples = []
    for index, row in enumerate(selected, 1):
        raw = json.loads(row.raw_response)
        list_time = _parse_time(raw.get("eiTime"))
        client._rate_limit()
        response = client.session.get(
            CONTENT_URL,
            params={
                "art_code": row.announcement_id,
                "client_source": "web",
                "page_index": 1,
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json().get("data") or {}
        content_time = _parse_time(content.get("eitime"))
        attachment = content.get("attach_url_web") or content.get("attach_url")
        attachment_time = _attachment_epoch(attachment)
        art_date = _art_code_date(row.announcement_id)
        list_content_match = (
            list_time is not None
            and content_time is not None
            and list_time == content_time
        )
        art_date_match = list_time is not None and art_date == list_time.date()
        title_match = "".join(row.title.split()) == "".join(
            str(content.get("notice_title") or "").split()
        )
        notice_date_match = (
            str(content.get("notice_date") or "")[:10]
            == row.published_date.isoformat()
        )
        attachment_delta = None
        if list_time is not None and attachment_time is not None:
            attachment_delta = (attachment_time - list_time).total_seconds()
            attachment_deltas.append(attachment_delta)
        previous_evening = (
            list_time is not None
            and list_time.date() < row.published_date
            and list_time.time() >= time(15, 0)
        )
        counts["sampled"] += 1
        counts["response_success"] += int(bool(content))
        counts["list_content_time_exact_match"] += int(list_content_match)
        counts["art_code_date_matches_eitime_date"] += int(art_date_match)
        counts["title_match"] += int(title_match)
        counts["notice_date_match"] += int(notice_date_match)
        counts["attachment_epoch_available"] += int(attachment_time is not None)
        counts["eitime_previous_evening_of_notice_date"] += int(previous_evening)
        samples.append(
            {
                "announcement_id": row.announcement_id,
                "published_date": row.published_date.isoformat(),
                "list_content_time_exact_match": list_content_match,
                "art_code_date_matches_eitime_date": art_date_match,
                "title_match": title_match,
                "notice_date_match": notice_date_match,
                "eitime_previous_evening_of_notice_date": previous_evening,
                "attachment_epoch_minus_eitime_seconds": attachment_delta,
            }
        )
        if index % 10 == 0 or index == len(selected):
            print(f"timestamp audit progress {index}/{len(selected)}", flush=True)

    report = {
        "contract": "announcement-candidate-time-semantics-audit-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_role": "train",
        "market_target_table_queried": False,
        "market_direction_labels_read": False,
        "counts": dict(counts),
        "attachment_epoch_minus_eitime_seconds": {
            "minimum": min(attachment_deltas) if attachment_deltas else None,
            "median": statistics.median(attachment_deltas) if attachment_deltas else None,
            "maximum": max(attachment_deltas) if attachment_deltas else None,
        },
        "samples": samples,
        "interpretation": [
            "Exact agreement among Eastmoney fields supports internal consistency only.",
            "It does not prove that eiTime is the exchange's legally authoritative publication timestamp.",
            "Until authoritative semantics are established, store it as source_eitime/candidate precision.",
            "Previous-evening candidates indicate the existing date-only target may enter one session late; targets must not be changed until a versioned timing contract is approved.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({key: report[key] for key in ("contract", "counts", "attachment_epoch_minus_eitime_seconds")}, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "validation"
            / "announcement_candidate_time_semantics_20260821.json"
        ),
    )
    args = parser.parse_args()
    audit(limit=max(args.limit, 0), output=args.output.resolve())


if __name__ == "__main__":
    main()

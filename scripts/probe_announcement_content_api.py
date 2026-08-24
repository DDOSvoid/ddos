#!/usr/bin/env python
"""Read-only probe of archived announcement content; never writes source rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session

from src.config import PROJECT_ROOT
from src.database.engine import get_engine
from src.database.models import Announcement
from src.pipeline.fetcher import EastmoneyClient
from src.prediction.development_contract import load_causal_development_contract
from src.prediction.splits import load_prediction_split_contract


def select_evenly(rows: list, limit: int) -> list:
    if limit <= 0 or not rows:
        return []
    if len(rows) <= limit:
        return rows
    indices = [round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)]
    return [rows[index] for index in indices]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized_text(value: object) -> str:
    return "".join(str(value or "").split())


def _candidate_time(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def probe(*, limit: int, output: Path) -> dict:
    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
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
            .order_by(Announcement.published_date, Announcement.id)
            .all()
        )
    selected = select_evenly(rows, limit)
    client = EastmoneyClient()
    samples = []
    for row in selected:
        data = client.fetch_announcement_content(row.announcement_id)
        content = str(data.get("notice_content") or "")
        attachment = str(
            data.get("attach_url_web") or data.get("attach_url") or ""
        )
        list_raw = json.loads(row.raw_response) if row.raw_response else {}
        list_time = _candidate_time(list_raw.get("eiTime"))
        content_time = _candidate_time(data.get("eitime"))
        pdf_bytes = b""
        pdf_content_type = None
        pdf_error = None
        if attachment:
            try:
                response = client.session.get(attachment, timeout=60)
                response.raise_for_status()
                pdf_bytes = response.content
                pdf_content_type = response.headers.get("Content-Type")
            except Exception as exc:
                pdf_error = type(exc).__name__
        samples.append(
            {
                "announcement_id": row.announcement_id,
                "published_date": row.published_date.isoformat(),
                "title_sha256": _sha256_text(row.title),
                "response_nonempty": bool(data),
                "response_keys": sorted(data),
                "art_code_matches": data.get("art_code") == row.announcement_id,
                "title_matches_after_whitespace_normalization": (
                    _normalized_text(data.get("notice_title"))
                    == _normalized_text(row.title)
                ),
                "notice_date_matches_published_date": (
                    str(data.get("notice_date") or "")[:10]
                    == row.published_date.isoformat()
                ),
                "list_and_content_eitime_match_to_second": (
                    list_time is not None
                    and content_time is not None
                    and list_time == content_time
                ),
                "content_pages_expected": data.get("_content_pages_expected"),
                "content_pages_fetched": data.get("_content_pages_fetched"),
                "content_complete": data.get("_content_complete", False),
                "content_chars": len(content),
                "content_sha256": _sha256_text(content) if content else None,
                "attachment_available": bool(attachment),
                "attachment_is_http_url": attachment.startswith(("http://", "https://")),
                "attachment_host": urlparse(attachment).hostname if attachment else None,
                "pdf_download_error": pdf_error,
                "pdf_content_type": pdf_content_type,
                "pdf_bytes": len(pdf_bytes),
                "pdf_magic_valid": pdf_bytes.startswith(b"%PDF"),
                "pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest() if pdf_bytes else None,
            }
        )
    successful = [item for item in samples if item["response_nonempty"]]
    with_content = [item for item in samples if item["content_chars"] > 0]
    with_attachment = [item for item in samples if item["attachment_available"]]
    complete = [item for item in samples if item["content_complete"]]
    consistent_identity = [
        item
        for item in samples
        if item["art_code_matches"]
        and item["title_matches_after_whitespace_normalization"]
        and item["notice_date_matches_published_date"]
    ]
    timestamp_matches = [
        item for item in samples if item["list_and_content_eitime_match_to_second"]
    ]
    valid_pdfs = [item for item in samples if item["pdf_magic_valid"]]
    report = {
        "contract": "announcement-content-readonly-probe-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "development_contract": development.contract,
        "development_contract_sha256": development.source_sha256,
        "dataset_role": "train",
        "market_target_table_queried": False,
        "market_direction_labels_read": False,
        "database_rows_modified": False,
        "requested_samples": limit,
        "sampled": len(samples),
        "response_success": len(successful),
        "content_success": len(with_content),
        "attachment_success": len(with_attachment),
        "complete_content_success": len(complete),
        "identity_consistency_success": len(consistent_identity),
        "candidate_timestamp_consistency_success": len(timestamp_matches),
        "valid_pdf_download_success": len(valid_pdfs),
        "samples": samples,
        "warning": (
            "A successful archive response proves retrievability only; immutable "
            "versioning and publication-time semantics still require validation."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "validation"
            / "announcement_content_probe_20260821.json"
        ),
    )
    args = parser.parse_args()
    probe(limit=max(args.limit, 0), output=args.output.resolve())


if __name__ == "__main__":
    main()

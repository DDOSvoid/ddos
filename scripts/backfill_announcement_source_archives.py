#!/usr/bin/env python
"""Resumable train-only backfill into immutable announcement source archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import or_
from sqlalchemy.orm import Session

from src.config import PROJECT_ROOT
from src.database.engine import get_engine, init_db
from src.database.models import (
    Announcement,
    AnnouncementSourceArchive,
    Classification,
)
from src.pipeline.announcement_content import assemble_pdf_extracted_content
from src.pipeline.fetcher import EastmoneyClient
from src.pipeline.source_archive import (
    PdfEvidence,
    archive_announcement_content,
    archive_pdf_bytes,
)
from src.prediction.development_contract import load_causal_development_contract
from src.prediction.splits import load_prediction_split_contract


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"last_announcement_db_id": 0, "processed": 0, "failures": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _fetch_pdf(client: EastmoneyClient, attachment_url: str | None) -> PdfEvidence:
    if not attachment_url:
        return PdfEvidence(status="missing_attachment")
    response = client.session.get(attachment_url, timeout=90)
    response.raise_for_status()
    return archive_pdf_bytes(
        response.content,
        content_type=response.headers.get("Content-Type"),
    )


def _fetch_static_pdf_content(
    client: EastmoneyClient,
    announcement: Announcement,
) -> tuple[dict, PdfEvidence]:
    art_code = str(announcement.announcement_id)
    attachment_url = f"https://pdf.dfcfw.com/pdf/H2_{art_code}_1.pdf"
    response = None
    for attempt in range(1, 5):
        client._rate_limit()  # Share the same polite global request cadence.
        try:
            response = client.session.get(attachment_url, timeout=90)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError("static attachment response is not a PDF")
            break
        except Exception:
            if attempt >= 4:
                raise
            time.sleep(10 * attempt)
    assert response is not None
    content = assemble_pdf_extracted_content(
        response.content,
        art_code=art_code,
        title=announcement.title,
        notice_date=announcement.published_date.isoformat(),
        source_eitime=(
            json.loads(announcement.raw_response or "{}").get("eiTime")
        ),
        attachment_url=attachment_url,
    )
    pdf = archive_pdf_bytes(
        response.content,
        content_type=response.headers.get("Content-Type"),
    )
    return content, pdf


def backfill(
    *,
    announcement_ids: list[str],
    limit: int,
    pdf_mode: str,
    only_accepted: bool,
    relevances: list[str],
    rate_limit_per_minute: int,
    refresh: bool,
    state_path: Path,
    report_path: Path,
    consecutive_failure_limit: int,
    content_source: str = "api",
) -> dict:
    init_db()
    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
    state = _load_state(state_path)
    selection = {
        "dataset_role": "train",
        "only_accepted": only_accepted,
        "relevances": sorted(set(relevances)),
    }
    if announcement_ids:
        canonical_ids = "\n".join(sorted(set(announcement_ids)))
        selection["explicit_announcement_ids_sha256"] = hashlib.sha256(
            canonical_ids.encode()
        ).hexdigest()
    if content_source != "api":
        selection["content_source"] = content_source
    if content_source == "pdf" and pdf_mode != "archive":
        raise ValueError("PDF content source requires pdf_mode=archive")
    if state.get("selection") not in (None, selection):
        raise ValueError(
            "state-file selection does not match this run; use a separate state file"
        )
    engine = get_engine()
    with Session(engine) as session:
        query = session.query(Announcement).filter(
            Announcement.published_date.between(split.train.start, split.train.end)
        )
        if announcement_ids:
            query = query.filter(Announcement.announcement_id.in_(announcement_ids))
        else:
            failed_db_ids = [
                int(item["announcement_db_id"])
                for item in state.get("failures", [])
                if item.get("announcement_db_id") is not None
            ]
            query = query.filter(
                or_(
                    Announcement.id > int(state.get("last_announcement_db_id", 0)),
                    Announcement.id.in_(failed_db_ids),
                )
            )
        if only_accepted or relevances:
            query = query.join(
                Classification,
                Classification.announcement_id == Announcement.id,
            )
        if only_accepted:
            query = query.filter(Classification.needs_review.is_(False))
        if relevances:
            query = query.filter(Classification.relevance.in_(relevances))
        query = query.order_by(Announcement.id)
        if limit > 0:
            query = query.limit(limit)
        rows = query.all()

        archived_query = session.query(
            AnnouncementSourceArchive.announcement_id,
            AnnouncementSourceArchive.pdf_fetch_status,
        ).filter(
            AnnouncementSourceArchive.development_contract_sha256
            == development.source_sha256
        )
        archived = {}
        for announcement_id, status in archived_query:
            archived.setdefault(announcement_id, set()).add(status)

    client = EastmoneyClient(
        rate_limit_per_minute=rate_limit_per_minute,
        content_max_retries=4,
        content_retry_backoff_seconds=10,
    )
    counts = Counter()
    failures = list(state.get("failures", []))
    consecutive_failures = 0
    started_at = datetime.now(UTC)
    for index, announcement in enumerate(rows, 1):
        required_status_satisfied = (
            announcement.id in archived
            and (
                pdf_mode == "none"
                or "archived" in archived[announcement.id]
                or "missing_attachment" in archived[announcement.id]
            )
        )
        if required_status_satisfied and not refresh:
            counts["skipped_existing"] += 1
        else:
            try:
                if content_source == "pdf":
                    content, pdf = _fetch_static_pdf_content(client, announcement)
                else:
                    content = client.fetch_announcement_content(
                        str(announcement.announcement_id)
                    )
                    attachment = content.get("attach_url_web") or content.get("attach_url")
                    pdf = (
                        _fetch_pdf(client, attachment)
                        if pdf_mode == "archive"
                        else PdfEvidence(status="not_requested")
                    )
                if not content:
                    raise ValueError("empty content response")
                with Session(engine) as session:
                    stored_announcement = session.get(Announcement, announcement.id)
                    archive, created = archive_announcement_content(
                        session,
                        announcement=stored_announcement,
                        content=content,
                        dataset_role="train",
                        retrieved_at=datetime.now(UTC),
                        development=development,
                        split=split,
                        pdf=pdf,
                        source=(
                            "eastmoney_static_pdf"
                            if content_source == "pdf"
                            else "eastmoney"
                        ),
                    )
                    session.commit()
                    counts["created"] += int(created)
                    counts["identical_existing"] += int(not created)
                    counts["content_chars"] += archive.content_chars
                    counts["content_pages"] += archive.pages_fetched
                    counts["pdf_archived"] += int(
                        archive.pdf_fetch_status == "archived"
                    )
                failures = [
                    item
                    for item in failures
                    if item.get("announcement_id") != announcement.announcement_id
                ]
                consecutive_failures = 0
            except Exception as exc:
                counts["failed"] += 1
                consecutive_failures += 1
                failures = [
                    item
                    for item in failures
                    if item.get("announcement_id") != announcement.announcement_id
                ]
                failures.append(
                    {
                        "announcement_db_id": announcement.id,
                        "announcement_id": announcement.announcement_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        counts["processed"] += 1
        state = {
            "contract": "announcement-source-archive-backfill-v1",
            "development_contract_sha256": development.source_sha256,
            "last_announcement_db_id": max(
                int(state.get("last_announcement_db_id", 0)), announcement.id
            ),
            "processed": int(state.get("processed", 0)) + 1,
            "failures": failures,
            "selection": selection,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _save_json_atomic(state_path, state)
        if index % 10 == 0 or index == len(rows):
            print(
                f"archive progress {index}/{len(rows)}; "
                f"created={counts['created']}; failed={counts['failed']}; "
                f"skipped={counts['skipped_existing']}",
                flush=True,
            )
        if consecutive_failures >= consecutive_failure_limit:
            counts["circuit_breaker_tripped"] += 1
            print(
                f"circuit breaker: {consecutive_failures} consecutive failures; "
                "stopping for a later resumable retry",
                flush=True,
            )
            break

    report = {
        "contract": "announcement-source-archive-backfill-v1",
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "dataset_role": "train",
        "development_contract": development.contract,
        "development_contract_sha256": development.source_sha256,
        "split_contract": split.contract,
        "split_source_sha256": split.source_sha256,
        "market_target_table_queried": False,
        "market_direction_labels_read": False,
        "announcements_selected": len(rows),
        "pdf_mode": pdf_mode,
        "content_source": content_source,
        "only_accepted": only_accepted,
        "relevances": selection["relevances"],
        "rate_limit_per_minute": rate_limit_per_minute,
        "refresh": refresh,
        "counts": dict(counts),
        "unresolved_failures": failures,
    }
    _save_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--announcement-id", action="append", default=[])
    parser.add_argument(
        "--announcement-plan",
        type=Path,
        help="JSON plan containing a train-only announcement_ids list",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pdf-mode", choices=("none", "archive"), default="none")
    parser.add_argument("--content-source", choices=("api", "pdf"), default="api")
    parser.add_argument("--only-accepted", action="store_true")
    parser.add_argument("--relevance", action="append", default=[])
    parser.add_argument("--rate-limit", type=int, default=30)
    parser.add_argument("--consecutive-failure-limit", type=int, default=3)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--state",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "backfills" / "announcement_source_archive.json"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "validation"
            / "announcement_source_archive_backfill.json"
        ),
    )
    args = parser.parse_args()
    announcement_ids = list(args.announcement_id)
    if args.announcement_plan:
        plan = json.loads(args.announcement_plan.read_text(encoding="utf-8"))
        if plan.get("dataset_role") != "train":
            parser.error("announcement plan must be train only")
        if plan.get("market_targets_queried") is not False:
            parser.error("announcement plan must declare market_targets_queried=false")
        announcement_ids.extend(str(value) for value in plan["announcement_ids"])
    backfill(
        announcement_ids=list(dict.fromkeys(announcement_ids)),
        limit=max(args.limit, 0),
        pdf_mode=args.pdf_mode,
        only_accepted=args.only_accepted,
        relevances=args.relevance,
        rate_limit_per_minute=max(args.rate_limit, 1),
        refresh=args.refresh,
        state_path=args.state.resolve(),
        report_path=args.report.resolve(),
        consecutive_failure_limit=max(args.consecutive_failure_limit, 1),
        content_source=args.content_source,
    )


if __name__ == "__main__":
    main()

"""Immutable, content-addressed announcement source archival."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from src.config import PROJECT_ROOT
from src.database.models import Announcement, AnnouncementSourceArchive
from src.prediction.development_contract import CausalDevelopmentContract
from src.prediction.splits import PredictionSplitContract

SHANGHAI = ZoneInfo("Asia/Shanghai")
DEFAULT_PDF_ARCHIVE_ROOT = PROJECT_ROOT / "data" / "source_archive" / "pdf"


@dataclass(frozen=True)
class PdfEvidence:
    status: str
    content_type: str | None = None
    byte_count: int | None = None
    sha256: str | None = None
    storage_path: str | None = None


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_source_eitime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        local = datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=SHANGHAI
        )
    except ValueError:
        return None
    return local.astimezone(UTC)


def archive_pdf_bytes(
    data: bytes,
    *,
    content_type: str | None,
    root: Path = DEFAULT_PDF_ARCHIVE_ROOT,
) -> PdfEvidence:
    """Write PDF bytes once under their SHA-256; existing objects are verified."""
    if not data.startswith(b"%PDF"):
        raise ValueError("attachment is not a PDF")
    digest = hashlib.sha256(data).hexdigest()
    root = root.resolve()
    workspace = PROJECT_ROOT.resolve()
    if root != workspace and workspace not in root.parents:
        raise ValueError("PDF archive root must stay inside the project workspace")
    target = root / digest[:2] / f"{digest}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = target.read_bytes()
        if hashlib.sha256(existing).hexdigest() != digest:
            raise ValueError("existing content-addressed PDF failed hash verification")
    else:
        with open(target, "xb") as file:
            file.write(data)
    relative = target.relative_to(workspace).as_posix()
    return PdfEvidence(
        status="archived",
        content_type=content_type,
        byte_count=len(data),
        sha256=digest,
        storage_path=relative,
    )


def archive_announcement_content(
    session: Session,
    *,
    announcement: Announcement,
    content: dict,
    dataset_role: str,
    retrieved_at: datetime,
    development: CausalDevelopmentContract,
    split: PredictionSplitContract,
    pdf: PdfEvidence | None = None,
    source: str = "eastmoney",
) -> tuple[AnnouncementSourceArchive, bool]:
    """Append a verified source snapshot, or return the identical existing snapshot."""
    if dataset_role != "train" or split.role_for(announcement.published_date) != "train":
        raise PermissionError("source backfill currently permits the train role only")
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("retrieved_at must be timezone-aware")
    if not content.get("_content_complete"):
        raise ValueError("incomplete paginated content cannot be archived")
    if content.get("art_code") not in (None, announcement.announcement_id):
        raise ValueError("source art_code does not match announcement")
    source_title = "".join(str(content.get("notice_title") or "").split())
    stored_title = "".join(announcement.title.split())
    if source_title and source_title != stored_title:
        raise ValueError("source title does not match announcement")
    notice_date_raw = str(content.get("notice_date") or "")
    if notice_date_raw and notice_date_raw[:10] != announcement.published_date.isoformat():
        raise ValueError("source notice_date does not match announcement")

    text = str(content.get("notice_content") or "")
    page_hashes = list(content.get("_content_page_sha256") or [])
    payload_hashes = list(content.get("_content_page_payload_sha256") or [])
    page_chars = list(content.get("_content_page_chars") or [])
    pages_expected = int(content.get("_content_pages_expected") or 0)
    pages_fetched = int(content.get("_content_pages_fetched") or 0)
    manifest_lengths = (len(page_hashes), len(payload_hashes), len(page_chars))
    if not (pages_expected == pages_fetched and manifest_lengths == (pages_fetched,) * 3):
        raise ValueError("content page manifest is inconsistent")
    if sum(int(value) for value in page_chars) != len(text):
        raise ValueError("content page lengths do not reconstruct the combined text")
    source_payload_sha256 = str(
        content.get("_content_source_payload_sha256") or ""
    )
    if len(source_payload_sha256) != 64:
        raise ValueError("source payload hash is missing")
    page_manifest = [
        {
            "page_index": index,
            "chars": int(chars),
            "content_sha256": page_hash,
            "payload_sha256": payload_hash,
        }
        for index, (chars, page_hash, payload_hash) in enumerate(
            zip(page_chars, page_hashes, payload_hashes, strict=True),
            1,
        )
    ]
    metadata = {
        key: value
        for key, value in content.items()
        if key != "notice_content" and not key.startswith("_content_")
    }
    pdf = pdf or PdfEvidence(status="not_requested")
    identity = {
        "announcement_id": announcement.id,
        "source": source,
        "source_art_code": announcement.announcement_id,
        "content_sha256": _sha256_text(text),
        "source_payload_sha256": source_payload_sha256,
        "pdf_sha256": pdf.sha256,
        "development_contract_sha256": development.source_sha256,
    }
    archive_id = _sha256_text(_canonical_json(identity))
    existing = session.query(AnnouncementSourceArchive).filter_by(
        archive_id=archive_id
    ).one_or_none()
    if existing is not None:
        return existing, False
    latest = (
        session.query(AnnouncementSourceArchive)
        .filter_by(announcement_id=announcement.id, source=source)
        .order_by(AnnouncementSourceArchive.id.desc())
        .first()
    )
    source_eitime_raw = str(content.get("eitime") or "") or None
    row = AnnouncementSourceArchive(
        archive_id=archive_id,
        announcement_id=announcement.id,
        supersedes_id=latest.id if latest else None,
        source=source,
        source_art_code=str(announcement.announcement_id),
        dataset_role=dataset_role,
        retrieved_at=retrieved_at.astimezone(UTC),
        notice_date_raw=notice_date_raw or None,
        source_eitime_raw=source_eitime_raw,
        source_time_candidate=_parse_source_eitime(source_eitime_raw),
        source_time_semantics="eastmoney_eitime_candidate",
        source_time_authoritative=False,
        content_text=text,
        content_chars=len(text),
        content_sha256=identity["content_sha256"],
        pages_expected=pages_expected,
        pages_fetched=pages_fetched,
        content_complete=True,
        page_manifest_json=_canonical_json(page_manifest),
        source_metadata_json=_canonical_json(metadata),
        source_payload_sha256=source_payload_sha256,
        attachment_url=content.get("attach_url_web") or content.get("attach_url"),
        pdf_fetch_status=pdf.status,
        pdf_content_type=pdf.content_type,
        pdf_bytes=pdf.byte_count,
        pdf_sha256=pdf.sha256,
        pdf_storage_path=pdf.storage_path,
        development_contract=development.contract,
        development_contract_sha256=development.source_sha256,
    )
    session.add(row)
    session.flush()
    return row, True

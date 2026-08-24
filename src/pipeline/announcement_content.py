"""Utilities for auditable multi-page announcement content."""

from __future__ import annotations

import hashlib
import json
import re
from io import BytesIO


def expected_content_pages(first_page: dict) -> int:
    value = first_page.get("page_size") or 1
    try:
        pages = int(value)
    except (TypeError, ValueError):
        pages = 1
    return max(pages, 1)


def assemble_content_pages(
    pages: list[dict],
    *,
    expected_pages: int,
) -> dict:
    """Combine fetched pages and expose completeness/provenance metadata."""
    if not pages:
        return {}
    first = dict(pages[0])
    art_code = first.get("art_code")
    contents = []
    page_hashes = []
    page_payload_hashes = []
    page_chars = []
    for page in pages:
        if art_code and page.get("art_code") not in (None, art_code):
            raise ValueError("announcement page art_code mismatch")
        content = str(page.get("notice_content") or "")
        contents.append(content)
        page_chars.append(len(content))
        page_hashes.append(hashlib.sha256(content.encode("utf-8")).hexdigest())
        page_payload_hashes.append(
            hashlib.sha256(
                json.dumps(
                    page,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
    first["notice_content"] = "".join(contents)
    first["_content_pages_expected"] = expected_pages
    first["_content_pages_fetched"] = len(pages)
    first["_content_complete"] = len(pages) == expected_pages
    first["_content_page_sha256"] = page_hashes
    first["_content_page_payload_sha256"] = page_payload_hashes
    first["_content_page_chars"] = page_chars
    first["_content_source_payload_sha256"] = hashlib.sha256(
        json.dumps(
            page_payload_hashes,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return first


def assemble_pdf_extracted_content(
    data: bytes,
    *,
    art_code: str,
    title: str,
    notice_date: str,
    source_eitime: str | None,
    attachment_url: str,
) -> dict:
    """Extract auditable page text from a verified Eastmoney static PDF."""
    if not data.startswith(b"%PDF"):
        raise ValueError("attachment is not a PDF")
    from pypdf import PdfReader
    from pypdf import __version__ as pypdf_version

    reader = PdfReader(BytesIO(data))
    if not reader.pages:
        raise ValueError("PDF contains no pages")
    pages = []
    for page_index, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if not text.strip():
            raise ValueError(f"PDF page {page_index} has no extractable text")
        page_data = {
            "art_code": art_code,
            "page_size": len(reader.pages),
            "notice_content": text,
            "pdf_page_index": page_index,
            "pdf_text_extractor": f"pypdf-{pypdf_version}",
        }
        if page_index == 1:
            page_data.update(
                {
                    "notice_title": title,
                    "notice_date": notice_date,
                    "eitime": source_eitime,
                    "attach_url_web": attachment_url,
                    "attach_type": "0",
                    "content_source": "eastmoney_static_pdf",
                }
            )
        pages.append(page_data)
    combined = assemble_content_pages(pages, expected_pages=len(pages))
    comparable_title = title
    for separator in (":", "："):
        prefix, found, remainder = comparable_title.partition(separator)
        if found and len(prefix) <= 20:
            comparable_title = remainder
            break
    comparable_title = re.sub(
        r"[（(][^（）()]{0,20}(?:稿|版)[^（）()]{0,10}[）)]$",
        "",
        comparable_title,
    )
    if "关于" in comparable_title:
        comparable_title = comparable_title[comparable_title.index("关于") :]
    normalized_title = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", comparable_title)
    normalized_first_pages = re.sub(
        r"[^0-9A-Za-z\u4e00-\u9fff]+",
        "",
        "".join(str(page["notice_content"]) for page in pages[:3]),
    )
    title_bigrams = {
        normalized_title[index : index + 2]
        for index in range(max(len(normalized_title) - 1, 0))
    }
    matched_bigrams = {
        value for value in title_bigrams if value in normalized_first_pages
    }
    title_score = (
        1.0
        if normalized_title in normalized_first_pages
        else len(matched_bigrams) / max(len(title_bigrams), 1)
    )
    if len(normalized_title) < 6 or title_score < 0.8:
        raise ValueError("stored announcement title not found in first three PDF pages")
    combined["pdf_title_verified"] = True
    combined["pdf_title_match_score"] = round(title_score, 6)
    combined["pdf_text_extractor"] = f"pypdf-{pypdf_version}"
    return combined

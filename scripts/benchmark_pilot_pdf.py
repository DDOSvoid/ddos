"""Compare a small train-only long-document sample with archived API text."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.announcement_content import assemble_pdf_extracted_content  # noqa: E402


def comparison(reference, candidate):
    def normalized(value):
        return re.sub(r"\s+", "", value)

    a, b = normalized(reference), normalized(candidate)
    grams = {a[i : i + 8] for i in range(max(0, len(a) - 7))}
    other = {b[i : i + 8] for i in range(max(0, len(b) - 7))}
    numbers_a = Counter(re.findall(r"\d+(?:[.,]\d+)*", reference))
    numbers_b = Counter(re.findall(r"\d+(?:[.,]\d+)*", candidate))
    recall = sum((numbers_a & numbers_b).values()) / max(1, sum(numbers_a.values()))
    coverage = len(grams & other) / max(1, len(grams))
    return dict(
        text_8gram_recall=coverage,
        number_occurrence_recall=recall,
        normalized_length_ratio=len(b) / max(1, len(a)),
        conservative_fidelity_pass=coverage >= 0.98 and recall >= 0.98,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(
        Path("data/backfills/event_ranking_pilot_electrical_v1/body_plan.json").read_text(
            encoding="utf-8"
        )
    )
    ids = set(plan["announcement_ids"])
    with sqlite3.connect("file:data/ddos.db?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT a.announcement_id,a.title,a.published_date,s.content_text,s.pages_expected "
            "FROM announcement_source_archives s JOIN announcements a ON a.id=s.announcement_id "
            "WHERE a.published_date BETWEEN '2023-01-01' AND '2024-12-31' "
            "AND s.dataset_role='train' AND s.content_complete=1 AND s.pages_expected>=20"
        ).fetchall()
    chosen = sorted(
        {row[0]: row for row in rows if row[0] in ids}.values(),
        key=lambda row: hashlib.sha256(row[0].encode()).hexdigest(),
    )[: args.limit]
    result = dict(
        purpose="exploratory_pdf_fidelity_and_latency",
        official_test_read=False,
        strict_oos_2026_read=False,
        automatic_source_switch_authorized_by_report=False,
        results=[],
    )
    session = requests.Session()
    session.trust_env = False
    for aid, title, published, reference, pages in chosen:
        started = time.monotonic()
        row = dict(
            announcement_id=aid,
            published_date=published,
            api_pages=pages,
            api_rate_limit_floor_seconds=max(pages - 1, 0) * 2,
        )
        try:
            url = f"https://pdf.dfcfw.com/pdf/H2_{aid}_1.pdf"
            response = session.get(url, timeout=(10, 60))
            response.raise_for_status()
            fetched = time.monotonic()
            content = assemble_pdf_extracted_content(
                response.content,
                art_code=aid,
                title=title,
                notice_date=published,
                source_eitime=None,
                attachment_url=url,
            )
            row.update(
                download_seconds=fetched - started,
                total_seconds=time.monotonic() - started,
                pdf_pages=content["_content_pages_fetched"],
                pdf_bytes=len(response.content),
                pdf_sha256=hashlib.sha256(response.content).hexdigest(),
                reference_sha256=hashlib.sha256(reference.encode()).hexdigest(),
                comparison=comparison(reference, content["notice_content"]),
            )
        except Exception as error:
            row.update(
                error_type=type(error).__name__,
                error=str(error)[:600],
                total_seconds=time.monotonic() - started,
            )
        result["results"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=True), flush=True)
    session.close()


if __name__ == "__main__":
    main()

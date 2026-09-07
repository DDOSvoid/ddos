"""Verified, expiring page checkpoints keyed by a freshly fetched first page.

The first page is always re-fetched. Its complete payload binds a generation;
different first-page metadata/content cannot reuse pages from an earlier one.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def payload_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


class ContentPageCache:
    def __init__(self, root: Path, art_code: str, first_page: dict, *, max_age=86400):
        self.art_code = art_code
        self.generation = payload_hash(first_page)
        self.directory = root / hashlib.sha256(art_code.encode()).hexdigest() / self.generation
        self.max_age = max_age

    def read(self, index: int) -> dict | None:
        path = self.directory / f"{index:04d}.json"
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            age = time.time() - float(row["saved_at"])
            data = row["payload"]
            if (
                not 0 <= age <= self.max_age
                or row["generation"] != self.generation
                or row["art_code"] != self.art_code
                or row["page_index"] != index
                or payload_hash(data) != row["payload_sha256"]
            ):
                return None
            if not isinstance(data, dict) or not str(data.get("notice_content") or "").strip():
                return None
            if data.get("art_code") not in (None, self.art_code):
                return None
            return data
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def write(self, index: int, data: dict) -> None:
        if not str(data.get("notice_content") or "").strip():
            return
        if data.get("art_code") not in (None, self.art_code):
            raise ValueError("announcement page art_code mismatch")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{index:04d}.json"
        temporary = path.with_suffix(".tmp")
        row = dict(
            art_code=self.art_code,
            generation=self.generation,
            page_index=index,
            saved_at=time.time(),
            payload=data,
            payload_sha256=payload_hash(data),
        )
        temporary.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

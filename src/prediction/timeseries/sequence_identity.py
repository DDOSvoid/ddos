"""Canonical cross-branch identities for company-day prediction samples."""

from __future__ import annotations

import hashlib
from datetime import date

CANONICAL_HORIZONS = (1, 3, 5)


def _stable_sha256(*parts: object) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_company_day_id(stock_code: str, published_date: date) -> str:
    code = str(stock_code).strip().upper()
    if not code:
        raise ValueError("stock_code cannot be empty")
    if not isinstance(published_date, date):
        raise TypeError("published_date must be a date")
    return _stable_sha256("company_day", code, published_date)


def canonical_sample_id(company_day_id: str, horizon_sessions: int) -> str:
    identity = str(company_day_id).strip().lower()
    if len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity):
        raise ValueError("company_day_id must be a lowercase SHA-256 hex digest")
    horizon = int(horizon_sessions)
    if horizon not in CANONICAL_HORIZONS:
        raise ValueError(f"unsupported horizon: {horizon}")
    return _stable_sha256("sample", identity, horizon)


def canonical_identities(
    stock_code: str,
    published_date: date,
    horizon_sessions: int,
) -> tuple[str, str]:
    company_day_id = canonical_company_day_id(stock_code, published_date)
    return company_day_id, canonical_sample_id(company_day_id, horizon_sessions)

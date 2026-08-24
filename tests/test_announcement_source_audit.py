"""Announcement source-audit helpers."""

from datetime import datetime

from scripts.audit_announcement_source_coverage import parse_eitime


def test_parse_eitime_accepts_millisecond_suffix():
    assert parse_eitime("2023-01-30 18:57:32:000") == datetime(
        2023, 1, 30, 18, 57, 32
    )


def test_parse_eitime_rejects_missing_or_invalid_values():
    assert parse_eitime("") is None
    assert parse_eitime(None) is None
    assert parse_eitime("not-a-time") is None

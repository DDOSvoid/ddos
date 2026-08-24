"""Candidate announcement-time audit helpers."""

from datetime import datetime

from scripts.audit_announcement_timestamp_semantics import (
    _art_code_date,
    _attachment_epoch,
)


def test_art_code_date_is_extracted():
    assert _art_code_date("AN202306191591068543").isoformat() == "2023-06-19"
    assert _art_code_date("invalid") is None


def test_attachment_epoch_is_converted_to_shanghai_time():
    value = _attachment_epoch(
        "https://pdf.dfcfw.com/pdf/a.pdf?1687193392000.pdf"
    )
    assert value == datetime(2023, 6, 20, 0, 49, 52)

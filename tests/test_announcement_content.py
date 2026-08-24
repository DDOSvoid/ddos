"""Multi-page announcement assembly tests."""

import pytest

from src.pipeline.announcement_content import (
    assemble_content_pages,
    expected_content_pages,
)


def test_expected_pages_defaults_safely():
    assert expected_content_pages({}) == 1
    assert expected_content_pages({"page_size": "3"}) == 3
    assert expected_content_pages({"page_size": "invalid"}) == 1


def test_partial_pages_are_marked_incomplete():
    result = assemble_content_pages(
        [{"art_code": "AN1", "notice_content": "first"}],
        expected_pages=2,
    )
    assert result["notice_content"] == "first"
    assert result["_content_complete"] is False


def test_page_identity_mismatch_is_rejected():
    with pytest.raises(ValueError, match="art_code mismatch"):
        assemble_content_pages(
            [
                {"art_code": "AN1", "notice_content": "first"},
                {"art_code": "AN2", "notice_content": "second"},
            ],
            expected_pages=2,
        )

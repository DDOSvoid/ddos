"""Tests for strict PDF-derived announcement text assembly."""

import pytest

from src.pipeline import announcement_content


class _Page:
    def __init__(self, text: str):
        self.text = text

    def extract_text(self) -> str:
        return self.text


class _Reader:
    def __init__(self, stream):
        self.pages = [_Page("测试公司 重大合同公告\n"), _Page("合同金额为1亿元。")]


def test_pdf_text_requires_title_and_builds_page_manifest(monkeypatch):
    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", _Reader)
    result = announcement_content.assemble_pdf_extracted_content(
        b"%PDF-test",
        art_code="AN202401010000000001",
        title="测试公司:测试公司重大合同公告(注册稿)",
        notice_date="2024-01-01",
        source_eitime="2023-12-31 18:00:00",
        attachment_url="https://pdf.example/test.pdf",
    )
    assert result["_content_complete"] is True
    assert result["_content_pages_fetched"] == 2
    assert result["pdf_title_verified"] is True
    assert result["pdf_title_match_score"] == 1.0
    assert "1亿元" in result["notice_content"]


def test_pdf_text_rejects_empty_page_and_title_mismatch(monkeypatch):
    import pypdf

    class EmptyReader:
        def __init__(self, stream):
            self.pages = [_Page("")]

    monkeypatch.setattr(pypdf, "PdfReader", EmptyReader)
    with pytest.raises(ValueError, match="no extractable text"):
        announcement_content.assemble_pdf_extracted_content(
            b"%PDF-test",
            art_code="AN1",
            title="标题",
            notice_date="2024-01-01",
            source_eitime=None,
            attachment_url="https://pdf.example/test.pdf",
        )

    class WrongTitleReader:
        def __init__(self, stream):
            self.pages = [_Page("完全不同的正文")]

    monkeypatch.setattr(pypdf, "PdfReader", WrongTitleReader)
    with pytest.raises(ValueError, match="title not found"):
        announcement_content.assemble_pdf_extracted_content(
            b"%PDF-test",
            art_code="AN1",
            title="目标标题",
            notice_date="2024-01-01",
            source_eitime=None,
            attachment_url="https://pdf.example/test.pdf",
        )

"""测试文本清洗工具。"""

import pytest
from src.utils.text_utils import (
    clamp,
    clean_chinese_text,
    clean_html,
    normalize_amount,
    normalize_whitespace,
    parse_chinese_number,
    remove_control_chars,
    safe_divide,
    truncate_for_model,
)


class TestCleanHtml:
    def test_remove_simple_tags(self):
        assert clean_html("<p>Hello</p>") == "Hello"

    def test_remove_script_content(self):
        assert clean_html('<script>alert("x")</script>Text') == "Text"

    def test_decode_entities(self):
        result = clean_html("A&amp;B&nbsp;C")
        assert "&" in result
        assert " " in result
        assert "&amp;" not in result

    def test_empty_input(self):
        assert clean_html("") == ""
        assert clean_html(None) == ""


class TestNormalizeWhitespace:
    def test_merge_spaces(self):
        assert normalize_whitespace("a    b") == "a b"

    def test_merge_newlines(self):
        assert normalize_whitespace("line1\n\n\nline2") == "line1 line2"

    def test_strip(self):
        assert normalize_whitespace("  hello  ") == "hello"


class TestRemoveControlChars:
    def test_remove_null(self):
        assert remove_control_chars("a\x00b") == "ab"

    def test_preserve_newline_and_tab(self):
        assert "\n" in remove_control_chars("a\nb")
        assert "\t" in remove_control_chars("a\tb")


class TestCleanChineseText:
    def test_full_pipeline(self):
        raw = "<p>公司实现营收<strong>100亿元</strong>，同比增长15%。</p>"
        cleaned = clean_chinese_text(raw)
        assert "营收" in cleaned
        assert "<p>" not in cleaned
        assert "<strong>" not in cleaned

    def test_multiline_html(self):
        raw = """<html>
        <body><p>公告正文内容</p></body>
        </html>"""
        cleaned = clean_chinese_text(raw)
        assert "公告正文内容" in cleaned


class TestTruncateForModel:
    def test_short_text(self):
        assert truncate_for_model("hello", 100) == "hello"

    def test_long_text(self):
        long_text = "x" * 3000
        assert len(truncate_for_model(long_text, 2000)) == 2000


class TestParseChineseNumber:
    def test_plain_number(self):
        assert parse_chinese_number("5000") == 5000

    def test_wan(self):
        assert parse_chinese_number("2000万") == 20_000_000

    def test_yi(self):
        assert parse_chinese_number("1.5亿") == 150_000_000

    def test_percent(self):
        assert parse_chinese_number("15%") == 0.15

    def test_with_comma(self):
        assert parse_chinese_number("3,000万") == 30_000_000

    def test_empty(self):
        assert parse_chinese_number("") is None
        assert parse_chinese_number(None) is None


class TestNormalizeAmount:
    def test_yi_to_wan(self):
        assert normalize_amount(1.5, "cny_100m") == 15_000  # 1.5亿 = 15000万

    def test_yuan_to_wan(self):
        assert normalize_amount(50000, "CNY") == 5  # 50000元 = 5万

    def test_default_wan(self):
        assert normalize_amount(100, "wan") == 100


class TestSafeDivide:
    def test_normal(self):
        assert safe_divide(10, 2) == 5.0

    def test_by_zero(self):
        assert safe_divide(10, 0) == 0.0

    def test_near_zero(self):
        assert safe_divide(10, 1e-12) == 0.0


class TestClamp:
    def test_within_range(self):
        assert clamp(0.5) == 0.5

    def test_below(self):
        assert clamp(-0.2) == 0.0

    def test_above(self):
        assert clamp(1.5) == 1.0

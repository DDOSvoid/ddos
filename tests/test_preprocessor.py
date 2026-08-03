"""测试文本预处理器。"""

from src.pipeline.preprocessor import Preprocessor


class TestPreprocessor:
    def test_clean_empty(self):
        pp = Preprocessor(max_length=512)
        result = pp.preprocess_text("")
        assert result == ""

    def test_clean_html(self):
        pp = Preprocessor(max_length=512)
        raw = "<html><body><p>公司2024年Q1实现营收<strong>100亿元</strong></p></body></html>"
        cleaned = pp.preprocess_text(raw)
        assert "营收" in cleaned
        assert "100亿元" in cleaned
        assert "<html>" not in cleaned
        assert "<strong>" not in cleaned

    def test_clean_special_chars(self):
        pp = Preprocessor(max_length=512)
        raw = "公告内容\x00\x01包含特殊字符"
        cleaned = pp.preprocess_text(raw)
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned

    def test_truncate_long_text(self):
        pp = Preprocessor(max_length=512)
        # 创建超长文本
        long_text = "公告正文。" * 5000
        cleaned = pp.preprocess_text(long_text)
        # 应被截断到 ~1536 字符 (512 * 3)
        assert len(cleaned) <= 512 * 3

    def test_title_body_merge(self):
        pp = Preprocessor(max_length=512)
        result = pp.preprocess_title_and_body(
            title="2024年一季报",
            body="公司实现营收100亿元，同比增长15%",
        )
        assert "2024年一季报" in result
        assert "营收" in result
        # 标题应在前面
        assert result.index("2024年一季报") < result.index("营收")

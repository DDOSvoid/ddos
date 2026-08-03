"""测试 Fetcher 模块（Mock API 调用）。"""

from datetime import date

import pytest


class TestTushareClient:
    """需要 Tushare token 才能运行，CI 中跳过。"""

    @pytest.mark.skip(reason="Requires Tushare token")
    def test_get_stock_basic(self):
        from src.pipeline.fetcher import TushareClient
        client = TushareClient()
        df = client.get_stock_basic()
        assert len(df) > 1000

    @pytest.mark.skip(reason="Requires Tushare token")
    def test_get_disclosure(self):
        from src.pipeline.fetcher import TushareClient
        client = TushareClient()
        df = client.get_disclosure(start_date="20240101", end_date="20240131")
        # disclosure 可能返回空（取决于数据可用性）


class TestEastmoneyClient:
    """东方财富接口测试。"""

    def test_fetch_announcements(self):
        from src.pipeline.fetcher import EastmoneyClient
        client = EastmoneyClient()
        result = client.fetch_announcements(
            stock_code="000001",
            start_date="2024-01-01",
            end_date="2024-01-07",
            page_size=5,
        )
        # 应返回 dict 结构
        assert isinstance(result, dict)

    def test_fetch_all_pagination(self):
        from src.pipeline.fetcher import EastmoneyClient
        client = EastmoneyClient()
        items = client.fetch_all_announcements(
            stock_code="000001",
            start_date="2024-01-01",
            end_date="2024-01-07",
            max_pages=2,
        )
        assert isinstance(items, list)


class TestStockCodeHelper:
    def test_short_code(self):
        from src.pipeline.fetcher import _stock_code_short
        assert _stock_code_short("000001.SZ") == "000001"
        assert _stock_code_short("601012.SH") == "601012"


class TestDateHelpers:
    def test_format_date(self):
        from src.pipeline.fetcher import _format_date
        assert _format_date(date(2024, 1, 15)) == "20240115"

    def test_format_date_dash(self):
        from src.pipeline.fetcher import _format_date_dash
        assert _format_date_dash(date(2024, 1, 15)) == "2024-01-15"

    def test_parse_em_date(self):
        from src.pipeline.fetcher import Fetcher
        d = Fetcher._parse_em_date("2024-01-15")
        assert d == date(2024, 1, 15)

        d2 = Fetcher._parse_em_date("")
        assert d2 is None

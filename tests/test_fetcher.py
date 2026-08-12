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


class TestEastmoneyContentApi:
    """内容接口测试（mock HTTP，不真实调用）。"""

    def test_fetch_announcement_content_parses(self, monkeypatch):
        from src.pipeline.fetcher import EastmoneyClient
        client = EastmoneyClient()

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": {
                    "notice_content": "公告正文内容",
                    "attach_url_web": "https://pdf.example.com/a.pdf",
                }}

        monkeypatch.setattr(client.session, "get", lambda *a, **k: FakeResp())
        result = client.fetch_announcement_content("AN123")
        assert result["notice_content"] == "公告正文内容"
        assert result["attach_url_web"] == "https://pdf.example.com/a.pdf"

    def test_fetch_announcement_content_error_returns_empty(self, monkeypatch):
        from src.pipeline.fetcher import EastmoneyClient
        client = EastmoneyClient()

        def _boom(*a, **k):
            raise ConnectionError("network down")

        monkeypatch.setattr(client.session, "get", _boom)
        assert client.fetch_announcement_content("AN123") == {}


class TestFetcherEnrichment:
    """Fetcher.run 正文富化测试（mock 网络与 repository，用临时文件库）。"""

    def test_run_enriches_full_text(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        import src.pipeline.fetcher as fetcher_mod
        from src.database.models import Announcement, Base

        # 临时文件库（内存库跨连接会丢数据）
        engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
        Base.metadata.create_all(engine)
        monkeypatch.setattr(fetcher_mod, "get_engine", lambda: engine)

        # mock 跟踪公司
        class FakeCompany:
            id = 1
            stock_code = "000001.SZ"

        monkeypatch.setattr(
            fetcher_mod.CompanyRepository,
            "get_tracked",
            lambda session: [FakeCompany()],
        )

        # 绕过 __init__（避免 TushareClient），替换 em 避免真实网络
        fetcher = fetcher_mod.Fetcher.__new__(fetcher_mod.Fetcher)
        fetcher.em = SimpleNamespace(
            fetch_all_announcements=lambda **kw: [{
                "art_code": "AN202608071827755919",
                "title": "测试公告",
                "notice_date": "2026-08-08 00:00:00",
                "notice_content": "",
            }],
            fetch_announcement_content=lambda art_code: {
                "notice_content": "<p>公司2026年半年度实现营收<b>100亿元</b>，净利润20亿元。</p>",
                "attach_url_web": "https://pdf.example.com/1.pdf",
                "attach_url": None,
            },
        )

        count = fetcher.run(target_date=date(2026, 8, 8), lookback_days=1)
        assert count == 1

        with Session(engine) as s:
            ann = s.query(Announcement).first()
            assert ann is not None
            # 正文已被 clean_chinese_text 清洗（HTML 标签剥离、内容保留）
            assert "100亿元" in ann.full_text
            assert "净利润" in ann.full_text
            assert "<p>" not in ann.full_text
            assert ann.pdf_url == "https://pdf.example.com/1.pdf"
            assert ann.processing_status == "fetched"

    def test_run_without_full_text_keeps_list_content(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        import src.pipeline.fetcher as fetcher_mod
        from src.database.models import Announcement, Base
        from src.config import config

        engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
        Base.metadata.create_all(engine)
        monkeypatch.setattr(fetcher_mod, "get_engine", lambda: engine)
        monkeypatch.setattr(config.pipeline, "fetch_full_text", False)

        class FakeCompany:
            id = 1
            stock_code = "000001.SZ"

        monkeypatch.setattr(
            fetcher_mod.CompanyRepository,
            "get_tracked",
            lambda session: [FakeCompany()],
        )

        fetcher = fetcher_mod.Fetcher.__new__(fetcher_mod.Fetcher)
        fetcher.em = SimpleNamespace(
            fetch_all_announcements=lambda **kw: [{
                "art_code": "AN2",
                "title": "测试",
                "notice_date": "2026-08-08 00:00:00",
                "notice_content": "列表摘要文本",
            }],
            fetch_announcement_content=lambda art_code: {
                "notice_content": "不应被调用",
                "attach_url_web": "https://x.pdf",
            },
        )

        count = fetcher.run(target_date=date(2026, 8, 8), lookback_days=1)
        assert count == 1

        with Session(engine) as s:
            ann = s.query(Announcement).first()
            assert ann.full_text == "列表摘要文本"
            assert ann.pdf_url is None

"""测试 CDP 模式东方财富客户端（mock Playwright，不启真浏览器）。"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.config import config

import src.pipeline.fetcher as fetcher_mod


def _make_client(page_mock=None):
    """构造 page.evaluate 被 mock 的 CdpEastmoneyClient，避免拉起真实 Chrome。"""
    from src.pipeline.cdp_fetcher import CdpEastmoneyClient

    client = CdpEastmoneyClient()
    client._min_interval = 0.0  # 关掉速率限制，避免测试 sleep
    client._page = page_mock if page_mock is not None else MagicMock()
    return client


def _sample_list_json(page_index=1, total_hits=3):
    return {
        "success": 1,
        "error": "",
        "data": {
            "page_index": page_index,
            "page_size": 50,
            "total_hits": total_hits,
            "list": [
                {"art_code": f"AN{page_index}0001", "title": f"公告{page_index}-1",
                 "notice_date": "2026-08-08 00:00:00", "notice_content": ""},
                {"art_code": f"AN{page_index}0002", "title": f"公告{page_index}-2",
                 "notice_date": "2026-08-08 00:00:00", "notice_content": ""},
            ],
        },
    }


class TestCdpFetchAnnouncements:
    """列表接口：URL 构建 + JSON 解析 + 分页。"""

    def test_fetch_announcements_parses(self):
        page = MagicMock()
        page.evaluate.return_value = _sample_list_json()
        client = _make_client(page)
        result = client.fetch_announcements(
            stock_code="000009.SZ", start_date="2026-08-01", end_date="2026-08-12"
        )
        assert result["data"]["total_hits"] == 3
        assert len(result["data"]["list"]) == 2

    def test_fetch_announcements_strips_exchange_suffix(self):
        # stock_code 带 .SZ 后缀时应剥掉，构造 stock_list=000009
        page = MagicMock()
        page.evaluate.return_value = _sample_list_json()
        client = _make_client(page)
        client.fetch_announcements(stock_code="000009.SZ")
        url = page.evaluate.call_args[0][1]
        assert "stock_list=000009" in url

    def test_fetch_all_pagination_total_hits(self):
        # 第1页 total_hits=100 → 需翻页；共翻 2 页后结束
        pages = iter([_sample_list_json(1, 100), _sample_list_json(2, 100)])
        page = MagicMock()
        page.evaluate.side_effect = lambda js, url: next(pages)
        client = _make_client(page)
        items = client.fetch_all_announcements(stock_code="000009", max_pages=5)
        assert len(items) == 4  # 2 页 × 2 条

    def test_fetch_all_single_page(self):
        # total_hits=2 → 单页即可，不应再翻页
        page = MagicMock()
        page.evaluate.return_value = _sample_list_json(1, 2)
        client = _make_client(page)
        items = client.fetch_all_announcements(stock_code="000009", max_pages=5)
        assert len(items) == 2

    def test_fetch_error_returns_empty_structures(self):
        page = MagicMock()
        page.evaluate.side_effect = RuntimeError("Failed to fetch")
        client = _make_client(page)
        assert client.fetch_all_announcements(stock_code="000009") == []
        assert client.fetch_announcements(stock_code="000009") == {"data": {"list": []}}


class TestCdpFetchContent:
    """正文接口：解析 notice_content / attach_url。"""

    def test_fetch_content(self):
        page = MagicMock()
        page.evaluate.return_value = {
            "data": {"notice_content": "公告正文", "attach_url_web": "https://pdf.x"}
        }
        client = _make_client(page)
        content = client.fetch_announcement_content("AN123")
        assert content["notice_content"] == "公告正文"
        assert content["attach_url_web"] == "https://pdf.x"

    def test_fetch_content_error_returns_empty(self):
        page = MagicMock()
        page.evaluate.side_effect = RuntimeError("network down")
        client = _make_client(page)
        assert client.fetch_announcement_content("AN123") == {}


class TestFetcherBackendSelection:
    """Fetcher 按 fetch_backend 配置选择 http / cdp 客户端。"""

    def test_init_selects_cdp_when_configured(self, monkeypatch):
        monkeypatch.setattr(config.pipeline, "fetch_backend", "cdp")
        monkeypatch.setattr(fetcher_mod, "TushareClient", lambda **kw: SimpleNamespace())
        from src.pipeline.cdp_fetcher import CdpEastmoneyClient

        fetcher = fetcher_mod.Fetcher()
        assert isinstance(fetcher.em, CdpEastmoneyClient)
        fetcher.em.close()  # 未启动浏览器，close 应幂等

    def test_init_selects_http_by_default(self, monkeypatch):
        monkeypatch.setattr(config.pipeline, "fetch_backend", "http")
        monkeypatch.setattr(fetcher_mod, "TushareClient", lambda **kw: SimpleNamespace())
        from src.pipeline.fetcher import EastmoneyClient

        fetcher = fetcher_mod.Fetcher()
        assert isinstance(fetcher.em, EastmoneyClient)

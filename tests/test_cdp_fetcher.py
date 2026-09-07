"""测试 CDP 模式东方财富客户端（mock Playwright，不启真浏览器）。"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import requests

import src.pipeline.fetcher as fetcher_mod
from src.config import config


def _make_client(page_mock=None):
    """构造 page.evaluate 被 mock 的 CdpEastmoneyClient，避免拉起真实 Chrome。"""
    from src.pipeline.cdp_fetcher import CdpEastmoneyClient

    client = CdpEastmoneyClient()
    client._min_interval = 0.0  # 关掉速率限制，避免测试 sleep
    client._page = page_mock if page_mock is not None else MagicMock()
    return client


def _response(payload):
    response = MagicMock()
    response.ok = True
    response.json.return_value = payload
    return response


def test_cdp_launch_bypasses_system_proxy_by_default():
    from src.pipeline.cdp_fetcher import CdpEastmoneyClient

    client = CdpEastmoneyClient(content_max_retries=1)
    assert client._launch_options()["args"] == ["--no-proxy-server"]


def test_cdp_launch_can_inherit_system_proxy_explicitly():
    from src.pipeline.cdp_fetcher import CdpEastmoneyClient

    client = CdpEastmoneyClient(bypass_system_proxy=False)
    assert "args" not in client._launch_options()


def test_cdp_launch_pins_content_host_to_public_dns(monkeypatch):
    import src.pipeline.cdp_fetcher as cdp_module

    monkeypatch.setattr(
        cdp_module, "_resolve_public_ipv4", lambda host: "43.159.109.234"
    )
    client = cdp_module.CdpEastmoneyClient()
    client._ensure_direct_host_mapping()

    assert client._launch_options()["args"] == [
        "--no-proxy-server",
        "--host-resolver-rules=MAP np-cnotice-stock.eastmoney.com 43.159.109.234",
    ]


def test_public_dns_falls_back_to_next_doh_endpoint(monkeypatch):
    import src.pipeline.cdp_fetcher as cdp_module

    session = MagicMock()
    session.get.side_effect = [
        requests.exceptions.SSLError("first endpoint unavailable"),
        MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(
                return_value={"Answer": [{"type": 1, "data": "43.159.109.234"}]}
            ),
        ),
    ]
    monkeypatch.setattr(cdp_module.requests, "Session", lambda: session)

    assert cdp_module._resolve_public_ipv4("np-cnotice-stock.eastmoney.com") == (
        "43.159.109.234"
    )
    assert session.trust_env is False
    assert [call.args[0] for call in session.get.call_args_list] == list(
        cdp_module._DNS_OVER_HTTPS_URLS[:2]
    )


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
        page.goto.return_value = _response(_sample_list_json())
        client = _make_client(page)
        result = client.fetch_announcements(
            stock_code="000009.SZ", start_date="2026-08-01", end_date="2026-08-12"
        )
        assert result["data"]["total_hits"] == 3
        assert len(result["data"]["list"]) == 2

    def test_fetch_announcements_strips_exchange_suffix(self):
        # stock_code 带 .SZ 后缀时应剥掉，构造 stock_list=000009
        page = MagicMock()
        page.goto.return_value = _response(_sample_list_json())
        client = _make_client(page)
        client.fetch_announcements(stock_code="000009.SZ")
        url = page.goto.call_args[0][0]
        assert "stock_list=000009" in url

    def test_fetch_all_pagination_total_hits(self):
        # 第1页 total_hits=100 → 需翻页；共翻 2 页后结束
        pages = iter(
            [_response(_sample_list_json(1, 100)), _response(_sample_list_json(2, 100))]
        )
        page = MagicMock()
        page.goto.side_effect = lambda *args, **kwargs: next(pages)
        client = _make_client(page)
        items = client.fetch_all_announcements(stock_code="000009", max_pages=5)
        assert len(items) == 4  # 2 页 × 2 条

    def test_fetch_all_single_page(self):
        # total_hits=2 → 单页即可，不应再翻页
        page = MagicMock()
        page.goto.return_value = _response(_sample_list_json(1, 2))
        client = _make_client(page)
        items = client.fetch_all_announcements(stock_code="000009", max_pages=5)
        assert len(items) == 2

    def test_fetch_error_returns_empty_structures(self):
        page = MagicMock()
        page.goto.side_effect = RuntimeError("network down")
        client = _make_client(page)
        assert client.fetch_all_announcements(stock_code="000009") == []
        assert client.fetch_announcements(stock_code="000009") == {"data": {"list": []}}


class TestCdpFetchContent:
    """正文接口：解析 notice_content / attach_url。"""

    def test_fetch_content(self):
        page = MagicMock()
        page.goto.return_value = _response(
            {
                "data": {
                    "notice_content": "公告正文",
                    "attach_url_web": "https://pdf.x",
                }
            }
        )
        client = _make_client(page)
        content = client.fetch_announcement_content("AN123")
        assert content["notice_content"] == "公告正文"
        assert content["attach_url_web"] == "https://pdf.x"
        assert content["_content_complete"] is True

    def test_fetch_content_assembles_all_pages(self):
        client = _make_client(MagicMock())
        client._fetch_json = MagicMock(
            side_effect=[
                {
                    "data": {
                        "art_code": "AN123",
                        "page_size": 2,
                        "notice_content": "first",
                    }
                },
                {
                    "data": {
                        "art_code": "AN123",
                        "page_size": 2,
                        "notice_content": "second",
                    }
                },
            ]
        )
        content = client.fetch_announcement_content("AN123")
        assert content["notice_content"] == "firstsecond"
        assert content["_content_pages_fetched"] == 2
        assert content["_content_complete"] is True

    def test_fetch_content_error_returns_empty(self):
        page = MagicMock()
        page.goto.side_effect = RuntimeError("network down")
        client = _make_client(page)
        assert client.fetch_announcement_content("AN123") == {}

    def test_fetch_content_uses_long_cooldown_for_http_567(self, monkeypatch):
        import src.pipeline.cdp_fetcher as cdp_module

        sleeps = []
        monkeypatch.setattr(cdp_module.time, "sleep", sleeps.append)
        client = cdp_module.CdpEastmoneyClient(
            content_max_retries=2,
            content_retry_backoff_seconds=1,
        )
        client._fetch_json = MagicMock(
            side_effect=[
                RuntimeError("HTTP 567"),
                {"data": {"art_code": "AN123", "notice_content": "body"}},
            ]
        )

        content = client.fetch_announcement_content("AN123")

        assert content["notice_content"] == "body"
        assert sleeps == [120.0]


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

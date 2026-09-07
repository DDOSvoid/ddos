"""CDP 模式东方财富公告客户端 — 用 Playwright 驱动系统 Chrome 抓取。

与 `EastmoneyClient`（HTTP requests 版）**接口完全同构**，产出同样的中间数据，
让 `Fetcher.run()` 的富化/入库逻辑无需改动。

实现原理:
  - 用 Playwright 启动系统 Chrome（channel="chrome"，不下载浏览器）
  - 用 DoH 取得正文域名的真实公网 IP，绕过本机透明代理的 Fake-IP
  - 列表与正文通过 Chrome 页面导航请求 API，避开跨域 fetch 的 CORS 限制
  - 读取 Chrome 收到的 JSON 响应，产出与 HTTP 版一致的结构化数据

用法:
    client = CdpEastmoneyClient()
    items = client.fetch_all_announcements("000009", "2026-08-05", "2026-08-12")
    content = client.fetch_announcement_content("AN202608071827755919")
    client.close()
"""

import atexit
import ipaddress
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
from loguru import logger

from src.config import config

# 正文 API（与 EastmoneyClient.fetch_announcement_content 一致）
_CONTENT_URL = "https://np-cnotice-stock.eastmoney.com/api/content/ann"
_CONTENT_HOST = "np-cnotice-stock.eastmoney.com"
_DNS_OVER_HTTPS_URLS = (
    "https://cloudflare-dns.com/dns-query",
    "https://dns.alidns.com/resolve",
    "https://dns.google/resolve",
)
# 启动页：用于建立正常的东方财富浏览器上下文。
_HOME_URL = "https://data.eastmoney.com/notices/"


def _resolve_public_ipv4(host: str, *, timeout_seconds: int = 10) -> str:
    """Resolve a public IPv4 through redundant DoH, bypassing fake-IP DNS."""
    session = requests.Session()
    session.trust_env = False
    errors: list[str] = []
    for endpoint in _DNS_OVER_HTTPS_URLS:
        try:
            response = session.get(
                endpoint,
                params={"name": host, "type": "A"},
                headers={"Accept": "application/dns-json"},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            for answer in response.json().get("Answer") or []:
                if int(answer.get("type") or 0) != 1:
                    continue
                candidate = ipaddress.ip_address(str(answer.get("data") or ""))
                if candidate.version == 4 and candidate.is_global:
                    return str(candidate)
            errors.append(f"{endpoint}: no public IPv4")
        except (requests.RequestException, TypeError, ValueError) as error:
            errors.append(f"{endpoint}: {type(error).__name__}: {error}")
            logger.warning(f"CDP DoH endpoint failed for {host}: {endpoint}: {error}")
    detail = "; ".join(errors)
    raise RuntimeError(f"all DoH endpoints failed for {host}: {detail}")


class CdpEastmoneyClient:
    """Playwright 驱动的东方财富公告客户端（CDP 模式）。

    浏览器**懒启动**：首次网络调用时才拉起 Chrome，`close()` 负责回收。
    """

    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 30000,
        *,
        rate_limit_per_minute: int | None = None,
        bypass_system_proxy: bool = True,
        content_max_retries: int = 4,
        content_retry_backoff_seconds: float = 10.0,
        page_cache_root: Path | None = None,
    ) -> None:
        self.base_url = config.eastmoney.base_url
        self.headless = headless
        self.timeout_ms = timeout_ms
        requests_per_minute = (
            config.eastmoney.rate_limit_per_minute
            if rate_limit_per_minute is None
            else rate_limit_per_minute
        )
        self._min_interval = 60.0 / max(1, requests_per_minute)
        self.bypass_system_proxy = bypass_system_proxy
        self._content_max_retries = max(1, content_max_retries)
        self._content_retry_backoff_seconds = max(
            0.0, content_retry_backoff_seconds
        )
        self._last_call: float = 0.0
        self._playwright = None
        self._browser = None
        self._page: object | None = None
        self._content_host_ip: str | None = None
        self.page_cache_root = page_cache_root

    def _ensure_direct_host_mapping(self) -> None:
        if not self.bypass_system_proxy or self._content_host_ip is not None:
            return
        self._content_host_ip = _resolve_public_ipv4(_CONTENT_HOST)
        logger.info(f"CDP direct DNS: {_CONTENT_HOST} -> {self._content_host_ip}")

    def _launch_options(self) -> dict:
        options = {"headless": self.headless}
        if self.bypass_system_proxy:
            # This browser is dedicated to Eastmoney. Avoid inheriting the
            # Windows WinINET proxy, which can independently drop API tunnels.
            options["args"] = ["--no-proxy-server"]
            if self._content_host_ip:
                options["args"].append(
                    "--host-resolver-rules="
                    f"MAP {_CONTENT_HOST} {self._content_host_ip}"
                )
        return options

    # ── 浏览器生命周期（懒启动） ──────────────────────────────

    def _ensure_page(self):
        """启动 Chrome 并固定停留到东财公告页；返回 Playwright page。"""
        if self._page is not None:
            return self._page
        # 延迟导入：非 CDP 场景（http 后端 / 单元测试）不强制要求安装 playwright
        from playwright.sync_api import sync_playwright

        self._ensure_direct_host_mapping()
        self._playwright = sync_playwright().start()
        launch_options = self._launch_options()
        try:
            self._browser = self._playwright.chromium.launch(
                channel="chrome", **launch_options,
            )
        except Exception as e:
            # 系统 Chrome 不可用（版本过旧等）时回退到 Playwright 自带 chromium
            logger.warning(f"CDP 用系统 Chrome 启动失败({e})，回退 Playwright chromium")
            self._browser = self._playwright.chromium.launch(**launch_options)

        self._page = self._browser.new_page()
        self._page.goto(_HOME_URL, timeout=self.timeout_ms, wait_until="domcontentloaded")
        logger.info(f"CDP Chrome 就绪: {self._page.title()}")
        # 进程退出（含异常路径）时回收浏览器，避免子进程残留
        atexit.register(self.close)
        return self._page

    def close(self) -> None:
        """关闭浏览器，释放资源（幂等）。"""
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception:
            pass
        self._playwright = self._browser = self._page = None

    def __enter__(self) -> "CdpEastmoneyClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── 核心请求 ──────────────────────────────────────────────

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def _fetch_json(self, url: str) -> dict:
        """Navigate Chrome to an API URL and parse its JSON response."""
        page = self._ensure_page()
        try:
            self._rate_limit()
            response = page.goto(
                url,
                timeout=self.timeout_ms,
                wait_until="domcontentloaded",
            )
            if response is None:
                raise RuntimeError("CDP navigation returned no response")
            if not response.ok:
                raise RuntimeError(f"HTTP {response.status}")
            return response.json()
        except Exception as e:
            logger.warning(f"CDP fetch 失败: {url}: {e}")
            raise

    # ── 与 EastmoneyClient 同构的公开接口 ────────────────────

    def fetch_binary(
        self, url: str, *, timeout_ms: int = 90000
    ) -> tuple[bytes, str | None]:
        """Fetch an attachment through the CDP client's direct network context."""
        page = self._ensure_page()
        self._rate_limit()
        response = page.request.get(
            url,
            timeout=timeout_ms,
            fail_on_status_code=True,
        )
        return response.body(), response.headers.get("content-type")

    def fetch_announcements(
        self,
        stock_code: str = "",
        start_date: str = "",
        end_date: str = "",
        page_size: int = 50,
        page_index: int = 1,
        ann_type: str = "A",
    ) -> dict:
        """获取公告列表（单页）。返回 API 原始 JSON dict，失败返回空结构。"""
        params = {"page_size": page_size, "page_index": page_index, "ann_type": ann_type}
        if stock_code:
            params["stock_list"] = stock_code.replace(".SH", "").replace(".SZ", "")
        if start_date:
            params["begin_time"] = start_date
        if end_date:
            params["end_time"] = end_date
        url = f"{self.base_url}?{urlencode(params)}"
        try:
            return self._fetch_json(url)
        except Exception:
            return {"data": {"list": []}}

    def fetch_all_announcements(
        self,
        stock_code: str = "",
        start_date: str = "",
        end_date: str = "",
        max_pages: int = 20,
    ) -> list[dict]:
        """分页获取全部公告。与 HTTP 版相同的 total_hits 分页逻辑。"""
        all_items = []
        for page in range(1, max_pages + 1):
            result = self.fetch_announcements(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                page_index=page,
            )
            data = result.get("data", {})
            items = data.get("list", []) if isinstance(data, dict) else []
            if not items:
                break
            all_items.extend(items)
            # 兼容 API 响应字段变更：旧版 total_page / 新版 total_hits
            total_pages = 0
            if isinstance(data, dict):
                total_pages = data.get("total_page") or 0
                total_hits = data.get("total_hits") or 0
                if not total_pages and total_hits:
                    total_pages = (total_hits + 49) // 50
            if page >= total_pages:
                break
        return all_items

    def fetch_announcement_content(self, art_code: str) -> dict:
        """获取单条公告正文 + PDF 链接。返回 data dict，失败返回空 dict。"""
        from src.pipeline.announcement_content import (
            assemble_content_pages,
            expected_content_pages,
        )

        pages = []
        expected_pages = 1
        page_cache = None
        cache_hits = 0
        try:
            for page_index in range(1, 501):
                query = {
                    "art_code": art_code,
                    "client_source": "web",
                    "page_index": page_index,
                }
                url = f"{_CONTENT_URL}?{urlencode(query)}"
                payload = None
                if page_cache is not None and page_index > 1:
                    cached = page_cache.read(page_index)
                    if cached is not None:
                        pages.append(cached)
                        cache_hits += 1
                        if page_index >= expected_pages:
                            break
                        continue
                for attempt in range(1, self._content_max_retries + 1):
                    try:
                        payload = self._fetch_json(url)
                        break
                    except Exception as error:
                        if attempt >= self._content_max_retries:
                            raise
                        delay = self._content_retry_backoff_seconds * attempt
                        error_text = str(error)
                        if "HTTP 429" in error_text or "HTTP 567" in error_text:
                            delay = max(delay, 120.0)
                        logger.warning(
                            f"Retrying CDP announcement content {art_code} page "
                            f"{page_index} after {delay:.1f}s"
                        )
                        self.close()
                        time.sleep(delay)
                assert payload is not None
                data = payload.get("data") or {}
                if not data:
                    break
                pages.append(data)
                if page_index == 1:
                    expected_pages = expected_content_pages(data)
                    if expected_pages > 500:
                        raise ValueError("unreasonable content page count")
                    if self.page_cache_root is not None:
                        from src.pipeline.content_page_cache import ContentPageCache

                        page_cache = ContentPageCache(self.page_cache_root, art_code, data)
                if page_cache is not None:
                    page_cache.write(page_index, data)
                if page_index >= expected_pages:
                    break
        except Exception as error:
            logger.warning(f"CDP content API error for {art_code}: {error}")
        if cache_hits:
            logger.info(f"CDP content checkpoint {art_code}: reused {cache_hits} pages")
        return assemble_content_pages(pages, expected_pages=expected_pages)

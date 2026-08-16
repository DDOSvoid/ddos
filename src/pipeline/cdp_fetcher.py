"""CDP 模式东方财富公告客户端 — 用 Playwright 驱动系统 Chrome 抓取。

与 `EastmoneyClient`（HTTP requests 版）**接口完全同构**，产出同样的中间数据，
让 `Fetcher.run()` 的富化/入库逻辑无需改动。

实现原理:
  - 用 Playwright 启动系统 Chrome（channel="chrome"，不下载浏览器）
  - 固定停留在东财公告页 `data.eastmoney.com/notices/` 建立同源上下文
  - 列表与正文都通过 `page.evaluate(fetch)` 在**浏览器内部**请求 API
    （真实 UA / cookie / origin / TLS 指纹），拿到与 HTTP 版完全一致的结构化 JSON

用法:
    client = CdpEastmoneyClient()
    items = client.fetch_all_announcements("000009", "2026-08-05", "2026-08-12")
    content = client.fetch_announcement_content("AN202608071827755919")
    client.close()
"""

import atexit
import time
from typing import Optional
from urllib.parse import urlencode

from loguru import logger

from src.config import config

# 正文 API（与 EastmoneyClient.fetch_announcement_content 一致）
_CONTENT_URL = "https://np-cnotice-stock.eastmoney.com/api/content/ann"
# 固定停留页：用于建立 data.eastmoney.com 同源上下文，浏览器内 fetch 才不跨域
_HOME_URL = "https://data.eastmoney.com/notices/"

# 浏览器内 fetch 的 JS 片段：请求 URL 并解析 JSON。
# 注意: 不能带 credentials: "include" —— 跨域 fetch 时服务端若不返回
# Access-Control-Allow-Credentials 会直接 Failed to fetch（CORS 拦截）。
_FETCH_JSON_JS = """
async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error("HTTP " + r.status);
  return await r.json();
}
"""


class CdpEastmoneyClient:
    """Playwright 驱动的东方财富公告客户端（CDP 模式）。

    浏览器**懒启动**：首次网络调用时才拉起 Chrome，`close()` 负责回收。
    """

    def __init__(self, headless: bool = True, timeout_ms: int = 30000) -> None:
        self.base_url = config.eastmoney.base_url
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._min_interval = 60.0 / max(1, config.eastmoney.rate_limit_per_minute)
        self._last_call: float = 0.0
        self._playwright = None
        self._browser = None
        self._page: Optional[object] = None

    # ── 浏览器生命周期（懒启动） ──────────────────────────────

    def _ensure_page(self):
        """启动 Chrome 并固定停留到东财公告页；返回 Playwright page。"""
        if self._page is not None:
            return self._page
        # 延迟导入：非 CDP 场景（http 后端 / 单元测试）不强制要求安装 playwright
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(
                channel="chrome", headless=self.headless,
            )
        except Exception as e:
            # 系统 Chrome 不可用（版本过旧等）时回退到 Playwright 自带 chromium
            logger.warning(f"CDP 用系统 Chrome 启动失败({e})，回退 Playwright chromium")
            self._browser = self._playwright.chromium.launch(headless=self.headless)

        self._page = self._browser.new_page()
        self._page.goto(_HOME_URL, timeout=self.timeout_ms, wait_until="domcontentloaded")
        logger.info(f"CDP Chrome 就绪: {self._page.title()}")
        # 进程退出（含异常路径）时回收浏览器，避免子进程残留
        atexit.register(self.close)
        return self._page

    def close(self) -> None:
        """关闭浏览器，释放资源（幂等）。"""
        for obj in (self._browser, self._playwright):
            try:
                if obj is not None:
                    obj.close()
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
        """在页面上下文内用浏览器 fetch 请求 API 并解析 JSON。"""
        page = self._ensure_page()
        try:
            self._rate_limit()
            return page.evaluate(_FETCH_JSON_JS, url)
        except Exception as e:
            logger.warning(f"CDP fetch 失败: {url}: {e}")
            raise

    # ── 与 EastmoneyClient 同构的公开接口 ────────────────────

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
        url = f"{_CONTENT_URL}?{urlencode({'art_code': art_code, 'client_source': 'web', 'page_index': 1})}"
        try:
            data = self._fetch_json(url)
            return data.get("data") or {}
        except Exception:
            return {}

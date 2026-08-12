"""数据获取层 — Tushare 元数据 + 东方财富公告原文。

Tushare (已验证可用):
  - stock_basic: 股票基础信息 ✅
  - daily: 日线行情 ✅
  - income/balancesheet: 财务数据 ✅ (评分基准)
  - disclosure: 定期报告披露时间表 ❌ (需更高积分)

东方财富 (公告主通道):
  - 公告列表（标题、摘要、类型）
  - 公告原文 HTML（可选，正文提取）
"""

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import requests
from loguru import logger
from sqlalchemy.orm import Session

from src.config import config
from src.database.engine import get_engine
from src.database.models import Announcement, Company
from src.database.repository import AnnouncementRepository, CompanyRepository
from src.utils.text_utils import clean_chinese_text


# ── 数据模型 ───────────────────────────────────────────────────


@dataclass
class RawAnnouncement:
    """原始公告数据（中间格式）。"""
    announcement_id: str
    stock_code: str
    title: str
    full_text: Optional[str] = None
    pdf_url: Optional[str] = None
    published_date: date | None = None
    source_url: Optional[str] = None
    raw_json: Optional[str] = None  # 原始 API 响应


# ── Tushare 客户端 ──────────────────────────────────────────────


class TushareClient:
    """Tushare Pro API 封装。

    文档: https://tushare.pro/document/2
    """

    def __init__(self, token: str | None = None):
        import tushare as ts

        token = token or config.tushare.token
        if not token:
            raise ValueError("Tushare token required. Set TUSHARE_TOKEN in .env")
        ts.set_token(token)
        self.pro = ts.pro_api()
        self._min_interval = 60.0 / config.tushare.rate_limit_per_minute
        self._last_call: float = 0.0

    def _rate_limit(self) -> None:
        """简单速率限制。"""
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def _call(self, func_name: str, **kwargs) -> "pd.DataFrame":
        """带重试和速率限制的 API 调用。"""
        import pandas as pd

        func = getattr(self.pro, func_name)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                result = func(**kwargs)
                if isinstance(result, pd.DataFrame) and not result.empty:
                    return result
                return pd.DataFrame()
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Tushare {func_name} attempt {attempt + 1} failed: {e}, retry in {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"Tushare {func_name} failed after {max_retries} attempts: {e}")
                    raise
        return pd.DataFrame()

    def get_stock_basic(self, exchange: str = "") -> "pd.DataFrame":
        """获取全部 A 股基础信息。"""
        return self._call("stock_basic",
            exchange=exchange,
            list_status="L",
            fields="ts_code,symbol,name,area,industry,list_date",
        )

    def get_disclosure(
        self,
        ts_code: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> "pd.DataFrame":
        """获取定期报告披露时间表。

        返回: ts_code, ann_date, end_date, pre_date, actual_date, modify_date
        """
        return self._call("disclosure",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )

    def get_income(
        self,
        ts_code: str,
        start_date: str = "",
        end_date: str = "",
        report_type: str = "",
    ) -> "pd.DataFrame":
        """获取利润表数据。

        report_type: 1=合并, 2=单季,  ''=默认
        """
        kwargs = {"ts_code": ts_code}
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        if report_type:
            kwargs["report_type"] = report_type
        return self._call("income", **kwargs)

    def get_balancesheet(
        self,
        ts_code: str,
        start_date: str = "",
        end_date: str = "",
    ) -> "pd.DataFrame":
        """获取资产负债表。"""
        kwargs = {"ts_code": ts_code}
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        return self._call("balancesheet", **kwargs)

    def get_daily(
        self,
        ts_code: str = "",
        trade_date: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> "pd.DataFrame":
        """获取日线行情。"""
        kwargs = {}
        if ts_code:
            kwargs["ts_code"] = ts_code
        if trade_date:
            kwargs["trade_date"] = trade_date
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        return self._call("daily", **kwargs)

    def get_namechange(self, ts_code: str = "") -> "pd.DataFrame":
        """获取股票曾用名。"""
        return self._call("namechange", ts_code=ts_code)


# ── 东方财富公告客户端 ─────────────────────────────────────────


class EastmoneyClient:
    """东方财富公告接口封装。

    公告列表 API: https://np-anotice-stock.eastmoney.com/api/security/ann
    """

    def __init__(self):
        self.base_url = config.eastmoney.base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": config.eastmoney.user_agent,
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://data.eastmoney.com/",
        })
        self._min_interval = 60.0 / config.eastmoney.rate_limit_per_minute
        self._last_call: float = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def fetch_announcements(
        self,
        stock_code: str = "",
        start_date: str = "",
        end_date: str = "",
        page_size: int = 50,
        page_index: int = 1,
        ann_type: str = "A",
    ) -> dict:
        """获取公告列表。

        Args:
            stock_code: 6位股票代码（不含后缀），多个用逗号分隔
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            page_size: 每页数量（最大 50）
            page_index: 页码
            ann_type: 公告类型 (A=全部)

        Returns:
            API 原始 JSON dict
        """
        params = {
            "page_size": page_size,
            "page_index": page_index,
            "ann_type": ann_type,
        }
        if stock_code:
            params["stock_list"] = stock_code.replace(".SH", "").replace(".SZ", "")
        if start_date:
            params["begin_time"] = start_date
        if end_date:
            params["end_time"] = end_date

        self._rate_limit()
        try:
            resp = self.session.get(self.base_url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Eastmoney API error: {e}")
            return {"data": {"list": []}}

    def fetch_all_announcements(
        self,
        stock_code: str = "",
        start_date: str = "",
        end_date: str = "",
        max_pages: int = 20,
    ) -> list[dict]:
        """分页获取全部公告。

        Returns:
            公告 dict 列表
        """
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
            total_pages = data.get("total_page", 0) if isinstance(data, dict) else 0
            if page >= total_pages:
                break
            time.sleep(0.3)  # 礼貌爬取
        return all_items

    def fetch_announcement_content(self, art_code: str) -> dict:
        """获取单条公告内容（正文 + PDF 链接）。

        列表接口只返回元数据（art_code/title/notice_date），正文需按 art_code
        单独调用内容接口。

        Args:
            art_code: 公告编号（列表接口返回的 art_code）

        Returns:
            data dict，含 notice_content / attach_url_web / attach_url；失败返回空 dict
        """
        url = "https://np-cnotice-stock.eastmoney.com/api/content/ann"
        params = {"art_code": art_code, "client_source": "web", "page_index": 1}
        self._rate_limit()
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json().get("data") or {}
        except Exception as e:
            logger.warning(f"Eastmoney content API error for {art_code}: {e}")
            return {}


# ── 主数据获取器 ────────────────────────────────────────────────


def _stock_code_short(code: str) -> str:
    """000001.SZ → 000001"""
    return code.replace(".SH", "").replace(".SZ", "")


def _format_date(d: date | str) -> str:
    """date → YYYYMMDD 字符串。"""
    if isinstance(d, str):
        return d.replace("-", "")
    return d.strftime("%Y%m%d")


def _format_date_dash(d: date | str) -> str:
    """date → YYYY-MM-DD 字符串。"""
    if isinstance(d, str):
        if len(d) == 8:
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        return d
    return d.strftime("%Y-%m-%d")


class Fetcher:
    """主数据获取器 — 结合 Tushare + 东方财富。

    每日运行:
        fetcher = Fetcher()
        count = fetcher.run(target_date=date.today())
    """

    def __init__(
        self,
        tushare_token: str | None = None,
    ) -> None:
        self.ts = TushareClient(token=tushare_token)
        self.em = EastmoneyClient()

    def run(
        self,
        target_date: date | None = None,
        lookback_days: int | None = None,
    ) -> int:
        """执行每日公告获取。"""
        if target_date is None:
            target_date = date.today()
        if lookback_days is None:
            lookback_days = config.pipeline.fetch_lookback_days

        start_date = target_date - timedelta(days=lookback_days)
        logger.info(f"Fetcher: {start_date} → {target_date}")

        engine = get_engine()
        with Session(engine) as session:
            # 1. 获取跟踪公司列表
            companies = CompanyRepository.get_tracked(session)
            if not companies:
                logger.warning("No tracked companies found. Run seed_database.py first.")
                return 0

            tracked_codes = [c.stock_code for c in companies]
            logger.info(f"Processing {len(tracked_codes)} tracked companies")

            # 2. 从东方财富获取公告
            all_raw: list[RawAnnouncement] = []

            for code in tracked_codes[:10]:  # MVP: 限制 10 只试点
                short_code = _stock_code_short(code)
                try:
                    items = self.em.fetch_all_announcements(
                        stock_code=short_code,
                        start_date=_format_date_dash(start_date),
                        end_date=_format_date_dash(target_date),
                        max_pages=3,  # 每只股票最多 3 页
                    )
                    for item in items:
                        art_code = str(item.get("art_code", ""))
                        raw = RawAnnouncement(
                            announcement_id=art_code,
                            stock_code=code,
                            title=item.get("title", ""),
                            full_text=None,
                            pdf_url=None,
                            published_date=self._parse_em_date(item.get("notice_date", "")),
                            source_url=art_code,
                            raw_json=None,
                        )
                        # 富化：按 art_code 抓取正文与 PDF 链接（失败降级为列表摘要，不中断整批）
                        if config.pipeline.fetch_full_text and art_code:
                            content = self.em.fetch_announcement_content(art_code)
                            raw.full_text = (
                                clean_chinese_text(content.get("notice_content", ""))
                                or item.get("notice_content", "")
                            )
                            raw.pdf_url = (
                                content.get("attach_url_web")
                                or content.get("attach_url")
                            )
                        else:
                            raw.full_text = item.get("notice_content", "")
                        all_raw.append(raw)
                except Exception as e:
                    logger.warning(f"Failed to fetch announcements for {code}: {e}")

            logger.info(f"Fetched {len(all_raw)} raw announcements from Eastmoney")

            # 3. 去重并写入数据库
            records = []
            seen_ids: set[str] = set()
            for raw in all_raw:
                if not raw.announcement_id or raw.announcement_id in seen_ids:
                    continue
                seen_ids.add(raw.announcement_id)
                company_id = None
                for c in companies:
                    if c.stock_code == raw.stock_code:
                        company_id = c.id
                        break
                if company_id is None:
                    continue
                records.append({
                    "company_id": company_id,
                    "announcement_id": raw.announcement_id,
                    "title": raw.title,
                    "full_text": raw.full_text,
                    "pdf_url": raw.pdf_url,
                    "published_date": raw.published_date or target_date,
                    "source_url": raw.source_url,
                    "processing_status": "fetched",
                })

            count = AnnouncementRepository.bulk_upsert(session, records)
            session.commit()
            logger.info(f"Fetcher done: {count} new announcements stored")

        return count

    @staticmethod
    def _parse_em_date(date_str: str) -> date | None:
        """解析东方财富日期格式。"""
        if not date_str:
            return None
        for fmt in ["%Y-%m-%d", "%Y%m%d", "%Y-%m-%d %H:%M:%S"]:
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue
        return None


# ── 上市公司财务数据同步 ────────────────────────────────────────


class FinancialDataSyncer:
    """定期同步上市公司财务数据（营收、净资产等）— 用于评分基准。"""

    def __init__(self, tushare_token: str | None = None) -> None:
        self.ts = TushareClient(token=tushare_token)

    def sync_company_financials(self, ts_code: str) -> dict | None:
        """拉取单只股票的最新财务数据。"""
        import pandas as pd

        try:
            df = self.ts.get_income(ts_code=ts_code, report_type="1")
            if df.empty:
                return None

            latest = df.iloc[0]
            return {
                "annual_revenue": latest.get("revenue"),
                "net_profit": latest.get("n_income"),
            }
        except Exception as e:
            logger.warning(f"Failed to sync financials for {ts_code}: {e}")
            return None

    def sync_all_tracked(self) -> int:
        """同步所有跟踪公司的财务数据。"""
        engine = get_engine()
        count = 0
        with Session(engine) as session:
            companies = CompanyRepository.get_tracked(session)
            for company in companies:
                fin_data = self.sync_company_financials(company.stock_code)
                if fin_data:
                    CompanyRepository.upsert(session, company.stock_code, **fin_data)
                    count += 1
                time.sleep(0.3)  # 速率限制
            session.commit()
        logger.info(f"Synced financials for {count}/{len(companies)} companies")
        return count

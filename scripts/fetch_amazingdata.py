#!/usr/bin/env python
"""AmazingData 独立数据拉取脚本 — 银河证券星耀数智。

从 AmazingData 拉取跟踪股票的日线/财务/龙虎榜/两融数据，以 CSV 分区落盘到
`data/amazingdata/`。**不接入公告分析管线**：不改数据库表结构、不进评分/报告，
仅做原始数据补充。后续要接入时再按需读取这些 CSV。

注意事项（均为实测结论）：
  1. AmazingData 依赖重（numba/llvmlite/tables），且其自带 numpy 2.4.x —— 本脚本
     必须**先把 vendor 目录插入 sys.path 并 import AmazingData**，之后所有 pandas/
     numpy 都用 vendor 版，避免与主管线 venv 的 numpy 2.5 冲突。
  2. `MarketData.query_kline` 的 `period` 必须传**整数**（`Period.day.value`），
     传枚举 `Period.day` 会因 `period == Period.day.value` 恒为 False 而落入坏分支
     报 `TypeError: 'NoneType' object cannot be interpreted as an integer`。
  3. InfoData 系列默认在 `D:\\AmazingData_local_data\\` 写缓存，这里统一把 local_path
     指到项目内 `data/amazingdata/_ad_cache/`，保持落盘可控。

用法:
    python scripts/fetch_amazingdata.py                        # 默认: 跟踪股, 近 1 年
    python scripts/fetch_amazingdata.py --begin 2025-08-13 --end 2026-08-13
    python scripts/fetch_amazingdata.py --codes 000009.SZ,000049.SZ --types kline
    python scripts/fetch_amazingdata.py --limit 5              # 只拉前 5 只(调试)
"""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# ── 路径 ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "amazingdata"
CACHE_DIR = DATA_DIR / "_ad_cache"
DB_URL = f"sqlite:///{PROJECT_ROOT / 'data' / 'ddos.db'}"

# 各数据类型 → 落盘子目录 + 说明
TYPES = {
    "kline": ("kline_daily", "日线(前复权)"),
    "income": ("financial_income", "利润表"),
    "balance": ("financial_balance", "资产负债表"),
    "cashflow": ("financial_cashflow", "现金流量表"),
    "longhubang": ("long_hu_bang", "龙虎榜"),
    "margin": ("margin_detail", "两融明细"),
}


def load_env() -> None:
    """加载 .env（gitignored）。"""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")


def get_tracked_codes() -> list[str]:
    """从公告库读取跟踪公司代码（只读，纯 SQL，不 import pandas/numpy）。"""
    import sqlalchemy
    from sqlalchemy import create_engine, text

    engine = create_engine(DB_URL)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT stock_code FROM companies WHERE is_tracked = 1 ORDER BY stock_code")
            ).fetchall()
        codes = [r[0] for r in rows]
    finally:
        engine.dispose()
    return codes


def _login(ad, username: str, password: str, host: str, port: int):
    ad.login(username=username, password=password, host=host, port=port)


def _save_result(code: str, df, out_dir: Path) -> int:
    """保存一个代码的结果 DataFrame 到 CSV，返回行数。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{code}.csv"
    if df is None or (hasattr(df, "empty") and bool(df.empty)):
        # 空结果（含该股无此数据，如非两融标的）也写空文件占位，便于 coverage 判断
        # 注意: 不能引用 pd（pandas 只在 main() 内导入），直接写空文件
        path.write_text("", encoding="utf-8-sig")
        return 0
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return len(df)


def _extract_code_rows(result, code: str):
    """把 SDK 返回（dict[code, DataFrame] 或 直接 DataFrame）规范化为 DataFrame。

    SDK 返回形态不统一：
      - query_kline / get_income / get_margin_detail → {code: DataFrame}
      - get_long_hu_bang → 直接一个 DataFrame（含 MARKET_CODE 列）
    这里统一抽成 DataFrame。
    """
    if result is None:
        return None
    if isinstance(result, dict):
        # 注意不能用 `or` 链——DataFrame 的真值判断是歧义的
        for key in (code, code.lower(), code.upper()):
            if key in result:
                return result[key]
        return None
    # 直接 DataFrame：若含 MARKET_CODE 列，按 code 过滤；否则原样返回
    if hasattr(result, "columns"):
        if "MARKET_CODE" in result.columns:
            sub = result[result["MARKET_CODE"].astype(str) == code]
            return sub
        return result
    return None


def main():
    parser = argparse.ArgumentParser(description="AmazingData 独立数据拉取（不接入公告管线）")
    parser.add_argument("--begin", default="", help="开始日期 YYYY-MM-DD（默认近 1 年）")
    parser.add_argument("--end", default="", help="结束日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--codes", default="", help="逗号分隔的股票代码；缺省=跟踪股")
    parser.add_argument("--types", default=",".join(TYPES),
                        help="数据类型: kline,income,balance,cashflow,longhubang,margin")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="落盘根目录")
    parser.add_argument("--limit", type=int, default=0, help="只拉前 N 只（调试用）")
    args = parser.parse_args()

    load_env()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = data_dir / "_ad_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 日期范围
    end = args.end or date.today().isoformat()
    begin = args.begin or (date.today() - timedelta(days=365)).isoformat()
    begin_int = int(begin.replace("-", ""))
    end_int = int(end.replace("-", ""))
    print(f"[amazingdata] 范围: {begin} ~ {end}")

    # 股票代码
    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    else:
        codes = get_tracked_codes()
    if args.limit:
        codes = codes[: args.limit]
    print(f"[amazingdata] 股票数: {len(codes)}")

    # 数据类型
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    unknown = set(types) - set(TYPES)
    if unknown:
        sys.exit(f"未知数据类型: {sorted(unknown)}，可选: {list(TYPES)}")

    # ── 关键: 先插 vendor 路径，用 vendor 的 numpy/pandas 加载 AmazingData ──
    vendor_path = os.getenv("AMAZINGDATA_PATH", "")
    if not vendor_path or not Path(vendor_path).exists():
        sys.exit("缺少 AMAZINGDATA_PATH 环境变量（指向 AmazingData SDK 解包目录）")
    if str(vendor_path) not in sys.path:
        sys.path.insert(0, vendor_path)
    import AmazingData as ad  # noqa: E402
    import pandas as pd  # noqa: E402
    print(f"[amazingdata] SDK {getattr(ad, '__version__', '?')} 加载完成 (numpy {pd.__version__} 来自 vendor)")

    username = os.getenv("AMAZINGDATA_USERNAME", "")
    password = os.getenv("AMAZINGDATA_PASSWORD", "")
    host = os.getenv("AMAZINGDATA_HOST", "101.230.159.234")
    port = int(os.getenv("AMAZINGDATA_PORT", "8600"))
    if not username or not password:
        sys.exit("缺少 AMAZINGDATA_USERNAME/PASSWORD 环境变量（见 .env）")

    # ── 登录 + 引导（带重试：TGW 登录是异步的，会话未就绪时首个查询可能拿 None）──
    base = market = info = None
    for attempt in range(1, 5):
        try:
            _login(ad, username, password, host, port)
            base = ad.BaseData()
            calendar = base.get_calendar()
            if calendar is None or len(calendar) == 0:
                raise RuntimeError("get_calendar 返回空")
            market = ad.MarketData(calendar)
            info = ad.InfoData()
            print(f"[amazingdata] 登录成功 (第 {attempt} 次尝试)")
            break
        except Exception as e:
            print(f"  ⚠ 登录/引导失败 (第 {attempt} 次): {e}")
            if attempt == 4:
                sys.exit(f"AmazingData 登录/引导多次重试仍失败: {e}")
            time.sleep(attempt * 3)  # 退避: 3s/6s/9s

    # ── 逐类型拉取 ─────────────────────────────────────────────
    manifest = {
        "begin": begin, "end": end, "codes": len(codes), "types": types,
        "pulled_at": datetime.now().isoformat(timespec="seconds"),
        "results": {},
    }
    t0 = time.time()

    for t in types:
        sub_dir = TYPES[t][0]
        out_dir = data_dir / sub_dir
        total_rows = 0
        ok = fail = 0

        for code in codes:
            try:
                if t == "kline":
                    res = market.query_kline(
                        [code], begin_date=begin_int, end_date=end_int,
                        period=ad.constant.Period.day.value,
                    )
                    df = _extract_code_rows(res, code)
                    n = _save_result(code, df, out_dir)
                elif t in ("income", "balance", "cashflow"):
                    fn = {"income": "get_income", "balance": "get_balance_sheet",
                          "cashflow": "get_cash_flow"}[t]
                    res = getattr(info, fn)([code], local_path=str(cache_dir) + os.sep, is_local=False)
                    df = _extract_code_rows(res, code)
                    n = _save_result(code, df, out_dir)
                elif t == "longhubang":
                    # 必须限定日期范围：不限时返回全历史，SDK 会踩到脏数据
                    # （strptime() 收到 float），且数据量也不必要地大
                    res = info.get_long_hu_bang(
                        [code], begin_date=begin_int, end_date=end_int,
                        local_path=str(cache_dir) + os.sep, is_local=False,
                    )
                    df = _extract_code_rows(res, code)
                    n = _save_result(code, df, out_dir)
                elif t == "margin":
                    res = info.get_margin_detail([code], local_path=str(cache_dir) + os.sep, is_local=False)
                    df = _extract_code_rows(res, code)
                    n = _save_result(code, df, out_dir)
                total_rows += n
                ok += 1
            except Exception as e:
                print(f"  ✗ {t} {code}: {e}")
                fail += 1

        elapsed = time.time() - t0
        manifest["results"][t] = {
            "dir": sub_dir, "rows": total_rows, "ok": ok, "fail": fail,
        }
        print(f"[amazingdata] {t:<10} ok={ok:>4} fail={fail:>3} rows={total_rows:>7} 累计 {elapsed:.0f}s")

    # 写 manifest
    manifest["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(data_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n[amazingdata] 完成 → {data_dir}")
    print(json.dumps(manifest["results"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Tushare 连通性测试 — 验证 token 和各接口可用性。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
import tushare as ts

token = os.getenv("TUSHARE_TOKEN")
print(f"Token: ...{token[-4:] if token else 'MISSING'}")

ts.set_token(token)
pro = ts.pro_api()

# 1. A 股列表
print("\n=== 1. A股列表 ===")
df = pro.stock_basic(exchange="", list_status="L",
                     fields="ts_code,symbol,name,area,industry,list_date")
print(f"上市公司总数: {len(df)}")
print(f"行业数量: {df['industry'].nunique()}")

# 2. 试点行业
print("\n=== 2. 试点行业 ===")
keywords = ["电气设备", "新能源", "光伏", "风电", "电池", "电力设备"]
mask = df["industry"].str.contains("|".join(keywords), na=False)
pilot = df[mask]
print(f"试点行业股票数: {len(pilot)}")
print(pilot[["ts_code", "name", "industry"]].head(10).to_string(index=False))

# 3. 定期报告披露时间表
print("\n=== 3. 披露时间表 ===")
df_disc = pro.disclosure(start_date="20260101", end_date="20260730")
print(f"2026年披露记录: {len(df_disc)}")
if not df_disc.empty:
    print(f"列: {list(df_disc.columns)}")
    print(df_disc.head(5).to_string(index=False))

# 4. 日线数据（宁德时代）
print("\n=== 4. 日线行情 (300750.SZ 宁德时代) ===")
df_daily = pro.daily(ts_code="300750.SZ", start_date="20260701", end_date="20260730")
print(f"7月日线记录: {len(df_daily)}")
if not df_daily.empty:
    cols = ["trade_date", "open", "high", "low", "close", "pct_chg", "vol"]
    available = [c for c in cols if c in df_daily.columns]
    print(df_daily[available].head(5).to_string(index=False))

# 5. 积分使用情况
print("\n=== 5. 积分使用 ===")
try:
    df_usage = pro.query("trade_cal", exchange="SSE", start_date="20260101", end_date="20261231", is_open="1")
    print(f"2026年上交所交易日数: {len(df_usage)}")
except Exception:
    pass

print("\n✅ Tushare 连通性测试完成")

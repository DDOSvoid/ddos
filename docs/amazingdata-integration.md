# AmazingData 数据接入与数据结构改进方案

> 状态: **待评审**(先出方案,未改代码)
> 日期: 2026-08-13
> 前提: 用户持有银河证券 AmazingData SDK 1.1.7(解包于 `D:\workspace_hyy\tmp\aido\vendor\amazingdata_inspect_pkg`,只读)

---

## 1. 现状与痛点

当前系统 7 张表(`companies / announcements / classifications / extracted_fields / scores / daily_reports / pipeline_runs`),数据源为 Tushare + 东方财富。**实测**存在三个明确短板:

| # | 短板 | 影响 | 证据 |
|---|------|------|------|
| 1 | **无行情数据** | `scores.market_reaction`(市场反应)字段**从未填充**,评分无法"预测→复盘"闭环 | 本次运行报告 composite=+0.023 后无任何市场验证 |
| 2 | **companies 财务字段全 NULL** | `annual_revenue / net_assets / market_cap / total_shares` 均为空,magnitude(强度)评分缺真实归一化基准 | `FinancialDataSyncer` 存在但管线从不调用 |
| 3 | **surprise 无预期数据** | `surprise` 恒为默认 0.5,无法区分"符合预期/超预期/低预期" | 评分明细中 surprise 无来源 |

## 2. 可行性结论(已实测)

- ✅ SDK **1.1.7** 的 WHEEL 标签为 `py3-none-any`(纯 Python),**兼容现有 Python 3.14 venv**(网上流传的 cp312 是旧版本 1.1.6,不适用)
- ✅ vendor 目录为**自包含 Python 3.14 环境**(665MB,含 `tgw-1.0.8.7` 传输库 / numba / llvmlite / pandas 3.0.3 / blosc2 / tables),将目录加入 `sys.path` 即可导入,已验证:
  ```python
  sys.path.insert(0, r'D:\workspace_hyy\tmp\aido\vendor\amazingdata_inspect_pkg')
  import AmazingData as ad   # ✅ v1.1.7
  ```
- ✅ **登录参数已确认**(来自 `D:\workspace_hyy\tmp\aido` 主项目的 `engine/data_source.py`,该账号已在其 ETF/期货策略中实盘使用):
  - host: `101.230.159.234` · port: `8600`
  - username / password: 见 gitignored 的 `.env`(`AMAZINGDATA_USERNAME` / `AMAZINGDATA_PASSWORD`)
  - 接入范式(照抄 aido 已验证代码):
    ```python
    import sys
    sys.path.insert(0, r'D:\workspace_hyy\tmp\aido\vendor\amazingdata_inspect_pkg')
    import AmazingData as ad
    ad.login(username=u, password=p, host="101.230.159.234", port=8600)
    base = ad.BaseData(); market = ad.MarketData(base.get_calendar()); info = ad.InfoData()
    ```
- ✅ **账号开通状态**: 无需担心"111 前缀"问题——该账号已在 aido 项目实盘调用 AmazingData,凭证有效。

## 3. AmazingData 能力盘点(实测 pyc 接口)

**MarketData**
- `query_kline` — K线(日/周/月 + 分钟线)
- `query_snapshot` — Level-1 实时快照

**InfoData**(本项目最相关)
- 财报: `get_balance_sheet / get_income / get_cash_flow`、`get_profit_notice`(业绩预告)、`get_profit_express`(业绩快报)、`get_dividend`(分红)、`get_right_issue`(配股)
- 市场情绪: `get_long_hu_bang`(龙虎榜)、`get_block_trading`(大宗交易)、`get_margin_summary` / `get_margin_detail`(融资融券)
- 股东: `get_share_holder`、`get_holder_num`(股东户数)、`get_equity_restricted`(限售)
- 行业: `get_industry_daily`、`get_industry_constituent`
- 其他: 国债收益率、可转债全套、基金净值

**BaseData**
- `get_calendar`(交易日历)、`get_code_list / get_code_info`、`get_adj_factor / get_backward_factor`(复权因子)、`get_hist_code_list`

## 4. 架构设计

**核心原则: 隔离运行。** AmazingData 依赖重(numba/llvmlite/tables 原生库),不应进入主管线进程。

```
                    ┌─────────────────────────────────────┐
                    │  scripts/sync_amazingdata.py        │
                    │  (独立进程, vendor 目录进 sys.path) │
                    │  AmazingData SDK → 增量拉取         │
                    └──────────────┬──────────────────────┘
                                   ▼ 写入同一 SQLite
   ┌───────────────┬───────────────┼──────────────┬────────────────┐
   ▼               ▼               ▼              ▼                ▼
daily_prices   financials     market_signals  adj_factors    (scores.market_reaction 回填)
   │               │               │                            ▲
   └───────────────┴───────────────┴────────────────────────────┘
                                   主管线(scorer/reporter)读取,不依赖 SDK
```

- **新增** `src/datasources/amazingdata_client.py`:SDK 封装(登录/重连/节流)
- **新增** `scripts/sync_amazingdata.py`:每日同步入口(增量、幂等)
- 主管线 `scorer.py / reporter.py` 只读新表,不 import AmazingData
- 通信介质 = 共享 SQLite 文件,天然解耦、可恢复

## 5. Schema 设计

### 5.1 新增表

**`daily_prices`** — 每公司每日行情(核心,支撑 market_reaction 与 surprise)
| 列 | 类型 | 说明 |
|----|------|------|
| id | Integer PK | |
| company_id | FK→companies | |
| trade_date | Date | 交易日 |
| open / high / low / close | Float | 价格 |
| volume | Float | 成交量(股) |
| amount | Float | 成交额(元) |
| adj_factor | Float | 复权因子(同步日写入) |
| source | String(20) | 'amazingdata' / 'tushare' |
| UNIQUE(company_id, trade_date) | | 幂等 |

**`financials`** — 财报时间序列(支撑 magnitude 归一化)
| 列 | 类型 | 说明 |
|----|------|------|
| id | Integer PK | |
| company_id | FK→companies | |
| report_period | Date | 报告期(如 2026-06-30) |
| ann_date | Date | 披露日 |
| revenue / net_profit | Float | 营收 / 归母净利(万元) |
| total_assets / total_liabilities | Float | 总资产 / 总负债(万元) |
| equity / roe / eps | Float | 净资产 / ROE / 每股收益 |
| gross_margin | Float | 毛利率 |
| UNIQUE(company_id, report_period) | | 幂等 |

**`market_signals`** — 龙虎榜/大宗/两融(辅助信号)
| 列 | 类型 | 说明 |
|----|------|------|
| id | Integer PK | |
| company_id | FK→companies | |
| signal_date | Date | 发生日 |
| signal_type | String(20) | dragon_tiger / block_trade / margin |
| amount | Float | 涉及金额(万元) |
| detail | Text(JSON) | 原始明细(买卖席位/融资余额等) |
| UNIQUE(company_id, signal_date, signal_type) | | 幂等 |

### 5.2 修改

1. **`scores.market_reaction` 激活**: 新增回填步骤,公告后第 1/3/5 个交易日相对前收的累计涨跌幅写入该字段(从 `daily_prices` 计算,含复权)。
2. **`companies` 补数据**: `market_cap / total_shares / annual_revenue / net_assets` 由同步脚本从 AmazingData 填(替代从未调用的 `FinancialDataSyncer`)。
3. *(可选)* `scorer.py` 的 `surprise` 维度: 有 `daily_prices` 后,可对比公告发布前 20 日累积涨幅判断是否已充分预期。

### 5.3 迁移方式

项目无 Alembic migration(靠 `init_db()` 建表)。做法:
- 新表: 直接在 `models.py` 定义 → `init_db()` 自动 `CREATE TABLE IF NOT EXISTS`(幂等)
- 现有表加列: 若需改动,SQLite 需手写 `ALTER TABLE`(在 `engine.py` 加一个轻量迁移钩子,避免全表重建)

## 6. 数据流

```
每日调度:
  1) run_pipeline.py          # 现有管线,产出公告评分
  2) sync_amazingdata.py      # 拉取当日行情/财务/情绪 → SQLite
  3) backfill_market_reaction # 用 daily_prices 回填昨日已评分公告
  4) (可选) reporter 增强     # 报告展示 market_reaction / 龙虎榜佐证
```

增量策略:
- 行情: 按交易日增量,`UNIQUE(company_id, trade_date)` 保证幂等;首次全量再按日增量
- 财务: 按 `report_period` 去重,`ann_date` 驱动增量
- 全市场 5543 只日线首次同步量级: 约 5500 × 250 交易日/年 ≈ 140 万行/年,SQLite 可承受;若长期累积建议 PostgreSQL

## 7. 实施阶段

| 阶段 | 内容 | 交付 |
|------|------|------|
| **Phase 0** | 确认登录 host/port、账号开通状态;连通性测试 | 可登录的验证脚本 |
| **Phase 1** | `amazingdata_client.py` + `daily_prices` 表 + `backfill_market_reaction` | 评分带市场验证闭环 |
| **Phase 2** | `financials` 表 + 同步脚本 + magnitude 归一化增强 | 评分强度真实化 |
| **Phase 3** | `market_signals` 表(龙虎榜/两融)+ 报告佐证栏 | 报告增强 |

## 8. 风险与待确认

1. ✅ **登录参数缺失** — **已解决**: host/port 已在 aido 主项目确认(`101.230.159.234:8600`),账号已在实盘使用。
2. ✅ **账号开通状态** — **已解决**: 该账号在 aido 项目中正常调用 AmazingData,凭证有效。
3. ⏳ **原生库环境**: `tgw` / numba / llvmlite 在 Windows 上的运行时依赖(firewall、VC++ runtime),Phase 1 连通性测试一并验证。
4. ⏳ **数据质量**: 该 SDK 数据可用于评分增强,但**不构成投资建议**;评分置信度与 `market_reaction` 的关联度需在 Phase 1 后用真实数据评估。
5. **安全**: 账号密码写入 gitignored 的 `.env`,**不得进 git**。
6. **隔离原则**: AmazingData 依赖重,不得直接进入主管线进程;且 `D:\workspace_hyy\tmp\aido` 属用户主项目(有 `AGENTS.md` 实盘保护区),接入代码只落在 `ddos` 项目内,**不触碰 aido 的任何文件**。

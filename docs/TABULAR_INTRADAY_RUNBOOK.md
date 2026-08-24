# 训练期15分钟行情回填运行手册

## 固定范围

- 数据角色：仅 `train`。
- 日期：2023-01-01 至 2024-12-31。
- 标的：训练期已接纳核心公告涉及的343只股票。
- 频率：AmazingData 15分钟，bar标签为区间起点。
- 用途：只生成公告前完整交易日的日级滞后聚合特征。
- 官方测试、隔离区和2026年前向数据：不查询。

## 运行命令

```powershell
.venv\Scripts\python.exe scripts\backfill_tabular_intraday.py `
  --source amazingdata `
  --start-date 2023-01-01 `
  --end-date 2024-12-31 `
  --state data\tabular\intraday_15m\state.amazingdata.json
```

状态文件：`data/tabular/intraday_15m/state.amazingdata.json`。

日志：

- `data/tabular/intraday_15m/amazingdata_download.stdout.log`
- `data/tabular/intraday_15m/amazingdata_download.stderr.log`

数据按 `raw/amazingdata_query_kline/<股票>/<年份>.parquet` 保存。每个完成块记录行数、时间范围和SHA-256；使用同一命令重启会跳过完成块并重试失败块。

## 已验证门禁

- 单标的2023、2024各242个交易日，每天固定16根bar。
- 2024全年242个交易日的开高低收与缓存日线100%一致。
- 成交量和成交额跨源中位比例为1，少数日期最高约差7%；因此不以分钟总量替代日线总量。
- `available_at_utc` 按bar起点加15分钟计算，最后一根14:45标签在15:00才可用。
- 源价格未复权；只计算单日内收益、实现波动、振幅、尾盘变化和日内量价占比。

Tushare曾完成单日冒烟，但年度请求权限为每小时1次，不用于全量回填。失败证据保留在 `data/tabular/intraday_15m/state.json`。

## 日级聚合

下载完成后运行：

```powershell
.venv\Scripts\python.exe scripts\build_tabular_intraday_aggregates.py
```

聚合器只接受完整16根bar的交易日，并生成日内收益、实现波动、下行波动、振幅、收盘位置、开盘/尾盘30分钟收益、尾盘60分钟收益、量价占比、HHI集中度、最大15分钟波动和Amihud非流动性。聚合状态为 `data/tabular/intraday_15m/state.aggregates.json`，源文件变化时自动按哈希重建。

## 完成审计

全部下载和最终聚合完成后运行：

```powershell
.venv\Scripts\python.exe scripts\audit_tabular_intraday_dataset.py
```

审计器会重新校验全部分片哈希、训练区边界、股票全集、每交易日16根bar、可用时间、聚合来源哈希和全部特征值，并从原始bar重新计算聚合结果逐文件比对。结果写入 `data/tabular/intraday_15m/audit.report.json`；任何错误都会返回非零退出码。

## 已完成结果（2026-08-21）

- 股票：343只。
- 年度分片：686个；608个有数据，78个为空，失败0个。
- 原始15分钟bar：2,307,216行。
- 完整交易日日级聚合：144,201行。
- 审计结论：通过；错误0个；`official_test_queried=false`。
- 审计证据：`data/tabular/intraday_15m/audit.report.json`。

# 个股排序训练数据队列

这条队列只处理 2023-01-01 至 2024-12-31 训练数据，不查询 2025 官方测试、隔离期或 2026 前向验证数据。

2026 严格结果留出的机器合同为 `config/strict_oos_2026_v1.yaml`。本队列在系统冻结前既不下载 2026 特征，也不构造或读取 2026 收益标签。

## 执行顺序

1. 等待全量公告元数据完成。
2. 冻结实际拥有训练期公告的股票池。
3. 用冻结的高精度标题规则识别核心事件；未命中规则的公告拒判，不调用未放行模型。
4. 对自动接纳的核心事件抓取完整分页正文并追加式归档。
5. 按“半年 × 事件类别”建立确定性 PDF 审计样本，默认最多归档 2,000 份附件。PDF 用于来源、解析和证据一致性审计，不把数十万份低价值附件无上限写入磁盘。
6. 下载股票与沪深 300 从 2022-09-01 到 2024-12-31 的日线。
7. 只构造训练期 1/3/5 个交易日超额收益标签。
8. 下载训练期 `daily_basic` 估值、换手与流动性数据。
9. 下载公司/行业历史与按实际披露日可用的三张财务报表原始数据。

## 状态与日志

- 总队列状态：`data/backfills/event_ranking_v1/data_queue.state.json`
- 公告元数据状态：`data/backfills/event_ranking_v1/full_announcements_train_2023_2024.json`
- 训练股票池：`data/backfills/event_ranking_v1/train_universe.json`
- PDF 抽样计划：`data/backfills/event_ranking_v1/pdf_audit_plan.json`
- 日线状态：`data/backfills/event_ranking_v1/market_prices.state.json`
- 队列标准输出：`logs/event_ranking_data_queue.stdout.log`
- 队列错误输出：`logs/event_ranking_data_queue.stderr.log`

每个下载阶段逐项原子更新状态；重启时会核对训练区间、股票代码哈希和数据合同，选择条件不同会拒绝复用旧状态。

## 恢复

先在当前 PowerShell 会话中设置已经轮换的 Tushare token，再启动队列：

```powershell
$env:TUSHARE_TOKEN = '<rotated-token>'
.venv\Scripts\python.exe scripts\run_event_ranking_data_queue.py
```

token 只应通过环境变量传入，不写进配置、状态或日志。后台任务是否存活以进程和状态文件共同判断；PID 可能因 Windows Python 启动器产生父子进程而变化。

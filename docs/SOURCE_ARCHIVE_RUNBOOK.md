# 训练集公告来源归档运行手册

最后更新：2026-08-21

## 当前任务

- 数据角色：仅 `train`。
- 筛选条件：高置信分类且 `relevance=core_event`。
- 训练期符合条件的核心事件总数：32,668。
- 正文：逐页完整抓取；任一页缺失则拒绝归档。
- PDF：正文全量回填阶段不统一下载，避免不可控磁盘占用；高价值分层样本和后续提取审计按内容寻址归档PDF。
- 市场目标表：不查询。
- 官方测试标签：不读取，读取次数保持0。

2026-08-21 11:47（Asia/Shanghai）首次启动隐藏后台进程 `20088`。每分钟60页在一篇44页长文后触发源站HTTP 567，该进程已人工停止。修复失败重试和熔断后，于11:52以每分钟30页重启为进程 `33332`。该进程在核心归档恢复稳定后，于12:05主动暂停，以单进程优先补齐结构化提取核验集缺失的三个半年度。时间覆盖补采完成后，正文API已经恢复，全量任务于12:25恢复为进程 `11416`。进程号在重启后可能变化，可信进度以状态文件为准。

## 审计文件

- 状态：`data/backfills/announcement_source_archive_train_core.json`
- 首次标准/错误输出：`data/backfills/logs/announcement_source_archive_train_core.stdout.log`、`data/backfills/logs/announcement_source_archive_train_core.stderr.log`
- 当前重试任务标准/错误输出：`data/backfills/logs/announcement_source_archive_train_core_retry.stdout.log`、`data/backfills/logs/announcement_source_archive_train_core_retry.stderr.log`
- 完成报告：`data/validation/announcement_source_archive_train_core_final.json`
- 时间分层优先计划：`data/backfills/structured_extraction_audit_priority.json`
- 时间分层优先状态：`data/backfills/structured_extraction_audit_priority.state.json`
- 时间分层优先日志：`data/backfills/logs/structured_extraction_audit_priority.stdout.log`、`data/backfills/logs/structured_extraction_audit_priority.stderr.log`
- 时间分层优先报告：`data/validation/structured_extraction_audit_priority.report.json`
- PDF回退状态：`data/backfills/structured_extraction_audit_priority_pdf.state.json`
- PDF回退日志：`data/backfills/logs/structured_extraction_audit_priority_pdf.stdout.log`、`data/backfills/logs/structured_extraction_audit_priority_pdf.stderr.log`
- PDF回退报告：`data/validation/structured_extraction_audit_priority_pdf.report.json`

状态文件每处理一条公告就原子更新。程序中断后，使用相同筛选条件和相同状态文件重启，会从最后一个数据库ID继续；已有相同归档不会重新抓取。

## 恢复命令

```powershell
.venv\Scripts\python.exe scripts\backfill_announcement_source_archives.py `
  --only-accepted `
  --relevance core_event `
  --rate-limit 30 `
  --pdf-mode none `
  --consecutive-failure-limit 3 `
  --state data\backfills\announcement_source_archive_train_core.json `
  --report data\validation\announcement_source_archive_train_core_final.json
```

不要用同一状态文件更换筛选条件。程序会检查状态文件中的 selection，条件不同会拒绝运行，防止跳过尚未归档的公告。

## 已验证行为

- 单页与17页真实公告归档成功，PDF文件哈希复核一致。
- 相同命令再次运行直接跳过，不产生重复快照。
- 归档内容变化时追加新快照，并通过 `supersedes_id` 连接旧版本。
- SQLAlchemy更新和删除旧快照均被拒绝。
- 候选 `eiTime` 永久标记为非权威时间，不能作为交易所正式发布时间。
- 首批70条普通训练归档和55条核心队列新增归档均为0失败。
- 每分钟60页被真实源站限流，已永久回退到每分钟30页。页级请求最多重试4次，采用10/20/30秒退避。
- 状态文件中的失败ID会在下一次运行中优先重试；成功后从失败清单移除，不会因 `last_announcement_db_id` 前移而永久漏抓。
- 连续3条公告最终失败时自动熔断停止，防止源站异常期间快速跳过大量记录。
- 结构化核验集补采期间，正文API连续3条HTTP 567后已熔断。静态PDF路径真实验证成功：40页、35,253字符，文件和文本哈希均已归档。PDF文本必须通过公告标题和逐页非空校验；不通过的记录留在失败清单单独复核。

## 完成后的门禁

1. 核对核心事件选择数、成功数、跳过数与失败数。
2. 对所有正文重新计算内容哈希和逐页清单一致性。
3. 按年份、事件类别和正文长度输出覆盖率。
4. 对失败记录单独重试，不以空文本代填。
5. 覆盖率和完整性通过后，才建立文本结构化提取审计集。

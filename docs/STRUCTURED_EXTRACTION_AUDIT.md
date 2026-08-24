# 公告结构化事实人工核验合同

最后更新：2026-08-21

## 目的与边界

本阶段只验证“大模型能否从公告原文中准确提取客观事实”，不让大模型直接生成历史涨跌标签，也不训练收益预测模型。核验数据只来自 `train` 分区的不可变正文归档；脚本不导入、不查询市场目标表。

输出 Schema 为 `announcement-structured-evidence-v1`，配置文件是 `config/structured_extraction_schema.yaml`，代码校验位于 `src/prediction/structured_extraction.py`。

## 证据规则

- 每个已支持事实必须包含原始值、原文字符起止位置和逐字一致的引文。
- 字符位置、引文、页码、逐页哈希和整篇正文 SHA-256 必须能够互相重建。
- 数字规范化后仍保留公告中的原始写法，不用估算值代替披露值。
- 公告没有披露或表述含糊时，分别标记 `absent` 或 `ambiguous`，并填写拒判原因。
- Schema 不允许方向、未来收益、真实涨跌或目标价字段。
- 当前 66 个公告子类别均有字段路由；例行文档只使用公共字段，不臆造事件专属事实。

## 300 份核验集门禁

正式核验集必须同时满足：

1. 仅使用训练期、核心事件、分类已接受、正文完整的公告；
2. 总数至少 300；
3. 2023-H1、2023-H2、2024-H1、2024-H2 每个半年度至少 50 份；
4. 按“半年度 × 子类别”轮转抽样，并在单元内使用固定哈希顺序；
5. JSONL 和 Schema 都记录 SHA-256，核验开始后不静默替换。

生成命令：

```powershell
.venv\Scripts\python.exe scripts\build_structured_extraction_audit_set.py
```

当任何半年度覆盖不足时，生成器会拒绝产出。2026-08-21 首次达到 300 份时全部来自 2023-H1，因此已降级保存到 `data/human_validation/provisional/`，不得用作阶段放行证据。

正式来源集已生成并通过独立校验：300 份，2023-H1 为 150 份，其余三个半年度各 50 份；其中正文API归档150份、静态PDF归档150份。JSONL SHA-256 为 `6767fd822257a7329a36ad9305c040f22d053db788e8a2096e529e95605f8d08`，300份当前均为 `pending`，不能宣称已经完成人工事实标注。

独立校验命令：

```powershell
.venv\Scripts\python.exe scripts\validate_structured_extraction_audit_set.py
```

## 当前时间覆盖补采

优先补采计划位于 `data/backfills/structured_extraction_audit_priority.json`：在另外三个半年度各补 50 份，共 150 份。计划文件明确标记 `dataset_role=train` 和 `market_targets_queried=false`，公告清单有独立 SHA-256。

正文 API 连续返回 HTTP 567 并触发安全熔断后，已验证静态 PDF 回退。首份真实样本共 40 页，使用 `pypdf-6.16.1` 提取 35,253 个字符，PDF、逐页文本和整篇文本均保存 SHA-256。PDF 回退必须满足标题在前三页出现、每页可提取文本、文件头为 `%PDF`；否则保留为失败项，不用空文本或弱校验结果代替。

时间覆盖补采已完成。150份首选清单中有7份被严格校验拒绝，另选7份替代公告后全部成功；失败样本保留用于来源审计。下一步冻结提取提示词和模型版本，填写300份事实标注，并计算数字/单位精确率、关键字段精确率、证据正确率和拒判率。

# 大模型公告事实提取协议

最后更新：2026-08-22

## 当前模块与边界

当前只开发四模型体系中的“大模型模块”。它只从单份公告提取带逐字证据的客观事实，不输出涨跌、利多程度或目标价。文本预测模型、表格模型和时间序列模型仍是独立的后续模块。

每次 API 调用都是新的无状态请求，仅包含固定提示词，以及当前公告的标题、当前分类、请求字段和完整原文。请求明确排除：

- 其他公告和模型历史回答；
- 人工金标准、对话历史和 Memory；
- 股价、市场目标、未来收益和涨跌标签；
- 2024 提取留出、2025 官方测试和 2026 前向验证数据。

候选记录必须保持 `human_review_included_in_prompt=false`、`conversation_history_included=false`、`memory_included=false` 和 `market_targets_queried=false`。

## 大模型内部时间分区

- `prompt_development`：2023-H1 的 100 份，只用于开发提示词和验证器；
- `model_selection_validation`：其余 2023 年的 100 份，用于已冻结候选的模型选择；
- `extraction_holdout`：2024 年的 100 份，候选通过模型选择并最终冻结前禁止读取。

运行器默认只处理开发样本。2024 留出由代码硬拒绝。模型选择样本必须先建立人工金标准锁；锁内固定具体 `audit_item_id`，运行器只允许请求这些 ID，防止误读已用于协议澄清的样本。

## 当前冻结候选

- 模型：`deepseek-v4-flash`
- 模式：`thinking=disabled`
- 温度：0
- 最大输出：8,192 tokens
- 提示词：`structured-extraction-prompt-v18`
- 提示词 SHA-256：`b2f47d6fcafd3e004b92e07cd15626909125636cf8301ef50e9bde265e670805`
- Schema：`announcement-structured-evidence-v2`
- 验证器：`evidence-validator-v10`
- 正式核验集 SHA-256：`1895cf8b4ba9462eac638eaacc8eed3331a7ec1ae748084831f751aba5e4ec9a`

预请求冻结文件为 `config/llm_extraction_v18_frozen.yaml`。第九批人工金标准在任何 v18 盲测请求之前完成并锁定于 `data/human_validation/model_selection_gold_batch1.lock.json`；正文为 `data/human_validation/model_selection_gold_batch9_draft.yaml`。上一批 v17 锁已逐字节归档为 `model_selection_gold_v17_batch8.lock.json`。本批 73 个 supported 金标字段在请求前全部通过冻结验证器和分类语义检查。

## v18 正式模型选择结果

10 份公告全部请求成功，候选全部通过文档级结构验证。105 个字段的独立评估结果：

- 状态精确率：84.76%；
- supported/abstain 二分类准确率：85.71%；
- 逐字严格字段准确率：54.29%；
- 锁定后人工语义字段准确率：81.90%；
- supported 语义精确率/召回率：84.21% / 87.67%；
- abstain 精确率/召回率：79.31% / 71.88%；
- 数值、日期、布尔精确率：87.50%。

结论仍为 **FAIL**，四项放行门槛均未通过。v18 已修复万元换算、业绩快报表格证据链、会计期间、合同履约风险复用等开发问题；新盲测中业绩快报数值全部正确，但模型仍会把不同股东、不同表格行或不同计量口径的数值任选或相加，也会把通用机构类别误当唯一对手方。重复表格日期和分段实体仍会被验证器误拒判，风险与控制字段有时只返回章节标题而非实质内容。盲测后没有修改候选实现。

完整审计见 `data/validation/llm_extraction_v18_model_selection_review_10.json`。v18 不得进入 2024 提取留出，也不得作为下游预测特征。v17 及更早历史结果仍保留于对应审计文件。

## 下一轮规则

v18 的第九批模型选择样本已消费，不能再用它报告新版本准确率。后续版本可把本轮错误作为验证反馈，但必须：

1. 只在开发数据上实现和自检新规则；
2. 冻结新的提示词、验证器、代码和文件哈希；
3. 从尚未读取正文的 2023 模型选择样本建立新的独立金标准；
4. 达到门槛后才允许一次读取 2024 提取留出；
5. 始终不读取 2025 官方测试与 2026 前向验证数据。

## 运行命令

开发集只读预检：

```powershell
.venv\Scripts\python.exe scripts\run_llm_extraction_candidates.py --role prompt_development --limit 1 --dry-run
```

模型选择只读预检：

```powershell
.venv\Scripts\python.exe scripts\run_llm_extraction_candidates.py --role model_selection_validation --dry-run
```

API 密钥只存于本地 `.env`，不得进入代码、日志、状态或核验文件。

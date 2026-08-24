# 2026-08-20 时间外盲审批次

本批次在读取 2026-08-16 之后的公告前，已冻结分类规则、分类体系和分类入口；随后抓取
2026-08-17 至 2026-08-20 的 467 条真实公告并锁定预测。锁定文件的 SHA-256 记录在
`manifest.json`，后续补正文或人工标注都不能改写预测。

## 当前状态

- 预测总数：467
- 程序接收：434
- 程序拒判：33
- 冻结规则覆盖率：92.93%
- 真人准确率：尚未产生；AI 初审不能替代真人金标准

## 人工盲审文件

1. `human_review_accuracy_sample.jsonl`：从 434 条已接收预测中按固定 SHA-256 顺序抽取
   120 条。文件不含程序预测，用于估计选择性准确率。
2. `human_review_coverage_gaps.jsonl`：全部 33 条拒判，文件不含 AI 候选，用于补分类体系和
   规则覆盖；不能用于估计已接收预测的准确率。
3. `human_review_blind.jsonl`：467 条全量盲审文件。首轮样本通过后，如需精确全量准确率，
   再审核此文件。
4. `ai_review_abstained.jsonl`：33 条拒判的 AI 候选意见，只能辅助复核，不能导入训练集或
   当作准确率真值。
5. `ai_review_accuracy_sample_filled.jsonl`：Codex 已逐条填写的 120 条准确率样本候选标签。
6. `ai_review_coverage_gaps_filled.jsonl`：Codex 已逐条填写的 33 条拒判候选标签。

两份 `ai_review_*_filled.jsonl` 均明确使用 `label_source=ai_candidate` 和
`label_status=candidate`。正式评估器会拒绝它们；真实审核人逐条确认后，应另存为
`*_verified.jsonl`，并填写真实审核者及 `human`/`verified`，不要直接改写候选文件。

人工填写时，每条记录至少需要：

- `major_category`、`sub_category`：真人独立判断的主事件；
- `label_status`: `verified`；
- `label_source`: `human`，如果先看过规则建议则填 `human_verified_rule`；
- `reviewer`：真实审核人；
- `labeled_at`：带时区的 ISO 时间，例如 `2026-08-20T18:00:00+08:00`；
- `evidence`：标题或正文中支持标签的短证据。

不要把 `ai_review_abstained.jsonl` 的标签批量复制成真人标签；无法从标题确认时应阅读当时已
公开的正文，仍无法确认则使用 `X/other` 并说明原因。

## 评估命令

```powershell
# 120 条准确率样本完成真人复核后
.\.venv\Scripts\python.exe scripts\temporal_validation.py evaluate-review-batch `
  --output-dir data\validation\oot_20260820 `
  --batch accuracy `
  --labels data\validation\oot_20260820\human_review_accuracy_sample_verified.jsonl

# 33 条拒判完成真人复核后
.\.venv\Scripts\python.exe scripts\temporal_validation.py evaluate-review-batch `
  --output-dir data\validation\oot_20260820 `
  --batch coverage_gaps `
  --labels data\validation\oot_20260820\human_review_coverage_gaps_verified.jsonl
```

评估器只接受 `real-label-v1` 的真人 verified 标签，并检查样本哈希、样本集合、重复 ID、
大类与小类映射。准确率报告同时输出 95% Wilson 区间；拒判批次只输出真实类别分布。

## 训练数据补充原则

- 财务公司风险持续评估报告：本批出现 10 条同型拒判；真人确认后可作为
  `X/routine_disclosure` 的真实回归样本，并补高精度标题规则。
- 非经营性资金占用及关联往来表/专项说明：应区分例行数据表和中介专项说明，分别积累
  `routine_disclosure` 与 `supporting_document` 的真实样本。
- 普通境外债券：不能标成 `convertible_bond`。应先新增 `bond_financing` 或
  `debt_financing` 类，再积累真实样本；在分类体系确定前暂用 `X/other`。
- 重大合同：需加入明确的正样本（标题明确“重大合同”或正文证明金额重要）和负样本
  （普通租赁合同、合同终止协议），防止只凭“签订+合同”判为重大。
- 法律意见书：主标签应描述底层事件，例如股东会、股权激励、定增；`legal_opinion` 更适合
  作为 `document_type`。需要用同一事件的主公告与法律意见书做成成对回归样本。
- 统计模型的每个小类仍至少需要 30 条训练、10 条验证、10 条时间外测试真人样本；不足的
  类只允许走经验证的高精度规则或拒判，不能用随机/合成数据凑数。

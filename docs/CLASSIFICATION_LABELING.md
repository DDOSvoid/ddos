# 真实公告分类标注与放行规范

## 当前原则

- `data/labeled/real_candidates_all.jsonl` 是真实公告候选集，不是训练真值。
- `suggested_*` 只帮助复核，不能复制后直接训练。
- 生产训练只接受 `real-label-v1`、人工确认且 `label_status=verified` 的记录。
- 合成数据、AI 候选、规则候选不能进入生产训练、校准或准确率统计。
- 训练、验证、测试严格按 `published_at` 前后切分，禁止随机拆分和未来数据泄漏。

## 一条训练数据的样子

```json
{
  "schema_version": "real-label-v1",
  "announcement_id": "AN202607161827024185",
  "published_at": "2026-07-17T00:00:00+08:00",
  "text": "公告标题和当时已经披露的正文",
  "major_category": "X",
  "sub_category": "corporate_admin",
  "label_source": "human",
  "label_status": "verified",
  "labeled_at": "2026-08-20T16:30:00+08:00",
  "reviewer": "reviewer_name",
  "dataset_role": "train",
  "evidence": "标题明确为完成工商变更登记并换发营业执照"
}
```

候选文件还包含 `suggested_rule_id`、`suggested_document_type`、
`suggested_relevance` 和旧模型候选，便于发现规则与模型冲突。这些辅助字段不属于真值。

## 人工复核规则

1. 只阅读公告在 `published_at` 时已经公开的标题和正文。
2. `major_category` 与 `sub_category` 填主事件；法律意见、审计报告、评估报告等同时用
   `document_type/relevance` 区分支持性文件，避免把文档形式当成业务事件。
3. 标题无法确认时必须保留 candidate，不猜测；正文仍无法确认时标成 `X/other` 并写明证据。
4. 多事件公告选择对上市公司影响最大的主事件，其他事件保留为 secondary tags。
5. 规则建议错误时填写正确标签，并在 `evidence` 记录错误原因；这些错误将进入规则反思清单。

## 时间切分

先固定日期边界，再开始标注，不能根据模型表现反复移动边界：

- 最早一段：`train`
- 中间一段：`validation`，只用于阈值和温度校准
- 最新一段：`test`，只允许最终评估一次

同一公告及其更正、法律意见、审计/评估附件应尽量放在同一侧，避免近重复泄漏。

当前候选集类别分布很不平衡：高频类有数百条，很多新增类只有 1–9 条。低频类暂时只走高精度规则，
不应强行纳入统计模型。计划进入模型的每个子类，最低应具备 30 条训练、10 条验证和 10 条测试样本；
不足时继续积累真实公告或合并到语义合理的上位类，不能合成补齐。

## 命令

```powershell
# 导出全部真实候选
python scripts/label_data.py --export --all --count 2000 `
  --output data/labeled/real_candidates_all.jsonl

# 校验一批人工标注
python scripts/label_data.py --import --input data/labeled/batch_001.jsonl

# 合并通过校验的人工批次
python scripts/label_data.py --merge --inputs data/labeled/batch_*.jsonl `
  --output data/labeled/real_verified.jsonl

# 训练、校准并执行时间外放行测试
python scripts/setup_model.py --data data/labeled/real_verified.jsonl --force
```

未生成真实数据训练清单、校准清单和通过门槛的评估清单时，模型会被视为未放行；程序只能采用
高精度规则，其他样本必须拒绝分类并进入复核。

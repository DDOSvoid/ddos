# 时间序列 LSTM v2 研究注册（2026-08-25）

状态：`HYPOTHESIS_ONLY_TRAIN_RESEARCH`。本文件只登记独立研究假设，不代表模型放行、冻结或官方测试授权。

目标口径已改为范围型目标：收益目标不再是项目级固定边界，而是在每个 outer fold 的训练段内按 10%/90% 分位数估计动态范围；覆盖率、区间宽度、Brier 改善和收益 MAE 改善使用可接受区间判断。区间端点仍必须由当时可用的训练数据因果拟合。

## 1. v1 失败证据

v1 合同、OOF 和检查点保持不可变：

- 合同：`config/timeseries_lstm_daily_v1.yaml`
- 合同 SHA-256：`ce3f577687d50162f15a1e9e7a43f8c6989a8793c195480882fc7471299cbc5f`
- 训练 OOF：`data/timeseries/lstm_daily_v1/oof/train_lstm_candidates.parquet`
- OOF SHA-256：`0041a6f614888f9999caaf02309b0167fee6463e7c7f96c4230641afe556cf78`
- 诊断：`data/timeseries/lstm_daily_v1/reports/v1_diagnostic_report.json`

只读诊断结论：12 个候选全部未通过绝对门槛；7 个候选通过简单序列增量，但没有候选可以进入表格增量比较。温度校准的外层 Brier 在 9/12 个候选变差、仅 3/12 个改善；所有 12 个候选的校准区间都比常数区间基线更宽，宽度增量范围为约 0.00674 至 0.02545。上述结果不能通过修改 v1 后重跑来解释，必须建立新合同。

## 2. 独立 v2 合同与边界

v2 注册文件：`config/timeseries_lstm_daily_v2_research.yaml`。

- 父合同仅用于继承日期、时点、身份、特征列、样本/fold 集合和评价门槛；父合同哈希必须匹配。
- 研究角色固定为 `train`；官方测试、quarantine、forward validation 和表格 OOF 均不可读取。
- 候选空间仍固定为 `lookback={20,60}`、`hidden={16,32}`、`horizon={1,3,5}`，共 12 个候选；仍为单层、单向、小容量 LSTM。
- 目标合同：`fold_local_empirical_return_range`，每个 outer fold 只用训练段拟合收益范围；默认范围为训练收益的 `[q10, q90]`，不是全项目固定收益数值。
- 可接受指标范围：区间覆盖率 `[0.80, 0.95]`、相对常数区间宽度 `[0.80, 1.00]`、相对常数 Brier 改善 `[0.00, 1.00]`、相对 fold-train 中位数收益 MAE 改善 `[0.00, 1.00]`；端点包含。
- 不增加层数、注意力、Transformer、双向结构、循环 dropout 或超参数搜索。
- v1 不被覆盖；v2 失败时保留 v1 和 v2 的独立证据，不得把 v2 结果回写为 v1 结果。

机器校验入口：`src/prediction/timeseries/lstm_research_contract.py`。其校验通过只说明“研究注册有效”，不说明预测有效。

## 3. 预登记假设与顺序

按一项一项变化执行；每项都要跑完整 12 候选和相同 expanding-window folds。

1. **target_scale**：只在 inner model-fit 段拟合预期收益及区间目标的均值/尺度，训练后反变换再评分，检验多头损失尺度失衡是否是原因。模型结构不变。
2. **shrinkable_interval**：只在 inner calibration 段拟合围绕预期收益的对称残差半径，允许区间收缩或扩张，避免 v1 只能扩张区间的限制。仅改变校准规则。
3. **brier_temperature**：在 inner calibration 段直接以 Brier 优化有界温度，并保留校准变差时的恒等回退，使校准目标与已登记门槛一致。仅改变校准目标。

每项假设都必须记录：训练/校准数据范围、候选与 fold、代码/合同/输入哈希、原始与校准指标、门槛结果和 `KEEP/REVERT`。不允许依据外层验证结果重新选择假设或范围。

## 4. 放行规则

v2 仍需同时通过 v1 继承的方向、收益、成本、区间、简单序列增量和表格增量门槛。任何一个期限失败都不能冻结该期限；没有候选通过时结论为 `KEEP_RESEARCH_ONLY_DO_NOT_FREEZE`。在训练门槛通过并完成冻结前，官方测试读取次数必须为 0。

## 5. 当前下一步

先运行机器合同和范围目标回归测试，随后实现并运行第 1 项 `target_scale` 的 train-only OOF。若第 1 项没有明确改善，再按注册顺序执行第 2 项；不得并行堆叠三项变化。

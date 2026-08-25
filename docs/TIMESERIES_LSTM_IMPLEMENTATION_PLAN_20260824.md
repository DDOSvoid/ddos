# 时间序列 LSTM 分支实施计划

最后更新：2026-08-24

状态：已确认，作为 `MODEL_DEVELOPMENT_ROADMAP.md` 阶段 5 的实施依据。

## 1. 目标与边界

时间序列分支只读取预测时点以前可用的股票、沪深300和成交量价序列，针对公告后 1、3、5 个交易日统一输出：

- 正超额收益概率 `bullish_probability`；
- 预期超额收益 `expected_excess_return`；
- 超额收益区间 `return_interval_lower` / `return_interval_upper`；
- 风险尺度 `risk_scale`；
- 序列覆盖与拒判标志；
- 模型版本、数据截止时间和全部证据哈希。

本分支不读取公告文本、结构化公告事实、表格模型输出、融合输出或 Memory。表格模型 OOF 只允许在候选预测完成后用于同样本增量评估，不得作为 LSTM 输入。

第一版只研究日频直接序列。15 分钟数据只能先形成公告日前完整交易日的日级序列或聚合基线；在取得权威公告发布时间并建立 timing-v2 以前，禁止使用公告当日盘中序列。

## 2. 永久数据规则

- 训练：2023-01-01 至 2024-12-31。
- 一次性官方测试：2025-01-01 至 2025-08-19。
- 永久隔离：2025-08-20 至 2025-12-31。
- 前向验证：2026-01-01 起，只验证，不训练、不调参。
- 禁止随机切分；训练期内部只允许 expanding-window。
- 任何标签必须在对应 fold 截止点前成熟。
- 特征和标签物理隔离。
- 缺失历史不能用未来数据填充。
- 预处理、早停和校准只能使用 outer-fold 训练区内部数据。
- 官方测试在全部候选冻结后最多读取一次，读取后不得调参。

## 3. 阶段 0：合同、时间与身份对齐

在训练 LSTM 前必须完成：

1. 新建 `causal-timeseries-lstm-daily-v1` 合同，保留旧 daily-v1/TCN 研究证据，不原地改写旧合同或报告。
2. 统一三个分支的逻辑预测时点为公告日期之后的下一交易日 08:30（Asia/Shanghai）。
3. 日期级保守行情截止继续固定为 `trade_date < published_date`；统一预测时点不授权读取公告日行情。
4. 使用跨分支统一的 `company_day_id`、`sample_id`、`horizon_sessions` 和 fold 名称。
5. 缺少序列的样本保留规范身份并输出 `sequence_available=false`，不得用默认高分填补。
6. 冻结统一输出 Schema 和风险定义。
7. 解决现存 split/config 哈希漂移；新产物必须绑定当前有效合同、源数据和代码哈希。

当前已确认的待修复事实：

- 路线图已经改为小容量单层 LSTM，但旧机器合同和校验器仍固定为 `small_tcn`。
- 现有时序 `prediction_as_of` 是公告日期当天 08:30，表格分支是下一交易日 08:30。
- 现有日频 manifest 的 split 哈希与当前 split 文件不一致；15 分钟聚合 manifest 的合同哈希与当前配置不一致。
- 表格和时序训练 OOF 按股票、公告日、期限可对齐 37,025 条，fold 一致率为 100%，但两路 `sample_id` 算法不同。

## 4. 阶段 1：数据补齐与直接序列产物

### 4.1 行情缓冲

- 补充至少从 2022-09-01 开始的日频缓冲行情。
- 重新审计价格来源、复权语义、停牌、缺失值和沪深300交易日历。
- 在补充前，60 日候选只能使用完整 60 日样本；禁止未来填充。

当前覆盖基线：69,904 个训练候选中，68,023 个满足最小 20 日历史，61,872 个具备完整 60 日序列；2023-H1 完整 60 日覆盖率为 64.29%。

### 4.2 序列格式

每个公司公告日只存一份直接序列，1/3/5 日标签单独保存，避免按期限复制相同输入。

第一版序列字段固定为：

- 股票单日收益；
- 沪深300单日收益；
- 单日超额收益；
- 股票日内振幅；
- 股票隔夜跳空；
- 对数成交量；
- 对数成交额；
- 股票可交易掩码；
- 成交量有效掩码；
- 成交额有效掩码。

产物建议：

- `train_sequences.npy`：`float32 [company_days, max_lookback, features]`；
- `train_masks.npy`：序列有效性掩码；
- `train_metadata.parquet`：规范身份、预测时点和最后可用 bar；
- `train_labels.parquet`：按 horizon 保存方向和收益标签；
- `manifest.json`：合同、数据、Schema、文件和代码哈希。

## 5. 阶段 2：公平基线

在相同样本和 folds 上固定比较：

1. fold-train 正例率/收益中位数常数基线；
2. 现有 30 项公告前聚合特征 Logistic/Ridge 基线；
3. 同一直接序列经过简单时间池化或展平后的 Logistic/Ridge 序列基线；
4. 小容量单层 LSTM。

聚合特征只能作为基线，不能替代直接序列实验。

## 6. 阶段 3：LSTM v1

第一版预登记为小容量单层 LSTM：

- lookback 候选只允许 20 和 60；
- hidden size 候选只允许 16 和 32；
- 单层 LSTM；
- dropout 放在 LSTM 输出之后；
- fold 内标准化；
- 固定 weight decay；
- 固定随机种子；
- 固定最大 epoch；
- 时间顺序 early stopping；
- 每个 1/3/5 日期限独立训练和放行。

每个期限包含三个输出任务：

- 方向分类头：输出正超额收益概率；
- 收益回归头：输出预期超额收益；
- 区间头：输出上下分位数或等价风险尺度。

第一版不引入双层 LSTM、attention、Transformer、动态门控或大规模超参数搜索。只有小模型证明稳定 OOF 增量后才允许研究更复杂结构。

## 7. 阶段 4：fold 内早停与校准

每个 outer fold 的训练段按时间继续划分为：

1. 模型拟合段；
2. early-stop 段；
3. 概率与区间校准段。

outer validation 只能评估，不能拟合标准化器、模型、阈值或校准器。概率校准与区间校准的配置和哈希必须写入 OOF 报告。

## 8. 阶段 5：OOF 评估与放行

每个期限必须满足通用门槛：

- OOF 利多信号至少 300；
- 利多命中率 Wilson 95% 下界严格高于 50%；
- 至少两个 fold 的利多命中率高于 50%；
- 最差 fold 不低于 48%；
- Brier Score 优于 fold-train 常数基线；
- 按 20 bps 往返成本计算的平均超额收益置信下界大于 0；
- 预期收益误差优于 fold-train 中位数基线；
- 风险区间覆盖和宽度优于预登记简单基线。

同时必须证明：

- LSTM 在相同样本和 folds 上稳定优于简单直接序列基线；
- LSTM 独立效果与表格模型可比较；
- 在表格/时序共同样本上，预登记的简单组合相对表格模型具有稳定增量。

增量评估不得读取表格模型训练内预测，只能读取表格 OOF。复杂模型不能因为 AUC 单项提高而放行。

## 9. 阶段 6：决策、冻结与官方测试

- 任一必要门槛失败：`KEEP_RESEARCH_ONLY_DO_NOT_FREEZE`。
- 通过训练期门槛后，冻结模型、预处理器、校准器、阈值、Schema、样本、环境、代码和全部文件哈希。
- 所有预登记候选在同一次官方测试读取中共同评估。
- 官方测试通过后才允许建立每日推理入口和融合层正式输入。

## 10. 实现目录

```text
config/
└── timeseries_lstm_daily_v1.yaml

src/prediction/timeseries/
├── lstm_contract.py
├── sequence_identity.py
├── sequence_artifacts.py
├── sequence_dataset.py
├── sequence_preprocessing.py
├── sequence_baselines.py
├── lstm_model.py
├── train_lstm.py
├── calibrate_lstm.py
├── evaluate_lstm.py
├── compare_tabular_oof.py
├── freeze_lstm.py
└── inference.py

data/timeseries/lstm_daily_v1/
├── sequences/
├── labels/
├── manifests/
├── oof/
├── calibration/
└── reports/

models/timeseries/lstm_daily_v1/
├── research/
└── frozen/

tests/
├── test_timeseries_lstm_contract.py
├── test_timeseries_sequence_identity.py
├── test_timeseries_sequence_artifacts.py
├── test_timeseries_sequence_dataset.py
├── test_timeseries_lstm_model.py
├── test_timeseries_lstm_walk_forward.py
├── test_timeseries_lstm_calibration.py
├── test_timeseries_lstm_evaluation.py
└── test_timeseries_tabular_comparison.py
```

## 11. 当前实施顺序

1. 合同、预测时点、样本身份和输出 Schema 对齐。
2. 补足 2022 年缓冲行情并重建可审计直接序列。
3. 跑常数、聚合和简单序列基线。
4. 跑小容量单层 LSTM expanding-window OOF。
5. 完成校准、成本、收益、风险和表格增量门槛。
6. 训练期通过后冻结；否则保留研究证据且官方测试继续封闭。

每完成一个步骤，都必须更新本文件和主路线图的状态、证据文件、版本和 KEEP/REVERT/FREEZE 结论。

## 12. 实施记录

### 2026-08-24：阶段 0 与阶段 1 完成

结论：`KEEP_RESEARCH_ONLY_DO_NOT_FREEZE`。合同、时间、身份和直接序列数据已经达到进入公平基线研究的条件；尚未运行 expanding-window OOF，不得冻结模型或读取官方测试。

已完成：

- 建立机器合同 `config/timeseries_lstm_daily_v1.yaml`，绑定当前 split、开发合同和父级日频合同哈希；旧 daily-v1/TCN 证据保持不变。
- 将逻辑预测时点统一为公告日期之后的下一交易日 08:30（Asia/Shanghai），同时继续禁止使用公告日行情。
- 统一 `company_day_id` 和 `sample_id` 算法；缺失序列保留身份并显式标记不可用。
- 从训练分区公告身份和缓存行情直接构造无标签序列，再独立重算训练标签；序列生成不查询目标表，标签生成不查询旧目标表。
- 回填 2022-09-01 至 2022-12-14 行情：19,784 条个股日线、69 条沪深300日线；53 个证券在该区间无记录，主要对应尚未上市证券。
- 重建后共有 17,211 个训练公司日，其中 16,882 个序列可用且全部具备完整 60 日窗口；329 个不可用身份保留拒判原因。
- 独立重算 50,802 条 1/3/5 日训练标签，与表格分支训练标签的身份、方向和超额收益逐条一致，最大收益差为 0。
- 训练工件审计通过，`official_test_read=false`、`quarantine_read=false`、`forward_validation_read=false`；39 项时序定向测试通过。

当前证据：

- `data/timeseries/lstm_daily_v1/manifests/train_manifest.json`
- `data/timeseries/lstm_daily_v1/manifests/train_labels_manifest.json`
- `data/timeseries/lstm_daily_v1/reports/train_artifact_audit.json`

下一步严格进入阶段 2：在相同样本与固定 folds 上实现并运行常数、聚合和简单直接序列 Logistic/Ridge 基线。基线证据固定后才允许启动小容量单层 LSTM OOF。

### 2026-08-24：阶段 2 公平直接序列基线完成

结论：六个预登记的“20/60 日 lookback × 1/3/5 日 horizon”组合全部 `FAIL`，仅保留为 LSTM 必须在相同样本和 folds 上超越的研究基准。状态继续为 `KEEP_RESEARCH_ONLY_DO_NOT_FREEZE`，官方测试读取次数保持 0。

固定实现：

- 每条直接序列仅计算 `last/mean/std/minimum/maximum/slope` 六类样本内时间统计，共 60 个输入；不读取其他分支预测。
- 方向模型固定为无权重 Logistic，`C=0.1`；收益模型固定为 Ridge，`alpha=1.0`；不进行外层 OOF 超参数搜索。
- 缺失值中位数填补和标准化均只在对应 outer-fold 训练段拟合。
- 20 日和 60 日候选在每个期限使用完全相同的样本身份与 folds。
- 区间基线使用 fold-train Ridge 残差的 10%/90% 分位数；常数基线使用 fold-train 正例率、收益中位数和收益分位数。

训练期 OOF 摘要：

| Lookback | Horizon | OOF | AUC | Brier / 常数 | 利多 Wilson 下界 | 收益 MAE / 中位数 | 扣费收益 CI 下界 | 结论 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20 | 1 | 12,563 | 0.5010 | 0.2568 / 0.2485 | 45.32% | 0.02136 / 0.02051 | -0.00248 | FAIL |
| 20 | 3 | 12,319 | 0.5101 | 0.2570 / 0.2478 | 44.54% | 0.03841 / 0.03683 | -0.00536 | FAIL |
| 20 | 5 | 12,143 | 0.5191 | 0.2632 / 0.2476 | 44.99% | 0.05036 / 0.04808 | -0.00447 | FAIL |
| 60 | 1 | 12,563 | 0.5157 | 0.2540 / 0.2485 | 46.36% | 0.02113 / 0.02051 | -0.00190 | FAIL |
| 60 | 3 | 12,319 | 0.5203 | 0.2691 / 0.2478 | 44.91% | 0.04035 / 0.03683 | -0.00490 | FAIL |
| 60 | 5 | 12,143 | 0.5437 | 0.2807 / 0.2476 | 46.69% | 0.05575 / 0.04808 | -0.00266 | FAIL |

虽然 60 日/5 日 AUC 达到 0.5437，但其 Brier、Wilson 下界、收益 MAE、扣费收益置信下界和区间诊断同时失败，不能因单一 AUC 指标放行。

证据：

- `data/timeseries/lstm_daily_v1/oof/train_simple_sequence_baselines.parquet`
- `data/timeseries/lstm_daily_v1/reports/train_simple_sequence_baselines.json`

下一步严格进入阶段 3：运行预登记的 20/60 日、hidden size 16/32、小容量单层 LSTM expanding-window OOF。每个候选必须同时接受绝对门槛和本节简单序列基线增量门槛；在此之前不得比较或读取官方测试。

### 2026-08-24：阶段 3 至阶段 5 的 LSTM v1 OOF 完成

结论：`KEEP_RESEARCH_ONLY_DO_NOT_FREEZE`。12个预登记候选、36个fold模型和148,100行OOF已完成；没有候选通过训练期绝对门槛，因此没有进入表格增量比较，官方测试读取保持0。

关键结果：

- 12个候选的AUC为0.4936至0.5350，但全部未通过完整绝对门槛。
- 所有候选的Wilson下界低于50%，收益MAE均差于fold-train中位数基线，绝对区间诊断全部失败。
- 7个候选稳定改善匹配的简单序列基线，但仍未稳定优于常数概率和收益中位数等绝对基线。
- 20日/hidden32/5日虽然简单序列增量通过且扣20bps收益CI下界为正，仍因Brier、Wilson、fold稳定性、收益MAE和区间门槛失败。
- 60日/hidden32/5日AUC最高为0.5350，但Brier 0.2482差于常数0.2476，不能按单一AUC放行。
- 36个检查点哈希全部匹配；OOF身份无重复、输出Schema完整，官方测试/隔离区/前瞻验证读取均为0。

完整报告：`docs/TIMESERIES_LSTM_V1_OOF_20260824.md`。

本计划的阶段3至阶段5已经执行，但放行条件未达成，因此阶段6冻结和官方测试保持关闭。后续研究必须新建版本化合同，不得原地调整 v1 后把同一OOF当作独立确认结果。

### 2026-08-25：v1 失败诊断与 v2 研究注册

v1 证据保持不可变。只读诊断报告 `data/timeseries/lstm_daily_v1/reports/v1_diagnostic_report.json` 显示：12 个候选全部绝对门槛失败；温度校准外层 Brier 在 9/12 个候选变差；12/12 个候选的区间宽度均高于常数基线。官方测试、quarantine、forward validation 和表格 OOF 均未读取。

因此不在 v1 上调参，建立独立合同 `config/timeseries_lstm_daily_v2_research.yaml`，并由 `src/prediction/timeseries/lstm_research_contract.py` 机器校验。v2 保持同一 train-only 数据、身份、样本、fold、12 个候选和小型单层 LSTM 结构，仅按预登记顺序研究：`target_scale`、`shrinkable_interval`、`brier_temperature`。详细规则与下一步见 `docs/TIMESERIES_LSTM_V2_RESEARCH_PLAN_20260825.md`。

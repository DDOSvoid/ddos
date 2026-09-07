# 三专家融合选股框架 v1

状态：`ACTIVE_RESEARCH`  
生效日期：2026-09-01

## 产品边界

第一版只对每日 08:30 前已有合格公告的 A 股公司进行公司日级排序。无公告股票不进入 v1 候选池；同一公司同一天的多份公告先聚合，再针对 1/3/5 个交易日期限各生成一次预测。

正式身份固定为：

```text
stock_code + prediction_as_of + horizon_sessions
```

历史公告级 `ImpactPrediction` 和规则 `composite_score` 保留作审计与文本基线，不再作为生产排序主输出。

## 专家职责

| 专家 | 主要输出 | 不再强制承担 |
|---|---|---|
| 文本 | 事件语义、重要性、意外度、证据质量、拒判、文本信号 | 独立同时通过方向/收益/风险门槛 |
| 表格 | 财务、估值、成长、行业相对位置、缺失质量、表格信号 | 独立完成最终股票排序 |
| 时序 | 趋势、波动、流动性、市场状态、交易风险、时序信号 | 独立成为完整选股系统 |

每个专家必须通过时点、来源、版本、覆盖和拒判硬门槛。预测价值通过融合层的 drop-one-expert OOF 消融判断；数据泄漏或来源不合格的专家不得以“可能有增量”为由进入融合。

## 融合与排序

融合层是唯一的生产预测和排序所有者。输入只允许使用组件 OOF，不读取组件训练内预测。第一轮顺序固定为：

1. 可用诊断头等权平均；
2. 非负静态 stacking；
3. 按进场日期分组的 LightGBM LambdaRank；
4. 静态模型证明稳定增量后，才讨论动态门控。

最终输出包括正超额收益概率、预期超额收益、风险尺度、风险成本调整排序分、每日名次、组件贡献和拒判原因。排序分第一版为：

```text
expected_excess_return - risk_aversion * risk_scale - round_trip_cost
```

主要评估指标改为每日 Rank IC、Precision@K、Top-K 超额收益、Top-K 相对其余候选的收益差、扣费收益、换手和覆盖率。方向命中率、Brier 和收益 MAE 保留为辅助指标。

机器合同：

- `config/causal_expert_ranking_development_v2.yaml`
- `config/expert_fusion_v1.yaml`
- `src/prediction/fusion/`

## 数据下载边界

训练股票池必须包含 Tushare `L/D/P` 三种状态，不能只下载当前上市公司。训练期为 2023–2024，行情前置窗口从 2022-09-01 开始。正式下载顺序为股票池、公告元数据、公告正文/PDF、股票与沪深300日线、训练标签、估值流动性、时点公司/行业/财务。

2026 年结果按 `config/strict_oos_2026_v1.yaml` 封存。三个专家和融合层的开发、消融与排名评估只能读取 2023–2024 的显式 `train` 标签。系统与融合版本完成内容哈希冻结后，才允许按历史时点重放 2026 输入并追加预测；全部预测锁定前不得读取 2026 实现收益、方向标签或任何排名指标。该口径是新专家融合架构的严格结果留出；仓库历史上已有不含股票收益值的 2026 公告分类审计和 forward-validation 行数报告，因此不得宣称“整个仓库从未触碰过任何 2026 数据”。

机器合同为 `config/event_ranking_data_v1.yaml`。2025 官方测试标签、隔离区和 2026 前向标签继续保持未读。

无 Tushare 凭证时，只允许用 `config/tracked_companies.yaml` 的 11 只股票验证公开东方财富公告链路；该 smoke 数据不能冒充正式研究股票池。

```powershell
# 凭证只写本地 .env，不提交、不粘贴到日志
copy .env.example .env

# 正式股票池（需要本地 TUSHARE_TOKEN）
.venv\Scripts\python scripts/seed_database.py

# 无凭证公开链路 smoke
.venv\Scripts\python scripts/seed_tracked_universe.py

# 下载状态预检
.venv\Scripts\python scripts/check_event_ranking_data.py --write-manifest
```

## 当前实施状态

- [x] 新专家/融合/排序机器合同
- [x] 公司日专家信号和融合预测 append-only 表
- [x] 联合 OOF 外连接、缺失掩码和未来数据拦截
- [x] 风险成本调整排序和横截面 Top-K 评估
- [x] 等权融合基线入口
- [x] 无幸存者偏差的 L/D/P 股票池下载入口
- [ ] 正式股票池下载：等待本地 Tushare 凭证
- [ ] 三路组件 OOF 适配与生成
- [ ] 静态 stacking / LambdaRank 训练
- [ ] 官方测试与每日生产接入

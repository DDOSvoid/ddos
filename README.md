# DDOS — 公告财报事件驱动分析系统

Disclosure-Driven Opportunity Scanner — A 股上市公司公告自动化分析系统。

## 系统架构

```
[Tushare API] + [东方财富 API]
        │
        ▼
    Fetcher ─── 数据获取（公告元数据 + 原文）
        │
        ▼
  Preprocessor ─── 文本清洗（HTML → 纯文本规范化）
        │
        ▼
  Classifier ─── BERT 分类（A-G 大类 + 30+ 子类）
        │
        ▼
  Extractor ─── LLM 字段提取（GPT-3.5/Qwen 批量结构化）
        │
        ▼
  Scorer ─── 规则评分（方向 × 强度 × 意外度 × 可信度）
        │
        ▼
  Reporter ─── 每日 Markdown 报告
```

## 快速开始

### 1. 环境准备

```powershell
# 克隆/进入项目目录
cd ddos

# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

```powershell
# 复制环境变量模板
copy .env.example .env

# 编辑 .env 文件，填入你的 API key:
#   TUSHARE_TOKEN=你的tushare_token
#   OPENAI_API_KEY=你的openai_api_key  （可选，MVP 可暂不填）
```

### 3. 初始化数据库 + 股票列表

```powershell
# 从 Tushare 拉取全部 A 股列表并写入数据库
python scripts/seed_database.py
```

### 4. 运行每日管线

```powershell
# 执行完整管线（拉取最近 3 天公告 → 清洗 → 分类 → 提取 → 评分 → 报告）
python scripts/run_pipeline.py --date yesterday

# 仅拉取数据（测试用）
python scripts/run_pipeline.py --date 2024-06-01 --stages fetch,preprocess
```

### 5. 查看报告

报告保存在 `data/reports/report_YYYY-MM-DD.md`。

## 项目结构

```
ddos/
├── config/                 # 配置文件
│   ├── config.yaml         # 主配置
│   ├── event_types.yaml    # A-G 事件分类体系（30+ 子类）
│   └── tracked_companies.yaml  # 跟踪公司列表
├── src/
│   ├── config.py           # Pydantic 配置加载
│   ├── database/           # SQLAlchemy ORM + Repository
│   ├── pipeline/           # 数据处理管线
│   │   ├── fetcher.py      # Tushare + 东方财富数据获取
│   │   ├── preprocessor.py # 文本清洗
│   │   ├── classifier.py   # BERT 分类步骤
│   │   ├── extractor.py    # LLM 字段提取
│   │   ├── scorer.py       # 规则评分引擎
│   │   ├── reporter.py     # 每日报告生成
│   │   └── orchestrator.py # 管线编排器
│   ├── ml/                 # ML 模块
│   │   ├── classifier_wrapper.py  # BERT 推理封装
│   │   └── llm_client.py   # LLM 客户端（OpenAI/兼容 API）
│   ├── training/           # 模型训练
│   │   ├── dataset.py      # PyTorch Dataset
│   │   ├── train_classifier.py    # 微调脚本
│   │   └── evaluate.py     # 评估脚本
│   └── utils/              # 工具函数
│       ├── text_utils.py   # 中文文本清理
│       └── date_utils.py   # 交易日历
├── scripts/                # 可执行脚本
│   ├── run_pipeline.py     # 每日运行入口
│   ├── seed_database.py    # 初始化数据库
│   └── label_data.py       # 数据标注助手
├── tests/                  # 测试（61 个测试）
├── notebooks/              # Jupyter 探索
├── data/                   # 运行时数据（gitignore）
└── models/                 # 训练好的模型（gitignore）
```

## 评分公式

```
composite_score = direction × magnitude × surprise × credibility

direction  ∈ [-1.0, +1.0]   ← 类别基线 + 字段调整
magnitude  ∈ [0.0, 1.0]     ← 金额 / 公司规模 归一化
surprise   ∈ [0.0, 1.0]     ← 有预期→delta，无→默认基线
credibility ∈ [0.0, 1.0]    ← 来源 × 数据完整度
```

## 事件分类

| 大类 | 说明 | 子类数 |
|------|------|--------|
| A | 财报主事件（年报/半年报/季报） | 4 |
| B | 财报前置事件（预告/快报/修正） | 3 |
| C | 股权变动（增持/减持/回购/解禁/激励） | 5 |
| D | 资本运作（定增/并购/资产出售/可转债） | 4 |
| E | 经营业务（合同/中标/产品获批/投产/战略合作） | 5 |
| F | 风险事件（诉讼/处罚/立案/债务逾期/退市风险） | 5 |
| G | 治理与交易状态（高管变动/实控人变更/停复牌/异常波动） | 4 |

## 下一步（你需要做的）

1. **注册 Tushare**: https://tushare.pro → 获取 token → 填入 `.env`
2. **测试 Tushare 连通性**: 运行 `notebooks/01_data_exploration.ipynb`
3. **初始化数据库**: `python scripts/seed_database.py`
4. **获取 API Key**（可选）: OpenAI API key 用于 LLM 提取和分析
5. **标注数据**: 用 `scripts/label_data.py` 导出公告 → 手动标注 → 微调分类模型

## 运行测试

```powershell
python -m pytest tests/ -v       # 全部测试
python -m pytest tests/ -m "not e2e"  # 跳过需要真实 API 的测试
```

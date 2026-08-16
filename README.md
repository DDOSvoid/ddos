# DDOS — 公告财报事件驱动分析系统

Disclosure-Driven Opportunity Scanner — A 股上市公司公告自动化分析系统。

自动抓取 A 股公告 → 清洗 → 分类 → LLM 提取结构化字段 → 规则评分 → 生成每日 Markdown 报告。

## 系统架构

```
[Tushare API] + [东方财富 API]
        │
        ▼
    Fetcher ─── 数据获取（公告元数据 + 正文富化 + PDF 链接）
        │
        ▼
  Preprocessor ─── 文本清洗（HTML → 纯文本规范化）
        │
        ▼
  Classifier ─── BERT 微调分类（A-G 大类 + 30 子类，置信度校准）
        │
        ▼
  Extractor ─── DeepSeek 字段提取（按分类路由提示词，批量结构化）
        │
        ▼
  Scorer ─── 规则评分（方向 × 强度 × 意外度 × 可信度）
        │
        ▼
  Reporter ─── 每日 Markdown 报告（高影响事件可选 DeepSeek 深度分析）
```

## 快速开始

### 1. 环境准备

```powershell
# 克隆项目
git clone git@github.com:DDOSvoid/ddos.git
cd ddos

# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置密钥

```powershell
# 复制环境变量模板
copy .env.example .env
```

编辑 `.env`，填入两个 key：

| 变量 | 说明 |
|------|------|
| `TUSHARE_TOKEN` | Tushare Pro token（https://tushare.pro），初始化股票列表用 |
| `OPENAI_API_KEY` | **DeepSeek** API key（OpenAI 兼容），字段提取 + 深度分析用 |
| `OPENAI_BASE_URL` | `https://api.deepseek.com`（模板已填） |

> 模型名（`deepseek-v4-flash`）在 `config/config.yaml` 的 `models.extraction` / `models.analysis` 配置，**不在** `.env` 中改。`OPENAI_API_KEY` 留空也能跑通除 extract/report 之外的阶段（调用时给出清晰报错）。

### 3. 构建分类模型（首次/新机器必做）

```powershell
python scripts/setup_model.py
```

**模型权重（`models/bert-classifier`，约 391M）不入 git**，由该脚本一键确定性复现：生成种子数据 → 微调 BERT → 温度校准。详见下文[模型构建与复现](#模型构建与复现)。

### 4. 初始化数据库 + 股票列表

```powershell
# 从 Tushare 拉取全部 A 股列表并写入数据库
python scripts/seed_database.py
```

### 5. 运行每日管线

```powershell
# 完整管线（默认拉取最近 3 天公告 → 清洗 → 分类 → 提取 → 评分 → 报告）
python scripts/run_pipeline.py

# 指定日期
python scripts/run_pipeline.py --date 2026-08-08

# 只跑部分阶段（逗号分隔）
python scripts/run_pipeline.py --date 2026-08-08 --stages fetch,preprocess
python scripts/run_pipeline.py --date 2026-08-08 --stages report
```

`--date` 支持 `today` / `yesterday` / `YYYYMMDD` / `YYYY-MM-DD`；`--stages` 可组合 `fetch,preprocess,classify,extract,score,report` 任意子集。

### 6. 查看报告

报告保存在 `data/reports/report_YYYY-MM-DD.md`。

## 模型构建与复现

**为什么**：模型权重 391M 不适合进 git（GitHub 单文件上限 100MB、clone 膨胀、曾尝试 Git LFS 但受配额与上传带宽限制），因此 `models/` 整体 gitignore。分类器完全由代码确定性复现——种子标注数据从 `config/event_types.yaml` 模板合成（固定 seed），微调使用固定随机种子。

`scripts/setup_model.py` 一键完成三步：

```powershell
# 1. 生成种子标注数据（data/labeled/seed.jsonl，确定性）
python scripts/generate_seed_data.py

# 2. 微调 BERT（models/bert-classifier）
python -m src.training.train_classifier --data data/labeled/seed.jsonl --output models/bert-classifier

# 3. 温度校准（锐化 softmax 置信度 → temperature.json）
python scripts/calibrate_temperature.py
```

> **第 3 步不能省**：种子模型 30 类间 softmax 偏平，top-1 置信度仅 0.1~0.25，不校准会被 `extraction_min_confidence=0.6`（`config.yaml`）全部挡掉，提取阶段空转。校准产物 `temperature.json` 与权重一起在 `models/` 下。

辅助诊断脚本：

```powershell
# 检查模型是否就绪（0=就绪，1=缺失）
python scripts/setup_model.py --check

# 强制重建
python scripts/setup_model.py --force

# 置信度诊断：对若干无歧义样例输出 top-3 概率
python scripts/check_confidence.py
```

### 用真实公告增强训练（可选）

种子模型用合成数据微调，**标注少量真实公告并入训练集**可显著提升真实场景分类精度：

```powershell
# 1. 导出真实公告（默认全部状态，优先有正文的；自动跳过已标注过的）
python scripts/label_data.py --export --output data/labeled/to_label.jsonl --count 200

# 2. 手动编辑 to_label.jsonl，为每行填 sub_category / major_category
#    （类别选项见 config/event_types.yaml）

# 3. 与种子数据合并（按 announcement_id/text 去重，校验类别合法性）
python scripts/label_data.py --merge \
    --inputs data/labeled/seed.jsonl data/labeled/to_label.jsonl \
    --output data/labeled/combined.jsonl

# 4. 用合并数据重训 + 校准
python -m src.training.train_classifier --data data/labeled/combined.jsonl --output models/bert-classifier
python scripts/calibrate_temperature.py --data data/labeled/combined.jsonl --model models/bert-classifier
```

> `--export` 默认从**全部状态**导出（管线跑完后公告状态已推进到 `reported`，旧的只导 `preprocessed` 会导出 0 条）；`--status preprocessed` 可指定只导某状态。合并阶段会拦截不在 `event_types.yaml` 里的类别值，避免错别字悄悄给模型加一个新类。

## 公告正文富化

东方财富**列表接口**只返回元数据（标题/日期），正文为空。管线默认调用**内容接口**按 `art_code` 拉取正文与 PDF 链接，落库 `full_text` / `pdf_url`。

```yaml
# config/config.yaml
pipeline:
  fetch_full_text: true   # false 则只用列表摘要（更快但提取效果差）
  fetch_backend: "http"   # 公告抓取后端: http=requests / cdp=Playwright 驱动系统 Chrome
```

`fetch_backend` 切换抓取方式：`http`（默认，轻量快）直接用 `requests` 调东方财富 API；`cdp` 用 Playwright 驱动系统已装 Chrome，在浏览器内请求（真实 UA/会话，更稳但更慢）。CDP 后端需先 `pip install playwright`（用系统 Chrome，无需 `playwright install` 下载浏览器）。

正文缺失时（历史上曾发生）LLM 提取全空 → `data_completeness=0` → `credibility=0` → 乘法评分一票否决，所有公告恒为中性。务必保持开启。

## 配置说明

| 文件 | 管什么 | 示例 |
|------|--------|------|
| `.env` | 密钥（gitignore） | Tushare token、DeepSeek key |
| `config/config.yaml` | 业务参数 + 模型 | `pipeline.*`、`models.extraction.model`、`scoring.*` |
| `config/event_types.yaml` | A-G 分类体系 + 每类提取字段 | 30 子类定义 |
| `config/tracked_companies.yaml` | 试点跟踪公司名单 | — |

## 评分公式

```
composite_score = direction × magnitude × surprise × credibility

direction  ∈ [-1.0, +1.0]   ← 类别基线 + 字段调整
magnitude  ∈ [0.0, 1.0]     ← 金额 / 公司规模 归一化
surprise   ∈ [0.0, 1.0]     ← 有预期→delta，无→默认基线
credibility ∈ [0.0, 1.0]    ← 来源可信度 × 数据完整度（填了/期望字段）
```

乘法公式中 `credibility` 为 0 时一票否决；`report_high_impact_threshold=0.5` 之上的事件进"高影响"区并可选 DeepSeek 深度分析。

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

## 项目结构

```
ddos/
├── config/                      # 配置文件
│   ├── config.yaml              # 主配置（模型/管线/评分）
│   ├── event_types.yaml         # A-G 事件分类体系（30 子类）
│   └── tracked_companies.yaml   # 跟踪公司名单
├── src/
│   ├── config.py                # Pydantic 配置加载 + 事件注册表
│   ├── database/                # SQLAlchemy ORM + Repository
│   ├── pipeline/                # 数据处理管线
│   │   ├── fetcher.py           # Tushare + 东方财富（列表 + 内容接口）
│   │   ├── preprocessor.py      # 文本清洗
│   │   ├── classifier.py        # BERT 分类步骤（无模型 fail-fast）
│   │   ├── extractor.py         # DeepSeek 字段提取
│   │   ├── scorer.py            # 规则评分引擎
│   │   ├── reporter.py          # 每日报告 + 深度分析
│   │   └── orchestrator.py      # 管线编排器
│   ├── ml/                      # ML 模块
│   │   ├── classifier_wrapper.py  # BERT 推理封装（温度校准）
│   │   └── llm_client.py          # LLM 客户端（OpenAI 兼容/DeepSeek）
│   ├── training/                # 模型训练
│   │   ├── dataset.py           # PyTorch Dataset
│   │   ├── train_classifier.py  # 微调脚本
│   │   └── evaluate.py          # 评估脚本
│   └── utils/                   # 工具函数
│       ├── text_utils.py        # 中文文本清理
│       └── date_utils.py        # 日期工具
├── scripts/                     # 可执行脚本
│   ├── run_pipeline.py          # 管线入口（--date / --stages）
│   ├── setup_model.py           # 一键复现分类器（gitignore 方案）
│   ├── generate_seed_data.py    # 合成种子训练数据
│   ├── calibrate_temperature.py # 温度校准
│   ├── check_confidence.py      # 置信度诊断
│   ├── seed_database.py         # 初始化数据库 + 股票列表
│   └── label_data.py            # 数据标注助手
├── tests/                       # 测试（71 passed, 2 skipped）
├── notebooks/                   # Jupyter 探索
├── data/                        # 运行时数据（gitignore）
└── models/                      # 训练好的模型（gitignore，setup_model.py 复现）
```

## 运行测试

```powershell
python -m pytest tests/ -q        # 全部测试（71 passed, 2 skipped）
python -m pytest tests/ -v        # 详细输出
```

2 个 skipped 是需要真实 Tushare token 的集成测试。

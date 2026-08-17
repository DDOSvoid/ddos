"""配置加载模块 — Pydantic + YAML。

从 config/config.yaml 加载基础配置，从环境变量 (.env) 覆盖敏感字段。
"""

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# ── 项目根目录 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"


def _load_env() -> None:
    """加载 .env 文件（如果存在）。"""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)


_load_env()


def _load_yaml(name: str) -> dict:
    """加载 YAML 配置文件。"""
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ── 配置模型 ─────────────────────────────────────────────────


class DatabaseConfig(BaseModel):
    url: str = f"sqlite:///{DATA_DIR}/ddos.db"
    echo: bool = False


class TushareConfig(BaseModel):
    token: Optional[str] = Field(default_factory=lambda: os.getenv("TUSHARE_TOKEN"))
    rate_limit_per_minute: int = 200


class EastmoneyConfig(BaseModel):
    base_url: str = "https://np-anotice-stock.eastmoney.com/api/security/ann"
    rate_limit_per_minute: int = 30
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )


class ClassifierModelConfig(BaseModel):
    name: str = "bert-base-chinese"
    local_path: str = "models/bert-classifier"
    max_length: int = 512


class ExtractionModelConfig(BaseModel):
    model: str = "deepseek-v4-flash"
    temperature: float = 0.0
    max_tokens: int = 1024
    request_timeout: int = 60


class AnalysisModelConfig(BaseModel):
    model: str = "deepseek-v4-flash"
    temperature: float = 0.3
    max_tokens: int = 2048
    request_timeout: int = 120


class ModelsConfig(BaseModel):
    classifier: ClassifierModelConfig = Field(default_factory=ClassifierModelConfig)
    extraction: ExtractionModelConfig = Field(default_factory=ExtractionModelConfig)
    analysis: AnalysisModelConfig = Field(default_factory=AnalysisModelConfig)


class PipelineConfig(BaseModel):
    fetch_lookback_days: int = 3
    fetch_backend: Literal["http", "cdp"] = "http"  # 公告抓取后端: http=requests / cdp=Playwright驱动Chrome
    fetch_max_stocks: int = 0  # 每次抓取的跟踪股上限，0=全部（原硬编码 10 已配置化）
    fetch_full_text: bool = True  # 抓取公告正文（内容接口）
    classifier_batch_size: int = 16
    extraction_min_confidence: float = 0.6
    report_high_impact_threshold: float = 0.5
    report_deep_analysis_top_n: int = 10
    report_tracked_only: bool = False


class ScoringDefaults(BaseModel):
    direction: float = 0.0
    magnitude: float = 0.5
    surprise: float = 0.5
    credibility: float = 0.8


class ScoringConfig(BaseModel):
    defaults: ScoringDefaults = Field(default_factory=ScoringDefaults)
    source_credibility: dict = Field(default_factory=lambda: {
        "regulatory_filing": 0.95,
        "company_announcement": 0.90,
        "media_report": 0.50,
        "analyst_report": 0.60,
    })


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/pipeline.log"
    rotation: str = "10 MB"
    retention: str = "30 days"


class Config(BaseModel):
    """全局配置聚合根。"""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    tushare: TushareConfig = Field(default_factory=TushareConfig)
    eastmoney: EastmoneyConfig = Field(default_factory=EastmoneyConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Optional[Path] = None) -> "Config":
        """从 YAML 文件加载配置，环境变量自动覆盖敏感字段。"""
        if path is None:
            path = CONFIG_DIR / "config.yaml"
        raw = _load_yaml(path.name) if path.parent == CONFIG_DIR else {}
        if not raw and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        return cls(**raw)

    def resolve_path(self, relative_path: str) -> Path:
        """将相对于项目根目录的路径转为绝对路径。"""
        return PROJECT_ROOT / relative_path


# ── 全局单例 ─────────────────────────────────────────────────
# 首次 import 时从 config.yaml 加载，后续模块直接引用此实例
config = Config.from_yaml()

# ── 事件分类元数据 ────────────────────────────────────────────


class SubCategoryDef(BaseModel):
    label: str
    direction_baseline: float = 0.0
    fields: list[str] = Field(default_factory=list)


class CategoryDef(BaseModel):
    label: str
    subcategories: dict[str, SubCategoryDef] = Field(default_factory=dict)


class EventTypeRegistry:
    """事件分类注册表 — 从 config/event_types.yaml 加载。"""

    def __init__(self) -> None:
        raw = _load_yaml("event_types.yaml")
        self.categories: dict[str, CategoryDef] = {}
        self._sub_to_major: dict[str, str] = {}
        for code, cat_data in raw.get("categories", {}).items():
            subcats = {}
            for sub_code, sub_data in cat_data.get("subcategories", {}).items():
                subcats[sub_code] = SubCategoryDef(**sub_data)
                self._sub_to_major[sub_code] = code
            self.categories[code] = CategoryDef(
                label=cat_data.get("label", ""),
                subcategories=subcats,
            )

    def get_major(self, sub_category: str) -> Optional[str]:
        """通过子类别代码获取大类别代码。"""
        return self._sub_to_major.get(sub_category)

    def get_sub_labels(self) -> list[str]:
        """获取所有子类别标签（用于分类模型的 30 个输出类别）。"""
        labels: list[str] = []
        for cat in self.categories.values():
            for sub_code, sub_def in cat.subcategories.items():
                labels.append(sub_code)
        return labels

    def num_subcategories(self) -> int:
        return sum(
            len(cat.subcategories) for cat in self.categories.values()
        )

    # 子类别 → 大类别的静态映射（用于模型输出后处理）
    @property
    def sub_to_major_map(self) -> dict[str, str]:
        return dict(self._sub_to_major)


# ── 行业划分注册表 ────────────────────────────────────────────


class IndustryRegistry:
    """行业划分注册表 — 从 config/industries.yaml 加载。

    将 Tushare 申万一级行业标签映射到粗分行业域，供 classify 步骤
    为公告写入 industry_group，便于下游按行业差异化处理。
    """

    def __init__(self) -> None:
        raw = _load_yaml("industries.yaml")
        self.default_group: str = raw.get("default_group", "其他")
        self.groups: dict[str, list[str]] = raw.get("groups", {})
        # industry → group 反向映射
        self._industry_to_group: dict[str, str] = {}
        for group, industries in self.groups.items():
            for industry in industries:
                self._industry_to_group[industry] = group

    def resolve(self, industry: str | None) -> str:
        """将申万一级行业解析为行业域；空/未命中归 default_group。"""
        if not industry:
            return self.default_group
        return self._industry_to_group.get(industry, self.default_group)

    def get_group_industries(self, group: str) -> list[str]:
        """获取某行业域下的所有申万一级行业。"""
        return list(self.groups.get(group, []))

    def group_names(self) -> list[str]:
        return list(self.groups.keys())


# 全局行业划分注册表单例
industry_registry = IndustryRegistry()

# 全局事件注册表单例
event_registry = EventTypeRegistry()

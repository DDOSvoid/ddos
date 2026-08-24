"""真实公告分类标注契约。

生产训练只接受经过人工确认的真实公告。规则和 AI 可以提出候选标签，但候选
标签不能进入验证集、测试集或置信度校准。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.config import event_registry

SCHEMA_VERSION = "real-label-v1"
DATASET_ROLES = {"train", "validation", "test"}
LABEL_STATUSES = {"candidate", "verified"}
LABEL_SOURCES = {
    "human",
    "human_verified_rule",
    "ai_candidate",
    "rule_candidate",
    "synthetic",
}
PRODUCTION_SOURCES = {"human", "human_verified_rule"}


@dataclass(frozen=True)
class VerifiedLabel:
    announcement_id: str
    text: str
    major_category: str
    sub_category: str
    published_at: datetime
    labeled_at: datetime
    dataset_role: str
    label_source: str
    reviewer: str


def _parse_time(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空 ISO 时间")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError(f"{field} 不是合法 ISO 时间: {value}") from exc


def validate_label_item(item: dict, *, production: bool = True) -> VerifiedLabel:
    """验证一条标注并返回强类型结果。"""
    if item.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version 必须为 {SCHEMA_VERSION}")
    if item.get("label_status") not in LABEL_STATUSES:
        raise ValueError("label_status 必须为 candidate 或 verified")
    if item.get("label_source") not in LABEL_SOURCES:
        raise ValueError(f"未知 label_source: {item.get('label_source')}")
    if production:
        if item.get("label_status") != "verified":
            raise ValueError("生产训练不接受 candidate 标签")
        if item.get("label_source") not in PRODUCTION_SOURCES:
            raise ValueError("生产训练只接受 human 或 human_verified_rule 标签")

    announcement_id = str(item.get("announcement_id") or "").strip()
    text = str(item.get("text") or "").strip()
    major = str(item.get("major_category") or "").strip()
    sub = str(item.get("sub_category") or "").strip()
    role = str(item.get("dataset_role") or "").strip()
    reviewer = str(item.get("reviewer") or "").strip()

    if not announcement_id:
        raise ValueError("announcement_id 不能为空")
    if len(text) < 10:
        raise ValueError("text 过短")
    if sub not in event_registry.sub_to_major_map:
        raise ValueError(f"sub_category 不在当前分类体系: {sub}")
    expected_major = event_registry.sub_to_major_map[sub]
    if major != expected_major:
        raise ValueError(f"major_category 应为 {expected_major}，实际为 {major}")
    if role not in DATASET_ROLES:
        raise ValueError("dataset_role 必须为 train/validation/test")
    if production and not reviewer:
        raise ValueError("生产标注必须记录 reviewer")

    published_at = _parse_time(item.get("published_at"), "published_at")
    labeled_at = _parse_time(item.get("labeled_at"), "labeled_at")
    if labeled_at < published_at:
        raise ValueError("labeled_at 不能早于 published_at")

    return VerifiedLabel(
        announcement_id=announcement_id,
        text=text,
        major_category=major,
        sub_category=sub,
        published_at=published_at,
        labeled_at=labeled_at,
        dataset_role=role,
        label_source=item["label_source"],
        reviewer=reviewer,
    )


def validate_temporal_roles(samples: list[VerifiedLabel]) -> None:
    """验证 train < validation < test 的全局时间顺序。"""
    by_role = {
        role: [sample.published_at for sample in samples if sample.dataset_role == role]
        for role in DATASET_ROLES
    }
    missing = [role for role, values in by_role.items() if not values]
    if missing:
        raise ValueError(f"数据集缺少时间分区: {', '.join(sorted(missing))}")
    if max(by_role["train"]) >= min(by_role["validation"]):
        raise ValueError("train 与 validation 时间重叠或倒置")
    if max(by_role["validation"]) >= min(by_role["test"]):
        raise ValueError("validation 与 test 时间重叠或倒置")

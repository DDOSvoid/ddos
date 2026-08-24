"""Role-gated datasets for causal impact modelling."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from src.database.models import (
    Announcement,
    AnnouncementMarketTarget,
    Classification,
    Company,
)
from src.prediction.splits import PredictionSplitContract

SEALED_ROLES = frozenset(
    {"test", "quarantine", "forward_validation"}
)


def _aware_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def load_company_day_samples(
    session: Session,
    *,
    role: str,
    split: PredictionSplitContract,
    allow_sealed: bool = False,
) -> list[dict]:
    """Load one explicit split role; sealed labels require deliberate authorization."""
    if role != "train" and role not in SEALED_ROLES:
        raise ValueError(f"unsupported eligible dataset role: {role}")
    if role in SEALED_ROLES and not allow_sealed:
        raise PermissionError(f"{role} labels are sealed")

    rows = (
        session.query(
            Announcement.id,
            Announcement.company_id,
            Company.stock_code,
            Announcement.published_date,
            Announcement.title,
            Classification.sub_category,
            Classification.needs_review,
            AnnouncementMarketTarget.horizon_sessions,
            AnnouncementMarketTarget.actual_direction,
            AnnouncementMarketTarget.outcome_available_at,
            AnnouncementMarketTarget.dataset_role,
            AnnouncementMarketTarget.split_contract,
            AnnouncementMarketTarget.split_source_sha256,
        )
        .join(Company, Company.id == Announcement.company_id)
        .join(Classification, Classification.announcement_id == Announcement.id)
        .join(
            AnnouncementMarketTarget,
            AnnouncementMarketTarget.announcement_id == Announcement.id,
        )
        .filter(AnnouncementMarketTarget.dataset_role == role)
        .filter(AnnouncementMarketTarget.split_contract == split.contract)
        .filter(AnnouncementMarketTarget.split_source_sha256 == split.source_sha256)
        .order_by(
            AnnouncementMarketTarget.horizon_sessions,
            Announcement.published_date,
            Announcement.company_id,
            Announcement.id,
        )
        .all()
    )

    groups = defaultdict(
        lambda: {
            "announcement_ids": [],
            "titles": [],
            "sub_categories": set(),
            "has_accepted_classification": False,
        }
    )
    for row in rows:
        if split.role_for(row.published_date, _aware_utc(row.outcome_available_at)) != role:
            raise ValueError(f"row has invalid split membership: announcement {row.id}")
        key = (row.company_id, row.published_date, row.horizon_sessions)
        group = groups[key]
        group["company_id"] = row.company_id
        group["stock_code"] = row.stock_code
        group["published_date"] = row.published_date
        group["horizon_sessions"] = row.horizon_sessions
        group["announcement_ids"].append(row.id)
        group["titles"].append((row.id, row.title))
        group["sub_categories"].add(row.sub_category)
        group["has_accepted_classification"] |= not bool(row.needs_review)
        direction = 1 if row.actual_direction > 0 else 0
        if "target" in group and group["target"] != direction:
            raise ValueError(f"inconsistent company-day target: {key}")
        group["target"] = direction
        available_at = _aware_utc(row.outcome_available_at)
        group["outcome_available_at"] = max(
            available_at,
            group.get("outcome_available_at", available_at),
        )

    samples = []
    for group in groups.values():
        if not group["has_accepted_classification"]:
            continue
        titles = [title for _, title in sorted(set(group["titles"]))]
        categories = " ".join(
            f"类别{category}" for category in sorted(group["sub_categories"])
        )
        samples.append(
            {
                **group,
                "text": f"{categories} 公告 " + "；".join(titles),
            }
        )
    return samples

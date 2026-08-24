"""Immutable announcement source archive tests."""

from datetime import UTC, date, datetime

import pytest

from src.database.models import Announcement, AnnouncementSourceArchive, Company
from src.pipeline.announcement_content import assemble_content_pages
from src.pipeline.source_archive import archive_announcement_content
from src.prediction.development_contract import load_causal_development_contract
from src.prediction.splits import load_prediction_split_contract


def _announcement(db_session, *, published_date: date) -> Announcement:
    company = Company(
        stock_code="000001.SZ",
        stock_name="测试公司",
        exchange="SZSE",
    )
    db_session.add(company)
    db_session.flush()
    announcement = Announcement(
        company_id=company.id,
        announcement_id="AN202406011234567890",
        title="测试公告",
        published_date=published_date,
        processing_status="fetched",
    )
    db_session.add(announcement)
    db_session.flush()
    return announcement


def _content(second: str = "second") -> dict:
    return assemble_content_pages(
        [
            {
                "art_code": "AN202406011234567890",
                "notice_title": "测试公告",
                "notice_date": "2024-06-02 00:00:00",
                "eitime": "2024-06-01 18:00:00",
                "page_size": 2,
                "notice_content": "first",
            },
            {
                "art_code": "AN202406011234567890",
                "page_size": 2,
                "notice_content": second,
            },
        ],
        expected_pages=2,
    )


def test_identical_snapshot_is_idempotent_and_changed_source_appends(db_session):
    announcement = _announcement(db_session, published_date=date(2024, 6, 2))
    split = load_prediction_split_contract()
    development = load_causal_development_contract(split=split)
    kwargs = {
        "announcement": announcement,
        "dataset_role": "train",
        "retrieved_at": datetime(2026, 8, 21, tzinfo=UTC),
        "development": development,
        "split": split,
    }
    first, created = archive_announcement_content(
        db_session, content=_content(), **kwargs
    )
    assert created is True
    same, created = archive_announcement_content(
        db_session, content=_content(), **kwargs
    )
    assert created is False
    assert same.id == first.id

    changed, created = archive_announcement_content(
        db_session, content=_content("changed"), **kwargs
    )
    assert created is True
    assert changed.supersedes_id == first.id
    assert db_session.query(AnnouncementSourceArchive).count() == 2


def test_source_archive_rejects_update_and_delete(db_session):
    announcement = _announcement(db_session, published_date=date(2024, 6, 2))
    split = load_prediction_split_contract()
    row, _ = archive_announcement_content(
        db_session,
        announcement=announcement,
        content=_content(),
        dataset_role="train",
        retrieved_at=datetime(2026, 8, 21, tzinfo=UTC),
        development=load_causal_development_contract(split=split),
        split=split,
    )
    db_session.commit()
    row.content_text = "overwrite"
    with pytest.raises(ValueError, match="immutable"):
        db_session.commit()
    db_session.rollback()
    stored = db_session.query(AnnouncementSourceArchive).one()
    db_session.delete(stored)
    with pytest.raises(ValueError, match="append-only"):
        db_session.commit()


def test_source_archive_refuses_test_partition(db_session):
    announcement = _announcement(db_session, published_date=date(2025, 2, 1))
    split = load_prediction_split_contract()
    with pytest.raises(PermissionError, match="train role only"):
        archive_announcement_content(
            db_session,
            announcement=announcement,
            content=_content(),
            dataset_role="test",
            retrieved_at=datetime(2026, 8, 21, tzinfo=UTC),
            development=load_causal_development_contract(split=split),
            split=split,
        )

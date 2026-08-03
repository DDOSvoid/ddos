"""测试数据库 Repository 层。"""

from datetime import date

import pytest

from src.database.models import Announcement, Company
from src.database.repository import (
    AnnouncementRepository,
    ClassificationRepository,
    CompanyRepository,
    ExtractedFieldRepository,
    ScoreRepository,
)


class TestCompanyRepository:
    def test_upsert_new(self, db_session, sample_company_data):
        company = CompanyRepository.upsert(db_session, **sample_company_data)
        db_session.commit()

        assert company.id is not None
        assert company.stock_code == "000001.SZ"
        assert company.is_tracked is True

    def test_upsert_existing(self, db_session, sample_company_data):
        company1 = CompanyRepository.upsert(db_session, **sample_company_data)
        db_session.commit()

        # Upsert with new name
        company2 = CompanyRepository.upsert(
            db_session, stock_code="000001.SZ", stock_name="平安银行(更名)"
        )
        db_session.commit()

        assert company2.id == company1.id
        assert company2.stock_name == "平安银行(更名)"

    def test_get_tracked(self, db_session, sample_company_data):
        CompanyRepository.upsert(db_session, **sample_company_data)
        CompanyRepository.upsert(
            db_session,
            stock_code="000002.SZ",
            stock_name="万科A",
            exchange="SZSE",
            industry="房地产",
            is_tracked=False,
        )
        db_session.commit()

        tracked = CompanyRepository.get_tracked(db_session)
        assert len(tracked) == 1
        assert tracked[0].stock_code == "000001.SZ"

    def test_get_by_code(self, db_session, sample_company_data):
        CompanyRepository.upsert(db_session, **sample_company_data)
        db_session.commit()

        found = CompanyRepository.get_by_code(db_session, "000001.SZ")
        assert found is not None
        assert found.stock_name == "平安银行"

        not_found = CompanyRepository.get_by_code(db_session, "999999.SZ")
        assert not_found is None


class TestAnnouncementRepository:
    def _seed_company(self, db_session):
        """Helper: 插入一家公司。"""
        company = Company(
            stock_code="000001.SZ",
            stock_name="平安银行",
            exchange="SZSE",
        )
        db_session.add(company)
        db_session.flush()
        return company

    def test_bulk_upsert(self, db_session):
        company = self._seed_company(db_session)

        records = [
            {
                "company_id": company.id,
                "announcement_id": "ANN-001",
                "title": "2024年一季报",
                "full_text": "营收100亿",
                "published_date": date(2024, 4, 25),
            },
            {
                "company_id": company.id,
                "announcement_id": "ANN-002",
                "title": "回购公告",
                "full_text": "拟回购1亿元",
                "published_date": date(2024, 4, 26),
            },
        ]
        count = AnnouncementRepository.bulk_upsert(db_session, records)
        db_session.commit()

        assert count == 2

    def test_bulk_upsert_duplicate(self, db_session):
        company = self._seed_company(db_session)

        records = [
            {
                "company_id": company.id,
                "announcement_id": "ANN-001",
                "title": "2024年一季报",
                "full_text": "营收100亿",
                "published_date": date(2024, 4, 25),
            },
        ]
        AnnouncementRepository.bulk_upsert(db_session, records)
        db_session.commit()

        # 再次 upsert，应更新而非新增
        records[0]["title"] = "2024年一季报(更新)"
        count = AnnouncementRepository.bulk_upsert(db_session, records)
        db_session.commit()

        assert count == 0  # 无新增
        ann = db_session.query(Announcement).filter_by(announcement_id="ANN-001").first()
        assert ann.title == "2024年一季报(更新)"

    def test_get_by_status(self, db_session):
        company = self._seed_company(db_session)
        ann = Announcement(
            company_id=company.id,
            announcement_id="ANN-003",
            title="测试公告",
            published_date=date(2024, 4, 25),
            processing_status="fetched",
        )
        db_session.add(ann)
        db_session.commit()

        results = AnnouncementRepository.get_by_status(db_session, "fetched")
        assert len(results) == 1
        assert results[0].announcement_id == "ANN-003"

    def test_update_status(self, db_session):
        company = self._seed_company(db_session)
        ann = Announcement(
            company_id=company.id,
            announcement_id="ANN-004",
            title="测试公告",
            published_date=date(2024, 4, 25),
            processing_status="fetched",
        )
        db_session.add(ann)
        db_session.commit()

        AnnouncementRepository.update_status(db_session, ann.id, "preprocessed")
        db_session.commit()

        updated = db_session.query(Announcement).filter_by(id=ann.id).first()
        assert updated.processing_status == "preprocessed"


class TestClassificationRepository:
    def _seed_announcement(self, db_session) -> Announcement:
        company = Company(
            stock_code="000001.SZ",
            stock_name="平安银行",
            exchange="SZSE",
        )
        db_session.add(company)
        db_session.flush()

        ann = Announcement(
            company_id=company.id,
            announcement_id="ANN-CLS-001",
            title="分类测试",
            published_date=date(2024, 4, 25),
        )
        db_session.add(ann)
        db_session.flush()
        return ann

    def test_upsert(self, db_session):
        ann = self._seed_announcement(db_session)

        cls_result = ClassificationRepository.upsert(
            db_session,
            announcement_id=ann.id,
            major_category="A",
            sub_category="earnings_q1",
            confidence=0.92,
        )
        db_session.commit()

        assert cls_result.major_category == "A"
        assert cls_result.sub_category == "earnings_q1"


class TestScoreRepository:
    def _seed_announcement(self, db_session) -> Announcement:
        company = Company(
            stock_code="000001.SZ",
            stock_name="平安银行",
            exchange="SZSE",
        )
        db_session.add(company)
        db_session.flush()

        ann = Announcement(
            company_id=company.id,
            announcement_id="ANN-SCR-001",
            title="评分测试",
            published_date=date(2024, 4, 25),
        )
        db_session.add(ann)
        db_session.flush()
        return ann

    def test_upsert(self, db_session):
        ann = self._seed_announcement(db_session)

        score = ScoreRepository.upsert(
            db_session,
            announcement_id=ann.id,
            direction=0.5,
            magnitude=0.6,
            surprise=0.4,
            credibility=0.85,
            composite_score=0.102,
            scoring_detail={"reason": "test"},
        )
        db_session.commit()

        assert score.composite_score == 0.102
        assert score.direction_label == "利好"
        assert not score.is_high_impact

    def test_high_impact(self, db_session):
        ann = self._seed_announcement(db_session)

        ScoreRepository.upsert(
            db_session,
            announcement_id=ann.id,
            direction=0.8,
            magnitude=0.9,
            surprise=0.8,
            credibility=0.95,
            composite_score=0.547,  # 0.8*0.9*0.8*0.95
        )
        db_session.commit()

        scores = ScoreRepository.get_high_impact(db_session, threshold=0.5)
        assert len(scores) == 1
        assert scores[0].is_high_impact

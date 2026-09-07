import json
from datetime import date

from scripts import backfill_announcement_source_archives as module
from src.database.models import Announcement, AnnouncementSourceArchive, Company
from src.pipeline.announcement_content import assemble_content_pages


def test_explicit_pdf_plan_advances_across_limited_batches(db_session, monkeypatch, tmp_path):
    company = Company(stock_code="000001.SZ", stock_name="Test", exchange="SZSE")
    db_session.add(company)
    db_session.flush()
    announcements = [
        Announcement(
            company_id=company.id,
            announcement_id=f"A{index}",
            title=f"Title {index}",
            published_date=date(2023, index, 1),
            processing_status="fetched",
        )
        for index in (1, 2)
    ]
    db_session.add_all(announcements)
    db_session.commit()
    engine = db_session.get_bind()

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def fetch_announcement_content(self, announcement_id):
            index = int(announcement_id[1:])
            return assemble_content_pages(
                [
                    {
                        "art_code": announcement_id,
                        "notice_title": f"Title {index}",
                        "notice_date": f"2023-0{index}-01 00:00:00",
                        "page_size": 1,
                        "notice_content": f"body {index}",
                    }
                ],
                expected_pages=1,
            )

    monkeypatch.setattr(module, "init_db", lambda: None)
    monkeypatch.setattr(module, "get_engine", lambda: engine)
    monkeypatch.setattr(module, "EastmoneyClient", FakeClient)
    state = tmp_path / "archive.state.json"
    report = tmp_path / "archive.report.json"
    kwargs = {
        "announcement_ids": ["A1", "A2"],
        "limit": 1,
        "pdf_mode": "none",
        "only_accepted": False,
        "relevances": [],
        "rate_limit_per_minute": 30,
        "refresh": False,
        "state_path": state,
        "report_path": report,
        "consecutive_failure_limit": 3,
    }
    assert module.backfill(**kwargs)["announcements_selected"] == 1
    assert module.backfill(**kwargs)["announcements_selected"] == 1
    saved_state = json.loads(state.read_text(encoding="utf-8"))
    saved_state["failures"] = [
        {
            "announcement_db_id": announcements[0].id,
            "announcement_id": "A1",
            "error_type": "ValueError",
            "error": "stale failure",
        }
    ]
    state.write_text(json.dumps(saved_state), encoding="utf-8")
    recovered = module.backfill(**kwargs)
    assert recovered["announcements_selected"] == 1
    assert recovered["unresolved_failures"] == []
    assert module.backfill(**kwargs)["announcements_selected"] == 0
    assert db_session.query(AnnouncementSourceArchive).count() == 2


def test_client_factory_selects_cdp_and_bypasses_proxy(monkeypatch):
    from src.pipeline import cdp_fetcher

    captured = {}

    class FakeCdpClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(cdp_fetcher, "CdpEastmoneyClient", FakeCdpClient)
    client = module._build_eastmoney_client(
        fetch_backend="cdp", rate_limit_per_minute=17
    )

    assert isinstance(client, FakeCdpClient)
    assert captured["rate_limit_per_minute"] == 17
    assert captured["bypass_system_proxy"] is True


def test_date_conflict_remains_unarchived_without_blocking_later_rows(
    db_session, monkeypatch, tmp_path
):
    company = Company(stock_code="000002.SZ", stock_name="Test", exchange="SZSE")
    db_session.add(company)
    db_session.flush()
    for aid in ("BAD", "GOOD"):
        db_session.add(Announcement(
            company_id=company.id, announcement_id=aid, title=aid,
            published_date=date(2023, 4, 4), processing_status="fetched",
        ))
    db_session.commit()
    calls = []

    class FakeClient:
        def fetch_announcement_content(self, aid):
            calls.append(aid)
            return assemble_content_pages([dict(
                art_code=aid, notice_title=aid, page_size=1, notice_content="body",
                notice_date="2023-04-03" if aid == "BAD" else "2023-04-04",
            )], expected_pages=1)

    monkeypatch.setattr(module, "init_db", lambda: None)
    monkeypatch.setattr(module, "get_engine", db_session.get_bind)
    monkeypatch.setattr(module, "_build_eastmoney_client", lambda **_: FakeClient())
    kwargs = dict(
        announcement_ids=["BAD", "GOOD"], limit=1, pdf_mode="none",
        only_accepted=False, relevances=[], rate_limit_per_minute=30, refresh=False,
        state_path=tmp_path / "state.json", report_path=tmp_path / "report.json",
        consecutive_failure_limit=3,
    )
    first = module.backfill(**kwargs)
    assert first["unresolved_failures"][0]["retryable"] is False
    assert module.backfill(**kwargs)["counts"]["created"] == 1
    final = module.backfill(**kwargs)
    assert final["announcements_selected"] == 0
    assert final["unresolved_failures"][0]["announcement_id"] == "BAD"
    assert calls == ["BAD", "GOOD"]
    assert db_session.query(AnnouncementSourceArchive).count() == 1
    assert "finished_at" not in json.loads(kwargs["state_path"].read_text())

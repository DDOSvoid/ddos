import pytest

from src.pipeline.fetcher import EastmoneyClient


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"data": {"list": [], "total_page": 0}}


def test_announcement_page_retries_then_succeeds(monkeypatch):
    client = EastmoneyClient()
    calls = 0

    def get(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("temporary")
        return _Response()

    monkeypatch.setattr(client, "_rate_limit", lambda: None)
    monkeypatch.setattr(client.session, "get", get)
    monkeypatch.setattr("src.pipeline.fetcher.time.sleep", lambda seconds: None)
    result = client.fetch_announcements(max_attempts=4, raise_on_error=True)
    assert result == {"data": {"list": [], "total_page": 0}}
    assert calls == 3


def test_strict_announcement_page_failure_is_not_empty_success(monkeypatch):
    client = EastmoneyClient()
    monkeypatch.setattr(client, "_rate_limit", lambda: None)
    monkeypatch.setattr(
        client.session,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError("down")),
    )
    monkeypatch.setattr("src.pipeline.fetcher.time.sleep", lambda seconds: None)
    with pytest.raises(ConnectionError, match="down"):
        client.fetch_all_announcements(
            max_pages=2,
            max_attempts=2,
            raise_on_error=True,
        )

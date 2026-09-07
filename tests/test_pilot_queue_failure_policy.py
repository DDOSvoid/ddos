import json

import pytest

from scripts import run_electrical_equipment_pilot_queue as queue
from scripts.backfill_announcement_source_archives import _retryable_failure


def test_only_known_date_conflict_is_nonretryable():
    assert not _retryable_failure(dict(
        error_type="ValueError", error="source notice_date does not match announcement"
    ))
    assert _retryable_failure(dict(error_type="ValueError", error="empty content response"))
    assert _retryable_failure(dict(error_type="RuntimeError", error="network error"))


def setup_queue(monkeypatch, tmp_path, report):
    monkeypatch.setattr(queue, "QUEUE_STATE", tmp_path / "queue.json")
    monkeypatch.setattr(queue.time, "sleep", lambda _: None)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    calls = []
    monkeypatch.setattr(queue, "_run", lambda *args: calls.append(1))
    return path, calls


def test_quarantine_preserves_failure_and_allows_next_stage(monkeypatch, tmp_path):
    failure = dict(announcement_id="A1", retryable=False)
    path, calls = setup_queue(monkeypatch, tmp_path, dict(
        announcements_selected=0, counts={}, unresolved_failures=[failure]
    ))
    state = {}
    for _ in range(2):
        queue._run_archive_until_complete(
            name="body", arguments=[], report_path=path, state=state
        )
    assert calls == [1]
    assert state["stages"]["body"]["status"] == "needs_review"
    assert state["stages"]["body"]["failures"] == [failure]


def test_transient_no_progress_is_bounded_and_persists(monkeypatch, tmp_path):
    path, calls = setup_queue(monkeypatch, tmp_path, dict(
        announcements_selected=1, counts={"failed": 1},
        unresolved_failures=[dict(retryable=True)]
    ))
    state = {}
    with pytest.raises(RuntimeError, match="no progress"):
        queue._run_archive_until_complete(
            name="body", arguments=[], report_path=path, state=state
        )
    assert len(calls) == 3
    assert state["stages"]["body"]["status"] == "blocked"
    assert json.loads(queue.QUEUE_STATE.read_text())["stages"]["body"]["no_progress_rounds"] == 3


def test_invalid_report_cannot_mark_complete(monkeypatch, tmp_path):
    path, _ = setup_queue(monkeypatch, tmp_path, {})
    with pytest.raises(RuntimeError, match="invalid archive report"):
        queue._run_archive_until_complete(
            name="body", arguments=[], report_path=path, state={}
        )

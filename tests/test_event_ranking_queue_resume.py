import json

from scripts import run_event_ranking_data_queue as module


def test_run_once_skips_completed_stage_with_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    state = {"stages": {"example": {"status": "completed"}}}
    calls = []
    monkeypatch.setattr(module, "_run", lambda *args, **kwargs: calls.append("run"))
    monkeypatch.setattr(
        module, "_set_stage", lambda *args, **kwargs: calls.append("set")
    )

    module._run_once("example", ["unused.py"], state, artifact=artifact)

    assert calls == []


def test_stateful_stage_skips_only_with_clean_finished_state(monkeypatch, tmp_path):
    child_state = tmp_path / "child.json"
    child_state.write_text(
        json.dumps({"finished_at": "2026-09-02T00:00:00Z", "failures": {}}),
        encoding="utf-8",
    )
    state = {"stages": {"example": {"status": "completed"}}}
    calls = []
    monkeypatch.setattr(module, "_run", lambda *args, **kwargs: calls.append("run"))

    module._run_stateful_until_finished(
        name="example",
        arguments=["unused.py"],
        state_path=child_state,
        finished_key="finished_at",
        state=state,
    )

    assert calls == []

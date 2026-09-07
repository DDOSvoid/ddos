#!/usr/bin/env python
"""Run the resumable event-ranking train-data queue after metadata completes."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVENT_ROOT = PROJECT_ROOT / "data" / "backfills" / "event_ranking_v1"
METADATA_STATE = EVENT_ROOT / "full_announcements_train_2023_2024.json"
QUEUE_STATE = EVENT_ROOT / "data_queue.state.json"
VALIDATION_ROOT = PROJECT_ROOT / "data" / "validation" / "event_ranking_v1"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_state(state: dict[str, object]) -> None:
    state["updated_at"] = datetime.now(UTC).isoformat()
    state["official_test_queried"] = False
    QUEUE_STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = QUEUE_STATE.with_suffix(QUEUE_STATE.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(QUEUE_STATE)


def _set_stage(state: dict[str, object], name: str, status: str, **extra: object) -> None:
    stages = state.setdefault("stages", {})
    assert isinstance(stages, dict)
    record = stages.setdefault(name, {})
    assert isinstance(record, dict)
    record.update({"status": status, "updated_at": datetime.now(UTC).isoformat(), **extra})
    _save_state(state)


def _run(name: str, arguments: list[str], state: dict[str, object]) -> None:
    command = [sys.executable, *arguments]
    for attempt in range(1, 6):
        _set_stage(state, name, "running", attempt=attempt)
        print(f"queue stage={name} attempt={attempt}", flush=True)
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode == 0:
            return
        _set_stage(
            state,
            name,
            "retrying",
            attempt=attempt,
            return_code=completed.returncode,
        )
        if attempt < 5:
            time.sleep(min(attempt * 60, 300))
    raise RuntimeError(f"queue stage failed after retries: {name}")


def _stage_completed(state: dict[str, object], name: str) -> bool:
    stages = state.get("stages") or {}
    record = stages.get(name) if isinstance(stages, dict) else None
    return isinstance(record, dict) and record.get("status") == "completed"


def _run_once(
    name: str,
    arguments: list[str],
    state: dict[str, object],
    *,
    artifact: Path | None = None,
) -> None:
    if _stage_completed(state, name) and (artifact is None or artifact.exists()):
        print(f"queue stage={name} already completed; skipping", flush=True)
        return
    _run(name, arguments, state)
    _set_stage(state, name, "completed")


def _metadata_complete() -> tuple[bool, int, int]:
    state = _load_json(METADATA_STATE)
    progress = state.get("progress") or {}
    completed = int(progress.get("completed") or 0)
    total = int(progress.get("total") or 0)
    return total > 0 and completed == total and bool(state.get("finished_at")), completed, total


def _run_archive_until_complete(
    *,
    name: str,
    arguments: list[str],
    report_path: Path,
    state: dict[str, object],
) -> None:
    if _stage_completed(state, name) and report_path.exists():
        print(f"queue stage={name} already completed; skipping", flush=True)
        return
    rounds = 0
    while True:
        rounds += 1
        _run(name, arguments, state)
        report = _load_json(report_path)
        selected = int(report.get("announcements_selected") or 0)
        failures = list(report.get("unresolved_failures") or [])
        counts = report.get("counts") or {}
        _set_stage(
            state,
            name,
            "running",
            rounds=rounds,
            selected_last_round=selected,
            processed_last_round=int(counts.get("processed") or 0),
            unresolved_failures=len(failures),
        )
        if selected == 0 and not failures:
            _set_stage(state, name, "completed", rounds=rounds)
            return
        if failures:
            time.sleep(60)


def _run_stateful_until_finished(
    *,
    name: str,
    arguments: list[str],
    state_path: Path,
    finished_key: str,
    state: dict[str, object],
) -> None:
    if _stage_completed(state, name):
        child = _load_json(state_path)
        if child.get(finished_key) and not child.get("failures"):
            print(f"queue stage={name} already completed; skipping", flush=True)
            return
    for round_number in range(1, 11):
        _run(name, arguments, state)
        child = _load_json(state_path)
        completed = len(child.get("completed") or {})
        failures = len(child.get("failures") or {})
        _set_stage(
            state,
            name,
            "running",
            rounds=round_number,
            completed_items=completed,
            failures=failures,
        )
        if child.get(finished_key) and failures == 0:
            _set_stage(state, name, "completed", rounds=round_number)
            return
        time.sleep(min(round_number * 60, 300))
    raise RuntimeError(f"stateful stage did not finish cleanly: {name}")


def main() -> None:
    state: dict[str, object] = _load_json(QUEUE_STATE)
    state.setdefault("contract", "event-ranking-train-data-queue-v1")
    state.setdefault("created_at", datetime.now(UTC).isoformat())
    state["process_id"] = __import__("os").getpid()
    state["status"] = "running"
    state.pop("stopped_at", None)
    state.pop("stop_reason", None)
    state.pop("error_type", None)
    state.pop("error", None)
    _save_state(state)
    try:
        last_progress = None
        while True:
            ready, completed, total = _metadata_complete()
            if ready:
                break
            progress = (completed, total)
            if progress != last_progress:
                print(f"waiting for metadata {completed}/{total}", flush=True)
                _set_stage(
                    state,
                    "announcement_metadata",
                    "running",
                    completed_segments=completed,
                    total_segments=total,
                )
                last_progress = progress
            time.sleep(30)
        _set_stage(
            state,
            "announcement_metadata",
            "completed",
            completed_segments=completed,
            total_segments=total,
        )

        _run_once(
            "train_universe",
            ["scripts/build_event_train_universe.py"],
            state,
            artifact=EVENT_ROOT / "train_universe.json",
        )

        classification_report = EVENT_ROOT / "title_classification.json"
        _run_once(
            "title_classification",
            [
                "scripts/classify_historical_rules.py",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2024-12-31",
                "--report",
                str(classification_report),
            ],
            state,
            artifact=classification_report,
        )

        pdf_plan = EVENT_ROOT / "pdf_audit_plan.json"
        _run_once(
            "content_plan",
            ["scripts/build_event_content_plans.py", "--output", str(pdf_plan)],
            state,
            artifact=pdf_plan,
        )

        body_report = VALIDATION_ROOT / "announcement_body_core.report.json"
        _run_archive_until_complete(
            name="announcement_body_core",
            arguments=[
                "scripts/backfill_announcement_source_archives.py",
                "--only-accepted",
                "--relevance",
                "core_event",
                "--rate-limit",
                "30",
                "--pdf-mode",
                "none",
                "--fetch-backend",
                "cdp",
                "--fallback-to-pdf",
                "--limit",
                "1000",
                "--state",
                str(EVENT_ROOT / "announcement_body_core.state.json"),
                "--report",
                str(body_report),
            ],
            report_path=body_report,
            state=state,
        )

        pdf_report = VALIDATION_ROOT / "announcement_pdf_audit.report.json"
        _run_archive_until_complete(
            name="announcement_pdf_audit",
            arguments=[
                "scripts/backfill_announcement_source_archives.py",
                "--announcement-plan",
                str(pdf_plan),
                "--rate-limit",
                "30",
                "--pdf-mode",
                "archive",
                "--fetch-backend",
                "cdp",
                "--limit",
                "100",
                "--state",
                str(EVENT_ROOT / "announcement_pdf_audit.state.json"),
                "--report",
                str(pdf_report),
            ],
            report_path=pdf_report,
            state=state,
        )

        market_state = EVENT_ROOT / "market_prices.state.json"
        _run_stateful_until_finished(
            name="market_prices",
            arguments=[
                "scripts/backfill_market_prices.py",
                "--state",
                str(market_state),
            ],
            state_path=market_state,
            finished_key="finished_at",
            state=state,
        )

        target_report = VALIDATION_ROOT / "market_targets_train.report.json"
        _run_once(
            "market_targets_train",
            ["scripts/build_market_targets.py", "--output", str(target_report)],
            state,
            artifact=target_report,
        )

        daily_basic_state = PROJECT_ROOT / "data" / "tabular" / "daily_basic" / "state.json"
        _run_stateful_until_finished(
            name="daily_basic_train",
            arguments=["scripts/backfill_tabular_daily_basic.py"],
            state_path=daily_basic_state,
            finished_key="finished_at_utc",
            state=state,
        )

        pit_root = PROJECT_ROOT / "data" / "tabular" / "point_in_time_sources"
        for group, filename in (
            ("company_industry", "company_industry.state.json"),
            ("fundamentals", "fundamentals.state.json"),
        ):
            _run_stateful_until_finished(
                name=f"point_in_time_{group}",
                arguments=[
                    "scripts/backfill_tabular_point_in_time_sources.py",
                    "--group",
                    group,
                ],
                state_path=pit_root / filename,
                finished_key="finished_at_utc",
                state=state,
            )

        state["finished_at"] = datetime.now(UTC).isoformat()
        state["status"] = "completed"
        _save_state(state)
        print("event-ranking data queue completed", flush=True)
    except Exception as error:
        state["status"] = "failed"
        state["error_type"] = type(error).__name__
        state["error"] = str(error)[:1000]
        _save_state(state)
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run the resumable electrical-equipment feasibility data queue."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PILOT_ROOT = PROJECT_ROOT / "data" / "backfills" / "event_ranking_pilot_electrical_v1"
VALIDATION_ROOT = PROJECT_ROOT / "data" / "validation" / "event_ranking_pilot_electrical_v1"
TABULAR_ROOT = PROJECT_ROOT / "data" / "tabular" / "event_ranking_pilot_electrical_v1"
QUEUE_STATE = PILOT_ROOT / "data_queue.state.json"
BODY_PLAN = PILOT_ROOT / "body_plan.json"
PDF_PLAN = PILOT_ROOT / "pdf_plan.json"
UNIVERSE = PILOT_ROOT / "train_universe.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_state(state: dict[str, object]) -> None:
    state.update(
        {
            "updated_at": datetime.now(UTC).isoformat(),
            "official_test_queried": False,
            "market_targets_outside_plan_queried": False,
            "forward_validation_queried": False,
        }
    )
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
        print(f"pilot queue stage={name} attempt={attempt}", flush=True)
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode == 0:
            return
        _set_stage(
            state, name, "retrying", attempt=attempt, return_code=completed.returncode
        )
        if attempt < 5:
            time.sleep(min(attempt * 60, 300))
    raise RuntimeError(f"pilot queue stage failed after retries: {name}")


def _completed(state: dict[str, object], name: str) -> bool:
    stages = state.get("stages") or {}
    record = stages.get(name) if isinstance(stages, dict) else None
    return isinstance(record, dict) and record.get("status") == "completed"


def _run_once(
    name: str, arguments: list[str], state: dict[str, object], *, artifact: Path
) -> None:
    if _completed(state, name) and artifact.exists():
        print(f"pilot queue stage={name} already completed; skipping", flush=True)
        return
    _run(name, arguments, state)
    _set_stage(state, name, "completed")


def _run_archive_until_complete(
    *, name: str, arguments: list[str], report_path: Path, state: dict[str, object]
) -> None:
    prior = (state.get("stages") or {}).get(name, {})
    if prior.get("status") == "needs_review":
        print(f"pilot queue stage={name} isolated; manual review required", flush=True)
        return
    if _completed(state, name) and report_path.exists():
        print(f"pilot queue stage={name} already completed; skipping", flush=True)
        return
    rounds = 0
    while True:
        rounds += 1
        _run(name, arguments, state)
        report = _load_json(report_path)
        if not {"announcements_selected", "counts", "unresolved_failures"} <= report.keys():
            raise RuntimeError(f"missing or invalid archive report: {report_path}")
        selected = int(report.get("announcements_selected") or 0)
        failures = list(report.get("unresolved_failures") or [])
        counts = report.get("counts") or {}
        no_progress = int(
            (state.get("stages") or {}).get(name, {}).get("no_progress_rounds", 0)
        )
        progressed = any(int(counts.get(key) or 0) for key in (
            "created", "identical_existing", "skipped_existing"
        ))
        no_progress = 0 if progressed else no_progress + 1
        _set_stage(
            state,
            name,
            "running",
            rounds=rounds,
            selected_last_round=selected,
            processed_last_round=int(counts.get("processed") or 0),
            created_last_round=int(counts.get("created") or 0),
            skipped_existing_last_round=int(counts.get("skipped_existing") or 0),
            unresolved_failures=len(failures),
            no_progress_rounds=no_progress,
        )
        if selected == 0 and not failures:
            _set_stage(state, name, "completed", rounds=rounds)
            return
        if selected == 0 and failures and all(
            item.get("retryable") is False for item in failures
        ):
            _set_stage(
                state, name, "needs_review", failures=failures,
                reason="deterministic source conflicts isolated; not complete",
            )
            return
        if no_progress >= 3:
            _set_stage(state, name, "blocked", failures=failures)
            raise RuntimeError(f"archive stage made no progress for 3 rounds: {name}")
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
    if _completed(state, name):
        child = _load_json(state_path)
        if child.get(finished_key) and not child.get("failures"):
            print(f"pilot queue stage={name} already completed; skipping", flush=True)
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
    raise RuntimeError(f"pilot stateful stage did not finish cleanly: {name}")


def _require_frozen_inputs() -> None:
    for path in (PILOT_ROOT / "manifest.json", BODY_PLAN, PDF_PLAN, UNIVERSE):
        if not path.exists():
            raise FileNotFoundError(f"pilot input is missing: {path}")
    for path in (BODY_PLAN, PDF_PLAN, UNIVERSE):
        value = _load_json(path)
        if value.get("official_test_queried") is not False:
            raise PermissionError(f"pilot input lacks official-test isolation: {path}")
        if value.get("forward_validation_queried") is not False:
            raise PermissionError(f"pilot input lacks 2026 isolation: {path}")


def main() -> None:
    _require_frozen_inputs()
    state: dict[str, object] = _load_json(QUEUE_STATE)
    state.setdefault("contract", "electrical-equipment-pilot-data-queue-v1")
    state.setdefault("created_at", datetime.now(UTC).isoformat())
    state["process_id"] = os.getpid()
    state["status"] = "running"
    state.pop("error_type", None)
    state.pop("error", None)
    _save_state(state)
    VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        body_report = VALIDATION_ROOT / "announcement_bodies.report.json"
        _run_archive_until_complete(
            name="sampled_announcement_bodies",
            arguments=[
                "scripts/backfill_announcement_source_archives.py",
                "--announcement-plan",
                str(BODY_PLAN),
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
                "--page-cache",
                str(PILOT_ROOT / "content_page_cache"),
                "--limit",
                "1000",
                "--state",
                str(PILOT_ROOT / "announcement_bodies.state.json"),
                "--report",
                str(body_report),
            ],
            report_path=body_report,
            state=state,
        )

        pdf_report = VALIDATION_ROOT / "announcement_pdf_audit.report.json"
        _run_archive_until_complete(
            name="sampled_pdf_audit",
            arguments=[
                "scripts/backfill_announcement_source_archives.py",
                "--announcement-plan",
                str(PDF_PLAN),
                "--rate-limit",
                "30",
                "--pdf-mode",
                "archive",
                "--fetch-backend",
                "cdp",
                "--limit",
                "100",
                "--state",
                str(PILOT_ROOT / "announcement_pdf_audit.state.json"),
                "--report",
                str(pdf_report),
            ],
            report_path=pdf_report,
            state=state,
        )

        market_state = PILOT_ROOT / "market_prices.state.json"
        _run_stateful_until_finished(
            name="sector_daily_prices",
            arguments=[
                "scripts/backfill_market_prices.py",
                "--universe-state",
                str(UNIVERSE),
                "--state",
                str(market_state),
            ],
            state_path=market_state,
            finished_key="finished_at",
            state=state,
        )

        target_report = VALIDATION_ROOT / "market_targets_train.report.json"
        _run_once(
            "sampled_train_labels",
            [
                "scripts/build_market_targets.py",
                "--announcement-plan",
                str(BODY_PLAN),
                "--output",
                str(target_report),
            ],
            state,
            artifact=target_report,
        )

        daily_basic_root = TABULAR_ROOT / "daily_basic"
        daily_basic_state = daily_basic_root / "state.json"
        _run_stateful_until_finished(
            name="sector_daily_basic",
            arguments=[
                "scripts/backfill_tabular_daily_basic.py",
                "--universe-state",
                str(UNIVERSE),
                "--output-root",
                str(daily_basic_root),
                "--state",
                str(daily_basic_state),
            ],
            state_path=daily_basic_state,
            finished_key="finished_at_utc",
            state=state,
        )

        pit_root = TABULAR_ROOT / "point_in_time_sources"
        for group, filename in (
            ("company_industry", "company_industry.state.json"),
            ("fundamentals", "fundamentals.state.json"),
        ):
            _run_stateful_until_finished(
                name=f"sector_point_in_time_{group}",
                arguments=[
                    "scripts/backfill_tabular_point_in_time_sources.py",
                    "--group",
                    group,
                    "--universe-state",
                    str(UNIVERSE),
                    "--output-root",
                    str(pit_root),
                ],
                state_path=pit_root / filename,
                finished_key="finished_at_utc",
                state=state,
            )

        state["finished_at"] = datetime.now(UTC).isoformat()
        state["status"] = (
            "completed_with_exceptions"
            if any(s.get("status") == "needs_review" for s in state["stages"].values())
            else "completed"
        )
        _save_state(state)
        print("electrical-equipment pilot data queue completed", flush=True)
    except Exception as error:
        state["status"] = "failed"
        state["error_type"] = type(error).__name__
        state["error"] = str(error)[:1000]
        _save_state(state)
        raise


if __name__ == "__main__":
    main()

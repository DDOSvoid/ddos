import json
import sqlite3
from datetime import date

from scripts.build_event_content_plans import build_pdf_plan


def test_pdf_plan_is_train_only_deterministic_and_bounded(tmp_path):
    database = tmp_path / "test.db"
    metadata = tmp_path / "metadata.json"
    classification_report = tmp_path / "classifications.json"
    output = tmp_path / "plan.json"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE announcements ("
            "id INTEGER PRIMARY KEY, announcement_id TEXT, published_date DATE);"
            "CREATE TABLE classifications ("
            "announcement_id INTEGER, sub_category TEXT, "
            "needs_review INTEGER, relevance TEXT);"
            "INSERT INTO announcements VALUES "
            "(1, 'A1', '2023-02-01'), (2, 'A2', '2023-03-01'), "
            "(3, 'A3', '2023-09-01'), (4, 'A4', '2024-03-01');"
            "INSERT INTO classifications VALUES "
            "(1, 'earnings', 0, 'core_event'), "
            "(2, 'earnings', 0, 'core_event'), "
            "(3, 'buyback', 0, 'core_event'), "
            "(4, 'other', 1, 'uncertain');"
        )
    metadata.write_text(
        json.dumps(
            {
                "progress": {"completed": 1, "total": 1},
                "finished_at": "2026-09-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    classification_report.write_text("{}", encoding="utf-8")
    plan = build_pdf_plan(
        database_path=database,
        metadata_state_path=metadata,
        classification_report_path=classification_report,
        output_path=output,
        train_start=date(2023, 1, 1),
        train_end=date(2024, 12, 31),
        per_stratum=1,
        maximum_pdfs=2,
    )
    assert len(plan["announcement_ids"]) == 2
    assert set(plan["announcement_ids"]) <= {"A1", "A2", "A3"}
    assert plan["selection"]["eligible_core_events"] == 3
    assert plan["official_test_queried"] is False
    assert plan["market_targets_queried"] is False

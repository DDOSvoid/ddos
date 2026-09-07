import json
import sqlite3

import yaml

from scripts.build_electrical_equipment_pilot import build_pilot
from src.prediction.data_download import load_train_universe_state


def test_pilot_selects_complete_train_only_date_cross_sections(tmp_path):
    database = tmp_path / "test.db"
    config = tmp_path / "pilot.yaml"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE companies (id INTEGER PRIMARY KEY, stock_code TEXT, industry TEXT);"
            "CREATE TABLE announcements ("
            "id INTEGER PRIMARY KEY, company_id INTEGER, announcement_id TEXT, "
            "published_date DATE);"
            "CREATE TABLE classifications ("
            "announcement_id INTEGER, major_category TEXT, sub_category TEXT, "
            "needs_review INTEGER, relevance TEXT);"
        )
        announcement_id = 1
        for company_id in range(1, 4):
            connection.execute(
                "INSERT INTO companies VALUES (?, ?, ?)",
                (company_id, f"00000{company_id}.SZ", "电气设备"),
            )
        connection.execute(
            "INSERT INTO companies VALUES (4, '600000.SH', '银行')"
        )
        for year in (2023, 2024):
            for month in range(1, 13):
                published_date = f"{year}-{month:02d}-15"
                for company_id in range(1, 4):
                    connection.execute(
                        "INSERT INTO announcements VALUES (?, ?, ?, ?)",
                        (
                            announcement_id,
                            company_id,
                            f"A{announcement_id}",
                            published_date,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO classifications VALUES (?, 'A', 'earnings', 0, 'core_event')",
                        (announcement_id,),
                    )
                    announcement_id += 1
                connection.execute(
                    "INSERT INTO announcements VALUES (?, 4, ?, ?)",
                    (announcement_id, f"A{announcement_id}", published_date),
                )
                connection.execute(
                    "INSERT INTO classifications VALUES (?, 'A', 'earnings', 0, 'core_event')",
                    (announcement_id,),
                )
                announcement_id += 1
    raw = {
        "contract": "causal-electrical-equipment-ranking-pilot-v1",
        "universe": {"industry_value": "电气设备"},
        "dates": {
            "train_start": "2023-01-01",
            "train_end": "2024-12-31",
            "official_test_start": "2025-01-01",
            "forward_validation_start": "2026-01-01",
        },
        "cross_section_sample": {
            "dates_per_quarter": 1,
            "minimum_companies_per_date": 3,
            "expected_selected_dates": 8,
            "date_selection": "deterministic_even_spacing_without_outcomes",
        },
        "pdf_audit": {"per_quarter_subcategory": 1, "maximum_pdfs": 8},
        "artifacts": {
            "universe": "out/universe.json",
            "body_plan": "out/body.json",
            "pdf_plan": "out/pdf.json",
            "manifest": "out/manifest.json",
        },
    }
    config.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    manifest = build_pilot(
        database_path=database,
        config_path=config,
        project_root=tmp_path,
    )
    body = json.loads((tmp_path / "out/body.json").read_text(encoding="utf-8"))
    assert body["selection"]["selected_dates"] == 8
    assert body["selection"]["selected_documents"] == 24
    assert body["selection"]["selected_companies"] == 3
    assert all(int(value[1:]) % 4 != 0 for value in body["announcement_ids"])
    assert load_train_universe_state(tmp_path / "out/universe.json") == [
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
    ]
    assert manifest["official_test_queried"] is False
    assert manifest["forward_validation_queried"] is False

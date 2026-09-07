#!/usr/bin/env python
"""Freeze the deterministic, train-only electrical-equipment ranking pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT

CONTRACT = "causal-electrical-equipment-ranking-pilot-v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _quarter(value: str) -> str:
    parsed = date.fromisoformat(value[:10])
    return f"{parsed.year}Q{((parsed.month - 1) // 3) + 1}"


def _evenly_spaced(values: list[str], count: int) -> list[str]:
    if len(values) < count:
        raise ValueError(f"only {len(values)} eligible dates are available for {count} slots")
    indexes = {
        min(len(values) - 1, math.floor((index + 0.5) * len(values) / count))
        for index in range(count)
    }
    if len(indexes) != count:
        raise ValueError("deterministic date selection produced duplicate indexes")
    return [values[index] for index in sorted(indexes)]


def _select_pdf_ids(
    rows: list[sqlite3.Row], *, per_stratum: int, maximum_pdfs: int
) -> list[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    all_ids: list[str] = []
    for row in rows:
        announcement_id = str(row["announcement_id"])
        key = f"{_quarter(str(row['published_date']))}:{row['sub_category']}"
        groups[key].append(announcement_id)
        all_ids.append(announcement_id)
    selected: set[str] = set()
    for key, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda value: _sha256_text(f"{key}:{value}"))
        selected.update(ordered[:per_stratum])
    if len(selected) < maximum_pdfs:
        remainder = sorted(
            (value for value in all_ids if value not in selected),
            key=lambda value: _sha256_text(f"pilot-pdf-fill:{value}"),
        )
        selected.update(remainder[: maximum_pdfs - len(selected)])
    return sorted(selected, key=lambda value: _sha256_text(f"pilot-pdf:{value}"))[
        :maximum_pdfs
    ]


def build_pilot(
    *, database_path: Path, config_path: Path, project_root: Path = PROJECT_ROOT
) -> dict[str, object]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if raw.get("contract") != CONTRACT:
        raise ValueError(f"unexpected pilot contract: {raw.get('contract')}")
    dates = raw.get("dates") or {}
    train_start = date.fromisoformat(str(dates["train_start"]))
    train_end = date.fromisoformat(str(dates["train_end"]))
    official_test_start = date.fromisoformat(str(dates["official_test_start"]))
    forward_start = date.fromisoformat(str(dates["forward_validation_start"]))
    if not train_end < official_test_start < forward_start:
        raise ValueError("pilot date boundaries do not preserve the sealed periods")

    universe_config = raw.get("universe") or {}
    industry = str(universe_config["industry_value"])
    sample = raw.get("cross_section_sample") or {}
    dates_per_quarter = int(sample["dates_per_quarter"])
    minimum_companies = int(sample["minimum_companies_per_date"])
    expected_dates = int(sample["expected_selected_dates"])
    pdf_config = raw.get("pdf_audit") or {}
    artifacts = raw.get("artifacts") or {}

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        base_params = (train_start.isoformat(), train_end.isoformat(), industry)
        date_rows = connection.execute(
            "SELECT a.published_date, COUNT(*) AS documents, "
            "COUNT(DISTINCT a.company_id) AS companies "
            "FROM announcements a "
            "JOIN classifications c ON c.announcement_id = a.id "
            "JOIN companies co ON co.id = a.company_id "
            "WHERE a.published_date BETWEEN ? AND ? "
            "AND co.industry = ? AND c.needs_review = 0 "
            "AND c.relevance = 'core_event' "
            "GROUP BY a.published_date ORDER BY a.published_date",
            base_params,
        ).fetchall()
        eligible_by_quarter: dict[str, list[str]] = defaultdict(list)
        date_counts: dict[str, dict[str, int]] = {}
        for row in date_rows:
            published_date = str(row["published_date"])
            date_counts[published_date] = {
                "documents": int(row["documents"]),
                "companies": int(row["companies"]),
            }
            if int(row["companies"]) >= minimum_companies:
                eligible_by_quarter[_quarter(published_date)].append(published_date)

        selected_by_quarter = {
            quarter: _evenly_spaced(values, dates_per_quarter)
            for quarter, values in sorted(eligible_by_quarter.items())
        }
        selected_dates = sorted(
            value for values in selected_by_quarter.values() for value in values
        )
        if len(selected_dates) != expected_dates:
            raise ValueError(
                f"expected {expected_dates} selected dates, found {len(selected_dates)}"
            )
        placeholders = ",".join("?" for _ in selected_dates)
        announcement_rows = connection.execute(
            "SELECT a.announcement_id, a.published_date, a.company_id, "
            "co.stock_code, c.major_category, c.sub_category "
            "FROM announcements a "
            "JOIN classifications c ON c.announcement_id = a.id "
            "JOIN companies co ON co.id = a.company_id "
            f"WHERE a.published_date IN ({placeholders}) "
            "AND co.industry = ? AND c.needs_review = 0 "
            "AND c.relevance = 'core_event' "
            "ORDER BY a.published_date, co.stock_code, a.id",
            (*selected_dates, industry),
        ).fetchall()
        universe_rows = connection.execute(
            "SELECT DISTINCT co.stock_code "
            "FROM announcements a "
            "JOIN classifications c ON c.announcement_id = a.id "
            "JOIN companies co ON co.id = a.company_id "
            "WHERE a.published_date BETWEEN ? AND ? "
            "AND co.industry = ? AND c.needs_review = 0 "
            "AND c.relevance = 'core_event' ORDER BY co.stock_code",
            base_params,
        ).fetchall()
        full_counts = connection.execute(
            "SELECT COUNT(*) AS documents, COUNT(DISTINCT a.company_id) AS companies, "
            "COUNT(DISTINCT a.company_id * 10000000 + "
            "CAST(julianday(a.published_date) AS INTEGER)) AS company_days "
            "FROM announcements a "
            "JOIN classifications c ON c.announcement_id = a.id "
            "JOIN companies co ON co.id = a.company_id "
            "WHERE a.published_date BETWEEN ? AND ? "
            "AND co.industry = ? AND c.needs_review = 0 "
            "AND c.relevance = 'core_event'",
            base_params,
        ).fetchone()

    announcement_ids = [str(row["announcement_id"]) for row in announcement_rows]
    if any(value in {"", "None"} for value in announcement_ids):
        raise ValueError("pilot contains an announcement without an external ID")
    if len(announcement_ids) != len(set(announcement_ids)):
        raise ValueError("pilot announcement IDs are not unique")
    codes = [str(row["stock_code"]) for row in universe_rows]
    company_days = len(
        {(str(row["published_date"]), str(row["stock_code"])) for row in announcement_rows}
    )
    selected_companies = len({str(row["stock_code"]) for row in announcement_rows})
    major_counts: dict[str, int] = defaultdict(int)
    for row in announcement_rows:
        major_counts[str(row["major_category"])] += 1

    now = datetime.now(UTC).isoformat()
    config_sha256 = _sha256_file(config_path)
    codes_sha256 = _sha256_text("\n".join(codes))
    ids_sha256 = _sha256_text("\n".join(sorted(announcement_ids)))
    dates_sha256 = _sha256_text("\n".join(selected_dates))
    body_plan: dict[str, object] = {
        "contract": "electrical-equipment-pilot-body-plan-v1",
        "pilot_contract": CONTRACT,
        "pilot_contract_sha256": config_sha256,
        "created_at": now,
        "dataset_role": "train",
        "official_test_queried": False,
        "market_targets_queried": False,
        "forward_validation_queried": False,
        "purpose": "complete_cross_sections_for_three_expert_fusion_feasibility",
        "selection": {
            "industry": industry,
            "start": train_start.isoformat(),
            "end": train_end.isoformat(),
            "date_selection": sample["date_selection"],
            "minimum_companies_per_date": minimum_companies,
            "dates_per_quarter": dates_per_quarter,
            "selected_dates": len(selected_dates),
            "selected_dates_sha256": dates_sha256,
            "selected_companies": selected_companies,
            "selected_company_days": company_days,
            "selected_documents": len(announcement_ids),
            "announcement_ids_sha256": ids_sha256,
            "major_category_counts": dict(sorted(major_counts.items())),
        },
        "selected_dates": selected_dates,
        "selected_dates_by_quarter": selected_by_quarter,
        "date_cross_section_counts": {
            value: date_counts[value] for value in selected_dates
        },
        "announcement_ids": announcement_ids,
    }

    pdf_ids = _select_pdf_ids(
        announcement_rows,
        per_stratum=int(pdf_config["per_quarter_subcategory"]),
        maximum_pdfs=int(pdf_config["maximum_pdfs"]),
    )
    pdf_plan: dict[str, object] = {
        "contract": "electrical-equipment-pilot-pdf-audit-v1",
        "pilot_contract": CONTRACT,
        "pilot_contract_sha256": config_sha256,
        "created_at": now,
        "dataset_role": "train",
        "official_test_queried": False,
        "market_targets_queried": False,
        "forward_validation_queried": False,
        "purpose": "deterministic_source_fidelity_audit_not_full_pdf_download",
        "selection": {
            "per_quarter_subcategory": int(pdf_config["per_quarter_subcategory"]),
            "maximum_pdfs": int(pdf_config["maximum_pdfs"]),
            "selected_pdfs": len(pdf_ids),
            "announcement_ids_sha256": _sha256_text("\n".join(sorted(pdf_ids))),
        },
        "announcement_ids": pdf_ids,
    }
    universe: dict[str, object] = {
        "contract": "electrical-equipment-pilot-train-universe-v1",
        "pilot_contract": CONTRACT,
        "pilot_contract_sha256": config_sha256,
        "created_at": now,
        "official_test_queried": False,
        "forward_validation_queried": False,
        "selection": {
            "dataset_role": "train",
            "start": train_start.isoformat(),
            "end": train_end.isoformat(),
            "industry": industry,
            "codes": len(codes),
            "codes_sha256": codes_sha256,
            "source": "accepted_core_event_sector_company_days",
        },
        "codes": codes,
        "announcement_count": int(full_counts["documents"]),
        "company_day_count": int(full_counts["company_days"]),
        "completed": {
            f"{code}:{train_start.isoformat()}:{train_end.isoformat()}": {
                "status": "eligible_train_sector_company"
            }
            for code in codes
        },
    }

    universe_path = project_root / str(artifacts["universe"])
    body_path = project_root / str(artifacts["body_plan"])
    pdf_path = project_root / str(artifacts["pdf_plan"])
    manifest_path = project_root / str(artifacts["manifest"])
    _write_json_atomic(universe_path, universe)
    _write_json_atomic(body_path, body_plan)
    _write_json_atomic(pdf_path, pdf_plan)
    manifest: dict[str, object] = {
        "contract": CONTRACT,
        "contract_sha256": config_sha256,
        "created_at": now,
        "dataset_role": "train",
        "official_test_queried": False,
        "market_targets_queried": False,
        "forward_validation_queried": False,
        "full_sector": {
            "industry": industry,
            "companies": int(full_counts["companies"]),
            "documents": int(full_counts["documents"]),
            "company_days": int(full_counts["company_days"]),
        },
        "pilot_sample": body_plan["selection"],
        "artifacts": {
            "universe": {"path": str(universe_path), "sha256": _sha256_file(universe_path)},
            "body_plan": {"path": str(body_path), "sha256": _sha256_file(body_path)},
            "pdf_plan": {"path": str(pdf_path), "sha256": _sha256_file(pdf_path)},
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "data" / "ddos.db")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "electrical_equipment_pilot_v1.yaml",
    )
    args = parser.parse_args()
    result = build_pilot(
        database_path=args.database.resolve(),
        config_path=args.config.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

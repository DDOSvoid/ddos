"""Read-only pilot source audit; never treats downloaded financials as PIT-approved."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prediction.development_contract import load_causal_development_contract
from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.contract import load_tabular_model_contract
from src.prediction.tabular.point_in_time_audit import audit_point_in_time_sources
from src.prediction.tabular.point_in_time_sources import (
    load_point_in_time_source_contract,
    load_train_universe,
    save_json_atomic,
)

PILOT = Path("data/backfills/event_ranking_pilot_electrical_v1")
TABULAR = Path("data/tabular/event_ranking_pilot_electrical_v1")


def check_daily_frame(frame, code):
    errors = []
    dates = pd.to_datetime(frame.trade_date, errors="coerce", utc=True)
    available = pd.to_datetime(frame.available_at_utc, errors="coerce", utc=True)
    if dates.isna().any() or not dates.between("2023-01-01", "2024-12-31").all():
        errors.append("invalid/out-of-train trade dates")
    if not (available == dates + pd.Timedelta(hours=10)).all():
        errors.append("daily availability is not 18:00 Shanghai")
    if set(frame.ts_code) != {code}:
        errors.append("stock identity mismatch")
    if frame.duplicated(["ts_code", "trade_date"]).any():
        errors.append("duplicate stock dates")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan_path = PILOT / "body_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["official_test_queried"] is False
    assert plan["forward_validation_queried"] is False
    codes = load_train_universe(PILOT / "train_universe.json")
    with sqlite3.connect("file:data/ddos.db?mode=ro", uri=True) as con:
        samples = pd.read_sql_query(
            "SELECT a.announcement_id,c.stock_code,a.published_date FROM announcements a "
            "JOIN companies c ON c.id=a.company_id "
            "WHERE a.published_date BETWEEN '2023-01-01' AND '2024-12-31'",
            con,
        )
        samples = samples[samples.announcement_id.isin(plan["announcement_ids"])]
        prices = pd.read_sql_query(
            "SELECT instrument_code AS ts_code,trade_date,close_price FROM daily_prices "
            "WHERE instrument_type='stock' AND trade_date BETWEEN '2023-01-01' AND '2024-12-31'",
            con,
        )
    assert set(samples.announcement_id) == set(plan["announcement_ids"])
    assert set(samples.stock_code) <= set(codes)
    samples.to_parquet(args.output / "audit_samples.parquet", index=False)
    daily_root = TABULAR / "daily_basic"
    state = json.loads((daily_root / "state.json").read_text(encoding="utf-8"))
    errors, empty, frames = [], [], []
    if state.get("failures") or not state.get("finished_at_utc"):
        errors.append("daily state incomplete")
    if state.get("official_test_queried") is not False:
        errors.append("daily isolation flag invalid")
    expected = {f"{code}:2023-01-01:2024-12-31" for code in codes}
    if set(state["completed"]) != expected:
        errors.append("daily completed keys differ from frozen universe")
    for key, item in state["completed"].items():
        code = key.split(":")[0]
        if item["status"] == "empty":
            empty.append(code)
            if item.get("rows") != 0 or item.get("path") is not None:
                errors.append(f"{code}: invalid empty record")
            continue
        try:
            path = (daily_root / item["path"]).resolve()
            path.relative_to(daily_root.resolve())
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
            frame = pd.read_parquet(path)
            assert len(frame) == item["rows"]
            errors.extend(f"{code}: {error}" for error in check_daily_frame(frame, code))
            frames.append(frame)
        except Exception as exc:
            errors.append(f"{code}: {type(exc).__name__}: {exc}")
    daily = pd.concat(frames, ignore_index=True)
    daily["trade_date"] = pd.to_datetime(daily.trade_date)
    prices = prices[prices.ts_code.isin(codes)].copy()
    prices["trade_date"] = pd.to_datetime(prices.trade_date)
    joined = daily.merge(prices, on=["ts_code", "trade_date"], how="outer", indicator=True)
    matched = joined[joined._merge == "both"]
    mismatches = int(((matched.close - matched.close_price).abs() > 0.011).sum())
    if mismatches:
        errors.append("daily_basic close differs from market prices")
    daily_report = dict(
        integrity_passed=not errors,
        errors=errors,
        rows=len(daily),
        empty_codes=empty,
        close_mismatches=mismatches,
        close_join_counts=joined._merge.value_counts().to_dict(),
        null_fractions=daily.isna().mean().to_dict(),
        note="availability is contractual metadata, not proof of historical vendor versions",
    )
    save_json_atomic(args.output / "daily_basic.audit.json", daily_report)
    split = load_prediction_split_contract()
    pit_root = TABULAR / "point_in_time_sources"
    contract = replace(
        load_point_in_time_source_contract(), universe_path=PILOT / "train_universe.json"
    )
    pit = audit_point_in_time_sources(
        root=pit_root,
        company_industry_state_path=pit_root / "company_industry.state.json",
        fundamentals_state_path=pit_root / "fundamentals.state.json",
        samples_path=args.output / "audit_samples.parquet",
        contract=contract,
        split=split,
        development=load_causal_development_contract(split=split),
        tabular=load_tabular_model_contract(split=split),
    )
    pit["effective_universe_override"] = str(contract.universe_path)
    pit["effective_universe_sha256"] = hashlib.sha256(
        contract.universe_path.read_bytes()
    ).hexdigest()
    save_json_atomic(args.output / "pit.audit.json", pit)
    report = dict(
        purpose="pilot_integrity_and_formal_expert_readiness",
        official_test_read=False,
        strict_oos_2026_read=False,
        samples=len(samples),
        universe_codes=len(codes),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        daily_basic_integrity_passed=daily_report["integrity_passed"],
        pit_raw_integrity_passed=pit["raw_integrity_passed"],
        pit_feature_release_eligible=pit["all_feature_groups_release_eligible"],
        formal_three_expert_training_authorized_by_report=False,
        financial_blockers=pit["fundamentals"]["release_blockers"],
    )
    save_json_atomic(args.output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

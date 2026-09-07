"""Prepare versioned train-only eligibility and disclosure-filtered source views.

These are NOT released model inputs until the independent expert gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prediction.tabular.point_in_time_sources import save_json_atomic

ROOT = Path("data/tabular/event_ranking_pilot_electrical_v1/point_in_time_sources")
AUDIT = Path("data/validation/event_ranking_pilot_electrical_v1/readiness_20260907_01")


def eligible_at_publication(published, listed):
    return pd.notna(listed) and pd.Timestamp(published) >= pd.Timestamp(listed)


def training_disclosure_mask(frame):
    disclosed = pd.to_datetime(frame.actual_disclosure_date, errors="coerce", utc=True)
    available = pd.to_datetime(frame.available_at_utc, errors="coerce", utc=True)
    return (
        disclosed.notna()
        & available.notna()
        & (disclosed < pd.Timestamp("2025-01-01", tz="UTC"))
        & (available < pd.Timestamp("2024-12-31T16:00:00Z"))
    )


def checked_frame(item, hashes):
    if item["status"] == "empty":
        return pd.DataFrame()
    path = (ROOT / item["path"]).resolve()
    path.relative_to(ROOT.resolve())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item["sha256"]:
        raise ValueError(f"source hash mismatch: {path}")
    hashes[item["path"]] = digest
    frame = pd.read_parquet(path)
    if len(frame) != item["rows"]:
        raise ValueError("source row count mismatch")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = json.loads((ROOT / "company_industry.state.json").read_text())
    hashes = {}
    stocks = checked_frame(state["completed"]["stock_basic:all_statuses"], hashes)
    listed = stocks.set_index("ts_code").list_date.to_dict()
    samples = pd.read_parquet(AUDIT / "audit_samples.parquet")
    memberships = {
        key.split(":", 1)[1]: checked_frame(item, hashes)
        for key, item in state["completed"].items()
        if key.startswith("industry:")
    }
    rows = []
    for row in samples[["stock_code", "published_date"]].drop_duplicates().itertuples():
        day = pd.Timestamp(row.published_date)
        eligible = eligible_at_publication(day, listed.get(row.stock_code))
        members = memberships[row.stock_code]
        active = members.loc[
            (pd.to_datetime(members.in_date) <= day)
            & (members.out_date.isna() | (day < pd.to_datetime(members.out_date)))
        ]
        identities = active[["l1_code", "l2_code", "l3_code"]].drop_duplicates()
        reason = (
            "pre_listing_or_listing_date_missing"
            if not eligible
            else ("historical_industry_unavailable" if len(identities) != 1 else "")
        )
        rows.append(
            dict(
                stock_code=row.stock_code,
                published_date=str(day.date()),
                list_date=str(listed.get(row.stock_code)),
                stock_eligible=bool(eligible),
                industry_available=len(identities) == 1,
                abstain_reason=reason,
            )
        )
    eligibility = pd.DataFrame(rows)
    eligibility.to_parquet(args.output / "company_day_eligibility.parquet", index=False)
    eligibility[eligibility.abstain_reason.ne("")].to_json(
        args.output / "eligibility_exceptions.json",
        orient="records",
        indent=2,
    )
    financial = json.loads((ROOT / "fundamentals.state.json").read_text())
    counts = {}
    for endpoint in ("income", "balancesheet", "cashflow"):
        frames = [
            checked_frame(item, hashes)
            for key, item in financial["completed"].items()
            if key.startswith(endpoint + ":") and item["status"] != "empty"
        ]
        frame = pd.concat(frames, ignore_index=True)
        keep = training_disclosure_mask(frame)
        frame.loc[keep].to_parquet(args.output / f"{endpoint}.train_candidate.parquet", index=False)
        frame.loc[
            ~keep,
            ["ts_code", "ann_date", "f_ann_date", "actual_disclosure_date", "available_at_utc"],
        ].to_parquet(
            args.output / f"{endpoint}.excluded_metadata.parquet",
            index=False,
        )
        counts[endpoint] = dict(retained=int(keep.sum()), excluded=int((~keep).sum()))
    report = dict(
        purpose="unreleased_pilot_source_views",
        model_input_release=False,
        company_days=len(eligibility),
        pre_listing_or_unknown=int((~eligibility.stock_eligible).sum()),
        eligible_industry_missing=int(
            (eligibility.stock_eligible & ~eligibility.industry_available).sum()
        ),
        financial_rows=counts,
        source_sha256=hashes,
        official_test_outcomes_read=False,
        strict_oos_2026_outcomes_read=False,
        per_sample_financial_join_required="available_at_utc strictly before prediction_as_of",
        raw_sources_modified=False,
    )
    save_json_atomic(args.output / "manifest.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()

"""Integrity, temporal-coverage, and release audit for PIT source archives."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.development_contract import CausalDevelopmentContract
from src.prediction.splits import PredictionSplitContract
from src.prediction.tabular.contract import TabularModelContract
from src.prediction.tabular.point_in_time_sources import (
    PIT_SOURCE_CONTRACT,
    PointInTimeSourceContract,
    load_train_universe,
    sha256_file,
)

PIT_AUDIT_CONTRACT = "tabular-point-in-time-source-audit-v1"


def _resolve_inside(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"point-in-time path escapes root: {relative}") from exc
    return path


def _validate_state(
    state: dict,
    *,
    group: str,
    contract: PointInTimeSourceContract,
    split: PredictionSplitContract,
    development: CausalDevelopmentContract,
    tabular: TabularModelContract,
) -> list[str]:
    errors = []
    expected = {
        "contract": PIT_SOURCE_CONTRACT,
        "source_contract_sha256": contract.source_sha256,
        "split_contract": split.contract,
        "split_source_sha256": split.source_sha256,
        "development_contract": development.contract,
        "development_source_sha256": development.source_sha256,
        "tabular_contract": tabular.contract,
        "tabular_source_sha256": tabular.source_sha256,
        "dataset_role": "train",
        "official_test_queried": False,
    }
    for name, value in expected.items():
        if state.get(name) != value:
            errors.append(f"{group} state {name} mismatch")
    if state.get("selection", {}).get("source_group") != group:
        errors.append(f"{group} state selection mismatch")
    if state.get("failures"):
        errors.append(f"{group} state contains failures")
    if not state.get("finished_at_utc"):
        errors.append(f"{group} state is not marked finished")
    return errors


def _load_completed_frame(
    root: Path,
    *,
    key: str,
    item: dict,
    required: set[str],
    errors: list[str],
) -> pd.DataFrame:
    if item.get("status") == "empty":
        if item.get("rows") != 0 or item.get("path") is not None:
            errors.append(f"{key}: invalid empty record")
        return pd.DataFrame()
    if item.get("status") != "downloaded":
        errors.append(f"{key}: invalid completion status")
        return pd.DataFrame()
    try:
        path = _resolve_inside(root, str(item["path"]))
        if sha256_file(path) != item.get("sha256"):
            raise ValueError("SHA-256 mismatch")
        frame = pd.read_parquet(path)
        if len(frame) != item.get("rows"):
            raise ValueError("row count mismatch")
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"missing columns {sorted(missing)}")
        return frame
    except Exception as exc:
        errors.append(f"{key}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def _industry_sample_coverage(
    memberships: pd.DataFrame, samples: pd.DataFrame
) -> dict:
    company_days = samples.loc[:, ["stock_code", "published_date"]].drop_duplicates()
    company_days["published_date"] = pd.to_datetime(
        company_days["published_date"], errors="raise"
    ).dt.date
    missing = 0
    ambiguous = 0
    covered = 0
    ambiguous_codes: set[str] = set()
    missing_codes: set[str] = set()
    for stock_code, dates in company_days.groupby("stock_code", sort=False):
        rows = memberships.loc[memberships["ts_code"] == stock_code]
        for published_date in dates["published_date"]:
            active = rows.loc[
                (rows["in_date"].notna())
                & (rows["in_date"] <= published_date)
                & (rows["out_date"].isna() | (published_date < rows["out_date"]))
            ]
            identities = active.loc[
                :, ["l1_code", "l2_code", "l3_code"]
            ].drop_duplicates()
            if identities.empty:
                missing += 1
                missing_codes.add(str(stock_code))
            elif len(identities) > 1:
                ambiguous += 1
                ambiguous_codes.add(str(stock_code))
            else:
                covered += 1
    total = len(company_days)
    return {
        "company_days": total,
        "unambiguous_company_days": covered,
        "missing_company_days": missing,
        "ambiguous_company_days": ambiguous,
        "unambiguous_fraction": covered / total if total else 0.0,
        "missing_codes": sorted(missing_codes),
        "ambiguous_codes": sorted(ambiguous_codes),
    }


def _audit_company_industry(
    *,
    root: Path,
    state: dict,
    contract: PointInTimeSourceContract,
    samples: pd.DataFrame,
    universe_codes: list[str],
) -> tuple[dict, list[str]]:
    errors: list[str] = []
    completed = state.get("completed", {})
    stock_item = completed.get("stock_basic:all_statuses")
    if stock_item is None:
        errors.append("stock_basic artifact is missing")
        stock_basic = pd.DataFrame()
    else:
        stock_basic = _load_completed_frame(
            root,
            key="stock_basic:all_statuses",
            item=stock_item,
            required={
                *contract.stock_basic_fields,
                "source",
                "retrieved_at_utc",
                "historical_release_status",
            },
            errors=errors,
        )
    membership_frames = []
    empty_partitions = 0
    for key, item in sorted(completed.items()):
        if not key.startswith("industry:"):
            continue
        if item.get("status") == "empty":
            empty_partitions += 1
        frame = _load_completed_frame(
            root,
            key=key,
            item=item,
            required={
                *contract.industry_fields,
                "query_is_new",
                "source",
                "retrieved_at_utc",
            },
            errors=errors,
        )
        if not frame.empty:
            membership_frames.append(frame)
    memberships = (
        pd.concat(membership_frames, ignore_index=True)
        if membership_frames
        else pd.DataFrame(columns=contract.industry_fields)
    )
    for column in ("in_date", "out_date"):
        memberships[column] = pd.to_datetime(
            memberships[column], errors="coerce"
        ).dt.date
    invalid_intervals = int(
        (
            memberships["in_date"].notna()
            & memberships["out_date"].notna()
            & (memberships["out_date"] < memberships["in_date"])
        ).sum()
    )
    if invalid_intervals:
        errors.append("industry contains out_date before in_date")
    coverage = _industry_sample_coverage(memberships, samples)
    requested_codes = int(state.get("selection", {}).get("codes", 0))
    universe = set(universe_codes)
    expected_hash = hashlib.sha256("\n".join(universe_codes).encode()).hexdigest()
    if requested_codes != len(universe_codes):
        errors.append("company/industry requested code count differs from universe")
    if state.get("selection", {}).get("codes_sha256") != expected_hash:
        errors.append("company/industry code hash differs from universe")
    expected_completion_keys = {"stock_basic:all_statuses"} | {
        f"industry:{stock_code}" for stock_code in universe_codes
    }
    if set(completed) != expected_completion_keys:
        errors.append("company/industry completion keys differ from universe")
    stock_codes = set(stock_basic.get("ts_code", pd.Series(dtype=str)).astype(str))
    membership_codes = set(memberships.get("ts_code", pd.Series(dtype=str)).astype(str))
    if stock_codes != universe:
        errors.append("stock_basic covered codes differ from universe")
    raw_integrity_passed = not errors
    release_eligible = (
        raw_integrity_passed
        and coverage["missing_company_days"] == 0
        and coverage["ambiguous_company_days"] == 0
    )
    report = {
        "raw_integrity_passed": raw_integrity_passed,
        "feature_release_eligible": release_eligible,
        "release_blockers": [
            name
            for name, blocked in (
                ("missing_train_date_membership", coverage["missing_company_days"] > 0),
                (
                    "overlapping_train_date_membership",
                    coverage["ambiguous_company_days"] > 0,
                ),
                ("raw_integrity_error", bool(errors)),
            )
            if blocked
        ],
        "requested_codes": requested_codes,
        "stock_basic_rows": len(stock_basic),
        "stock_basic_covered_codes": len(stock_codes),
        "industry_rows": len(memberships),
        "industry_covered_codes": len(membership_codes),
        "industry_empty_partitions": empty_partitions,
        "industry_current_rows": int((memberships.get("is_new") == "Y").sum()),
        "industry_historical_rows": int((memberships.get("is_new") == "N").sum()),
        "invalid_intervals": invalid_intervals,
        "sample_coverage": coverage,
    }
    return report, errors


def _metric_signature(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    canonical = frame.loc[:, columns].astype("string").fillna("<NA>").agg("|".join, axis=1)
    return canonical.map(lambda value: hashlib.sha256(value.encode()).hexdigest())


def _financial_sample_coverage(
    frames: dict[str, pd.DataFrame], samples: pd.DataFrame
) -> dict:
    company_days = samples.loc[:, ["stock_code", "published_date"]].drop_duplicates()
    company_days["published_date"] = pd.to_datetime(
        company_days["published_date"], errors="raise"
    ).dt.date
    endpoint_available: dict[str, set[tuple[str, object]]] = {}
    for endpoint, frame in frames.items():
        available: set[tuple[str, object]] = set()
        for stock_code, dates in company_days.groupby("stock_code", sort=False):
            disclosures = np.array(
                sorted(
                    frame.loc[
                        frame["ts_code"] == stock_code, "actual_disclosure_date"
                    ].dropna()
                ),
                dtype="datetime64[D]",
            )
            if not len(disclosures):
                continue
            published = np.array(dates["published_date"], dtype="datetime64[D]")
            mask = np.searchsorted(disclosures, published, side="left") > 0
            available.update(
                (str(stock_code), value)
                for value in dates.loc[mask, "published_date"]
            )
        endpoint_available[endpoint] = available
    all_keys = set(
        zip(company_days["stock_code"].astype(str), company_days["published_date"], strict=True)
    )
    all_three = set.intersection(*endpoint_available.values()) if endpoint_available else set()
    missing_keys = all_keys - all_three
    return {
        "company_days": len(company_days),
        "available_company_days_by_endpoint": {
            endpoint: len(values) for endpoint, values in endpoint_available.items()
        },
        "all_three_available_company_days": len(all_three),
        "all_three_missing_company_days": len(missing_keys),
        "all_three_available_fraction": len(all_three) / len(all_keys) if all_keys else 0.0,
        "all_three_missing_codes": sorted({stock_code for stock_code, _ in missing_keys}),
    }


def _audit_fundamentals(
    *,
    root: Path,
    state: dict,
    contract: PointInTimeSourceContract,
    samples: pd.DataFrame,
    universe_codes: list[str],
) -> tuple[dict, list[str]]:
    errors: list[str] = []
    completed = state.get("completed", {})
    frames_by_endpoint: dict[str, list[pd.DataFrame]] = {
        endpoint: [] for endpoint in contract.financial_fields
    }
    empty_by_endpoint = {endpoint: 0 for endpoint in contract.financial_fields}
    for key, item in sorted(completed.items()):
        endpoint, _, stock_code = key.partition(":")
        if endpoint not in contract.financial_fields or not stock_code:
            errors.append(f"unexpected financial completion key: {key}")
            continue
        if item.get("status") == "empty":
            empty_by_endpoint[endpoint] += 1
        frame = _load_completed_frame(
            root,
            key=key,
            item=item,
            required={
                *contract.financial_fields[endpoint],
                "actual_disclosure_date",
                "availability_date_source",
                "available_at_utc",
                "source",
                "retrieved_at_utc",
                "dataset_role",
            },
            errors=errors,
        )
        if not frame.empty:
            if set(frame["ts_code"].astype(str).unique()) != {stock_code}:
                errors.append(f"{key}: stock code mismatch")
            frames_by_endpoint[endpoint].append(frame)
    combined = {
        endpoint: (
            pd.concat(values, ignore_index=True)
            if values
            else pd.DataFrame(columns=contract.financial_fields[endpoint])
        )
        for endpoint, values in frames_by_endpoint.items()
    }
    conflicts = {}
    revision_groups = {}
    fallback_rows = {}
    post_query_window_disclosures = {}
    rows = {}
    covered_codes = {}
    for endpoint, frame in combined.items():
        if frame.empty:
            conflicts[endpoint] = 0
            revision_groups[endpoint] = 0
            fallback_rows[endpoint] = 0
            post_query_window_disclosures[endpoint] = 0
            rows[endpoint] = 0
            covered_codes[endpoint] = 0
            continue
        for column in ("ann_date", "f_ann_date", "end_date", "actual_disclosure_date"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
        available = pd.to_datetime(frame["available_at_utc"], errors="coerce", utc=True)
        if available.isna().any():
            errors.append(f"{endpoint}: invalid available_at_utc")
        if frame["actual_disclosure_date"].isna().any():
            errors.append(f"{endpoint}: missing actual disclosure date")
        if (
            frame["ann_date"].min() < contract.financial_start
            or frame["ann_date"].max() > contract.financial_end
        ):
            errors.append(f"{endpoint}: crosses announcement query window")
        identity = list(contract.financial_identity_fields)
        metric_columns = [
            name
            for name in contract.financial_fields[endpoint]
            if name not in set(identity)
        ]
        working = frame.copy()
        working["__metric_signature"] = _metric_signature(working, metric_columns)
        conflict_count = int(
            (working.groupby(identity, dropna=False)["__metric_signature"].nunique() > 1).sum()
        )
        conflicts[endpoint] = conflict_count
        revision_groups[endpoint] = int(
            (
                working.groupby(
                    ["ts_code", "end_date", "report_type", "comp_type", "end_type"],
                    dropna=False,
                )["actual_disclosure_date"].nunique()
                > 1
            ).sum()
        )
        fallback_rows[endpoint] = int(
            (frame["availability_date_source"] == "ann_date").sum()
        )
        post_query_window_disclosures[endpoint] = int(
            (frame["actual_disclosure_date"] > contract.financial_end).sum()
        )
        rows[endpoint] = len(frame)
        covered_codes[endpoint] = int(frame["ts_code"].nunique())
        if conflict_count:
            errors.append(f"{endpoint}: conflicting values share one PIT identity")
    requested = int(state.get("selection", {}).get("codes", 0))
    expected_hash = hashlib.sha256("\n".join(universe_codes).encode()).hexdigest()
    if requested != len(universe_codes):
        errors.append("fundamental requested code count differs from universe")
    if state.get("selection", {}).get("codes_sha256") != expected_hash:
        errors.append("fundamental code hash differs from universe")
    expected_completion_keys = {
        f"{endpoint}:{stock_code}"
        for endpoint in contract.financial_fields
        for stock_code in universe_codes
    }
    if set(completed) != expected_completion_keys:
        errors.append("fundamental completion keys differ from universe")
    raw_integrity_passed = not errors
    coverage = _financial_sample_coverage(combined, samples)
    revision_semantics_status = "requires_provider_snapshot_semantics_acceptance"
    report = {
        "raw_integrity_passed": raw_integrity_passed,
        "availability_metadata_passed": not any(
            "available_at" in error or "disclosure" in error for error in errors
        ),
        "revision_identity_conflicts_passed": sum(conflicts.values()) == 0,
        "feature_release_eligible": False,
        "release_blockers": [revision_semantics_status]
        + (["raw_integrity_error"] if errors else []),
        "revision_semantics_status": revision_semantics_status,
        "requested_codes": requested,
        "rows_by_endpoint": rows,
        "covered_codes_by_endpoint": covered_codes,
        "empty_partitions_by_endpoint": empty_by_endpoint,
        "ann_date_fallback_rows_by_endpoint": fallback_rows,
        "post_query_window_disclosures_by_endpoint": post_query_window_disclosures,
        "revision_groups_by_endpoint": revision_groups,
        "conflicting_identity_groups_by_endpoint": conflicts,
        "sample_coverage": coverage,
    }
    return report, errors


def audit_point_in_time_sources(
    *,
    root: Path,
    company_industry_state_path: Path,
    fundamentals_state_path: Path,
    samples_path: Path,
    contract: PointInTimeSourceContract,
    split: PredictionSplitContract,
    development: CausalDevelopmentContract,
    tabular: TabularModelContract,
) -> dict:
    company_state = json.loads(
        company_industry_state_path.read_text(encoding="utf-8")
    )
    financial_state = json.loads(fundamentals_state_path.read_text(encoding="utf-8"))
    samples = pd.read_parquet(samples_path, columns=["stock_code", "published_date"])
    universe_codes = load_train_universe(contract.universe_path)
    if not set(samples["stock_code"].astype(str)).issubset(universe_codes):
        raise ValueError("audit samples contain stock codes outside the train universe")
    company_state_errors = _validate_state(
        company_state,
        group="company_industry",
        contract=contract,
        split=split,
        development=development,
        tabular=tabular,
    )
    financial_state_errors = _validate_state(
        financial_state,
        group="fundamentals",
        contract=contract,
        split=split,
        development=development,
        tabular=tabular,
    )
    company_report, company_errors = _audit_company_industry(
        root=root,
        state=company_state,
        contract=contract,
        samples=samples,
        universe_codes=universe_codes,
    )
    financial_report, financial_errors = _audit_fundamentals(
        root=root,
        state=financial_state,
        contract=contract,
        samples=samples,
        universe_codes=universe_codes,
    )
    errors = {
        "company_industry": [*company_state_errors, *company_errors],
        "fundamentals": [*financial_state_errors, *financial_errors],
    }
    raw_integrity_passed = (
        not errors["company_industry"]
        and not errors["fundamentals"]
        and company_report["raw_integrity_passed"]
        and financial_report["raw_integrity_passed"]
    )
    return {
        "contract": PIT_AUDIT_CONTRACT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_role": "train",
        "official_test_queried": False,
        "raw_integrity_passed": raw_integrity_passed,
        "all_feature_groups_release_eligible": (
            company_report["feature_release_eligible"]
            and financial_report["feature_release_eligible"]
        ),
        "source_contract_sha256": contract.source_sha256,
        "split_source_sha256": split.source_sha256,
        "development_source_sha256": development.source_sha256,
        "tabular_source_sha256": tabular.source_sha256,
        "samples_sha256": sha256_file(samples_path),
        "universe_codes": len(universe_codes),
        "company_industry_state_sha256": sha256_file(company_industry_state_path),
        "fundamentals_state_sha256": sha256_file(fundamentals_state_path),
        "company_industry": company_report,
        "fundamentals": financial_report,
        "errors": errors,
    }

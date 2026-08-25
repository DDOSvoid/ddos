"""Resumable legacy-industry archive and train-date coverage audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from src.config import PROJECT_ROOT
from src.prediction.development_contract import CausalDevelopmentContract
from src.prediction.splits import PredictionSplitContract
from src.prediction.tabular.contract import TabularModelContract
from src.prediction.tabular.point_in_time_sources import (
    TusharePointInTimeClient,
    _fetch_with_retries,
    _record_frame,
    load_train_universe,
    save_json_atomic,
    sha256_file,
)

INDUSTRY_SUPPLEMENT_CONFIG_CONTRACT = "tabular-industry-supplement-v1"
INDUSTRY_SUPPLEMENT_BACKFILL_CONTRACT = "tabular-industry-supplement-backfill-v1"
INDUSTRY_SUPPLEMENT_AUDIT_CONTRACT = "tabular-industry-supplement-audit-v1"
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "tabular_industry_supplement_v1.yaml"
)


@dataclass(frozen=True)
class IndustrySupplementContract:
    universe_path: Path
    classification_sources: tuple[str, ...]
    classification_levels: tuple[str, ...]
    classification_fields: tuple[str, ...]
    membership_fields: tuple[str, ...]
    membership_all_fields: tuple[str, ...]
    source_path: Path
    source_sha256: str


def load_industry_supplement_contract(
    path: Path = DEFAULT_CONFIG_PATH,
) -> IndustrySupplementContract:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    if raw.get("contract") != INDUSTRY_SUPPLEMENT_CONFIG_CONTRACT:
        raise ValueError("unexpected industry supplement contract")
    if raw.get("dataset_role") != "train" or raw.get("official_test_allowed") is not False:
        raise PermissionError("industry supplement must remain train-only")
    if raw["membership"].get("interval_semantics") != "[in_date, out_date)":
        raise ValueError("unexpected industry interval semantics")
    policy = raw["release_policy"]
    if policy.get("current_snapshot_backfill_forbidden") is not True:
        raise ValueError("current industry snapshot backfill must stay forbidden")
    if policy.get("allow_explicit_taxonomy_not_yet_effective_state") is not True:
        raise ValueError("taxonomy not-yet-effective state must be explicit")
    return IndustrySupplementContract(
        universe_path=(PROJECT_ROOT / raw["universe"]["source"]).resolve(),
        classification_sources=tuple(str(value) for value in raw["classification"]["sources"]),
        classification_levels=tuple(str(value) for value in raw["classification"]["levels"]),
        classification_fields=tuple(str(value) for value in raw["classification"]["fields"]),
        membership_fields=tuple(str(value) for value in raw["membership"]["fields"]),
        membership_all_fields=tuple(
            str(value) for value in raw["membership_all"]["fields"]
        ),
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _normalize_classification(
    frame: pd.DataFrame,
    *,
    source: str,
    level: str,
    fields: tuple[str, ...],
    fetched_at: datetime,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"empty classification response for {source}/{level}")
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ValueError(f"classification response missing columns: {sorted(missing)}")
    result = frame.loc[:, fields].copy()
    if not result["src"].eq(source).all() or not result["level"].eq(level).all():
        raise ValueError("classification response identity mismatch")
    if result["index_code"].isna().any() or result["index_code"].duplicated().any():
        raise ValueError("classification response has invalid index codes")
    result["source"] = "tushare_index_classify"
    result["retrieved_at_utc"] = pd.Timestamp(fetched_at).tz_convert("UTC")
    result["dataset_role"] = "train"
    return result.sort_values("index_code", kind="mergesort").reset_index(drop=True)


def _normalize_membership(
    frame: pd.DataFrame,
    *,
    stock_code: str,
    fields: tuple[str, ...],
    fetched_at: datetime,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=fields)
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ValueError(f"index_member response missing columns: {sorted(missing)}")
    result = frame.loc[:, fields].copy()
    if set(result["con_code"].astype(str).unique()) != {stock_code}:
        raise ValueError("index_member response contains another stock code")
    for column in ("in_date", "out_date"):
        result[column] = pd.to_datetime(
            result[column], format="%Y%m%d", errors="coerce"
        ).dt.date
    if result["in_date"].isna().any():
        raise ValueError("index_member response contains no usable in_date")
    invalid = result["out_date"].notna() & (result["out_date"] < result["in_date"])
    if invalid.any():
        raise ValueError("index_member response contains an invalid interval")
    result["query_ts_code"] = stock_code
    result["source"] = "tushare_index_member"
    result["retrieved_at_utc"] = pd.Timestamp(fetched_at).tz_convert("UTC")
    result["dataset_role"] = "train"
    return result.sort_values(
        ["in_date", "out_date", "index_code"], kind="mergesort"
    ).reset_index(drop=True)


def _normalize_membership_all(
    frame: pd.DataFrame,
    *,
    stock_code: str,
    fields: tuple[str, ...],
    fetched_at: datetime,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["index_code", "con_code", "in_date", "out_date", "is_new"]
        )
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ValueError(f"index_member_all response missing columns: {sorted(missing)}")
    if set(frame["ts_code"].astype(str).unique()) != {stock_code}:
        raise ValueError("index_member_all response contains another stock code")
    result = frame.loc[:, fields].copy()
    for column in ("in_date", "out_date"):
        result[column] = pd.to_datetime(
            result[column], format="%Y%m%d", errors="coerce"
        ).dt.date
    if result["in_date"].isna().any():
        raise ValueError("index_member_all response contains no usable in_date")
    invalid = result["out_date"].notna() & (result["out_date"] < result["in_date"])
    if invalid.any():
        raise ValueError("index_member_all response contains an invalid interval")
    normalized = result.rename(
        columns={"l1_code": "index_code", "ts_code": "con_code"}
    )[["index_code", "con_code", "in_date", "out_date", "is_new"]]
    normalized["query_ts_code"] = stock_code
    normalized["source"] = "tushare_index_member_all"
    normalized["retrieved_at_utc"] = pd.Timestamp(fetched_at).tz_convert("UTC")
    normalized["dataset_role"] = "train"
    return normalized.sort_values(
        ["in_date", "out_date", "index_code"], kind="mergesort"
    ).reset_index(drop=True)


def download_industry_supplement(
    *,
    codes: list[str],
    output_root: Path,
    state_path: Path,
    client: TusharePointInTimeClient,
    contract: IndustrySupplementContract,
    split: PredictionSplitContract,
    development: CausalDevelopmentContract,
    tabular: TabularModelContract,
) -> dict:
    selected_codes = sorted(set(codes))
    selection = {
        "dataset_role": "train",
        "codes": len(selected_codes),
        "codes_sha256": hashlib.sha256("\n".join(selected_codes).encode()).hexdigest(),
        "classification_sources": list(contract.classification_sources),
        "classification_levels": list(contract.classification_levels),
    }
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if state.get("selection") not in (None, selection):
        raise ValueError("industry supplement state selection differs from this run")
    state.setdefault("started_at_utc", datetime.now(UTC).isoformat())
    state.update(
        {
            "contract": INDUSTRY_SUPPLEMENT_BACKFILL_CONTRACT,
            "source_contract": INDUSTRY_SUPPLEMENT_CONFIG_CONTRACT,
            "source_contract_sha256": contract.source_sha256,
            "split_contract": split.contract,
            "split_source_sha256": split.source_sha256,
            "development_contract": development.contract,
            "development_source_sha256": development.source_sha256,
            "tabular_contract": tabular.contract,
            "tabular_source_sha256": tabular.source_sha256,
            "dataset_role": "train",
            "official_test_queried": False,
            "selection": selection,
        }
    )
    completed = dict(state.get("completed", {}))
    failures = dict(state.get("failures", {}))
    classification_work = [
        (source, level)
        for source in contract.classification_sources
        for level in contract.classification_levels
    ]
    work = [("classification", item) for item in classification_work] + [
        (kind, stock_code)
        for kind in ("membership", "membership_all")
        for stock_code in selected_codes
    ]
    for index, (kind, item) in enumerate(work, 1):
        if kind == "classification":
            source, level = item
            key = f"classification:{source}:{level}"
        else:
            source = level = None
            stock_code = str(item)
            key = f"{kind}:{stock_code}"
        if key in completed:
            continue
        try:
            fetched_at = datetime.now(UTC)
            if kind == "classification":
                raw = _fetch_with_retries(
                    client,
                    "index_classify",
                    maximum_attempts=4,
                    src=source,
                    level=level,
                    fields=",".join(contract.classification_fields),
                )
                normalized = _normalize_classification(
                    raw,
                    source=str(source),
                    level=str(level),
                    fields=contract.classification_fields,
                    fetched_at=fetched_at,
                )
                relative = Path("raw") / "classification" / f"{source}_{level}.parquet"
            elif kind == "membership":
                raw = _fetch_with_retries(
                    client,
                    "index_member",
                    maximum_attempts=4,
                    ts_code=stock_code,
                    fields=",".join(contract.membership_fields),
                )
                normalized = _normalize_membership(
                    raw,
                    stock_code=stock_code,
                    fields=contract.membership_fields,
                    fetched_at=fetched_at,
                )
                relative = Path("raw") / "membership" / f"{stock_code}.parquet"
            else:
                raw = _fetch_with_retries(
                    client,
                    "index_member_all",
                    maximum_attempts=4,
                    ts_code=stock_code,
                    fields=",".join(contract.membership_all_fields),
                )
                normalized = _normalize_membership_all(
                    raw,
                    stock_code=stock_code,
                    fields=contract.membership_all_fields,
                    fetched_at=fetched_at,
                )
                relative = (
                    Path("raw") / "membership_all" / f"{stock_code}.parquet"
                )
            completed[key] = _record_frame(
                frame=normalized,
                target=output_root / relative,
                relative=relative,
            )
            failures.pop(key, None)
        except Exception as exc:  # pragma: no cover - live service behavior
            failures[key] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "updated_at_utc": datetime.now(UTC).isoformat(),
            }
        state.update(
            {
                "completed": completed,
                "failures": failures,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        save_json_atomic(state_path, state)
        print(
            f"industry supplement {index}/{len(work)}; "
            f"completed={len(completed)}; failures={len(failures)}",
            flush=True,
        )
    if not failures and len(completed) == len(work):
        state.setdefault("finished_at_utc", datetime.now(UTC).isoformat())
        save_json_atomic(state_path, state)
    return state


def _load_recorded_frame(
    *, root: Path, item: dict, required: set[str], key: str, errors: list[str]
) -> pd.DataFrame:
    if item.get("status") == "empty":
        if item.get("rows") != 0 or item.get("path") is not None:
            errors.append(f"{key}: invalid empty completion")
        return pd.DataFrame()
    try:
        path = (root / str(item["path"])).resolve()
        path.relative_to(root.resolve())
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


def _active_identities(frame: pd.DataFrame, day: object, identity: str) -> set[str]:
    if frame.empty or identity not in frame.columns or "in_date" not in frame.columns:
        return set()
    day = pd.Timestamp(day).date()
    in_dates = pd.Series(
        [
            value.date() if pd.notna(value) else None
            for value in pd.to_datetime(frame["in_date"], errors="coerce")
        ],
        index=frame.index,
        dtype="object",
    )
    out_dates = pd.Series(
        [
            value.date() if pd.notna(value) else None
            for value in pd.to_datetime(frame["out_date"], errors="coerce")
        ],
        index=frame.index,
        dtype="object",
    )
    active = frame.loc[
        in_dates.notna()
        & (in_dates <= day)
        & (out_dates.isna() | (day < out_dates))
    ]
    return set(active[identity].dropna().astype(str))


def audit_industry_supplement(
    *,
    root: Path,
    state_path: Path,
    existing_root: Path,
    samples_path: Path,
    contract: IndustrySupplementContract,
    split: PredictionSplitContract,
    development: CausalDevelopmentContract,
    tabular: TabularModelContract,
) -> dict:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    codes = load_train_universe(contract.universe_path)
    expected_hash = hashlib.sha256("\n".join(codes).encode()).hexdigest()
    errors: list[str] = []
    expected_state = {
        "contract": INDUSTRY_SUPPLEMENT_BACKFILL_CONTRACT,
        "source_contract_sha256": contract.source_sha256,
        "split_source_sha256": split.source_sha256,
        "development_source_sha256": development.source_sha256,
        "tabular_source_sha256": tabular.source_sha256,
        "dataset_role": "train",
        "official_test_queried": False,
    }
    for name, value in expected_state.items():
        if state.get(name) != value:
            errors.append(f"state {name} mismatch")
    if state.get("selection", {}).get("codes_sha256") != expected_hash:
        errors.append("state universe hash mismatch")
    if state.get("failures"):
        errors.append("state contains failures")
    if not state.get("finished_at_utc"):
        errors.append("state is not marked finished")
    completed = state.get("completed", {})
    expected_keys = {
        f"classification:{source}:{level}"
        for source in contract.classification_sources
        for level in contract.classification_levels
    } | {
        f"{kind}:{stock_code}"
        for kind in ("membership", "membership_all")
        for stock_code in codes
    }
    if set(completed) != expected_keys:
        errors.append("completion keys differ from the train universe")

    classifications = []
    memberships = []
    memberships_all = []
    for key, item in sorted(completed.items()):
        required = (
            {*contract.classification_fields, "source", "retrieved_at_utc", "dataset_role"}
            if key.startswith("classification:")
            else {
                *contract.membership_fields,
                "query_ts_code",
                "source",
                "retrieved_at_utc",
                "dataset_role",
            }
        )
        frame = _load_recorded_frame(
            root=root, item=item, required=required, key=key, errors=errors
        )
        if frame.empty:
            continue
        if key.startswith("classification:"):
            classifications.append(frame)
        elif key.startswith("membership_all:"):
            memberships_all.append(frame)
        else:
            memberships.append(frame)
    classification = pd.concat(classifications, ignore_index=True)
    membership = pd.concat(memberships, ignore_index=True) if memberships else pd.DataFrame()
    membership_all = (
        pd.concat(memberships_all, ignore_index=True)
        if memberships_all
        else pd.DataFrame()
    )
    l1_codes = set(
        classification.loc[classification["level"].eq("L1"), "index_code"].astype(str)
    )
    membership = membership.loc[membership["index_code"].astype(str).isin(l1_codes)].copy()
    membership_all = membership_all.loc[
        membership_all["index_code"].astype(str).isin(l1_codes)
    ].copy()
    for column in ("in_date", "out_date"):
        membership[column] = pd.to_datetime(membership[column], errors="coerce").dt.date
        membership_all[column] = pd.to_datetime(
            membership_all[column], errors="coerce"
        ).dt.date

    existing_frames = [
        pd.read_parquet(path)
        for path in (existing_root / "raw" / "company_industry" / "sw_membership").glob("*.parquet")
    ]
    existing = pd.concat(existing_frames, ignore_index=True)
    for column in ("in_date", "out_date"):
        existing[column] = pd.to_datetime(existing[column], errors="coerce").dt.date
    samples = pd.read_parquet(samples_path, columns=["stock_code", "published_date"])
    samples = samples.drop_duplicates(["stock_code", "published_date"])
    samples["published_date"] = pd.to_datetime(samples["published_date"]).dt.date
    counts = {
        "company_days": len(samples),
        "supplement_selected": 0,
        "existing_fallback_selected": 0,
        "taxonomy_not_yet_effective": 0,
        "missing": 0,
        "ambiguous": 0,
        "cross_source_comparable": 0,
        "cross_source_agree": 0,
        "cross_source_disagree": 0,
    }
    missing_codes: set[str] = set()
    taxonomy_not_effective_codes: set[str] = set()
    newly_covered_codes: set[str] = set()
    newly_covered_days = 0
    for stock_code, dates in samples.groupby("stock_code", sort=False):
        supplement_rows = membership.loc[membership["con_code"].eq(stock_code)]
        supplement_all_rows = membership_all.loc[
            membership_all["con_code"].eq(stock_code)
        ]
        existing_rows = existing.loc[existing["ts_code"].eq(stock_code)]
        for day in dates["published_date"]:
            supplement_ids = _active_identities(supplement_rows, day, "index_code")
            if not supplement_ids:
                supplement_ids = _active_identities(
                    supplement_all_rows, day, "index_code"
                )
            existing_ids = _active_identities(existing_rows, day, "l1_code")
            if supplement_ids and existing_ids:
                counts["cross_source_comparable"] += 1
                if supplement_ids == existing_ids:
                    counts["cross_source_agree"] += 1
                else:
                    counts["cross_source_disagree"] += 1
            if len(supplement_ids) == 1:
                counts["supplement_selected"] += 1
                if not existing_ids:
                    newly_covered_days += 1
                    newly_covered_codes.add(str(stock_code))
            elif len(supplement_ids) > 1:
                counts["ambiguous"] += 1
            elif len(existing_ids) == 1:
                counts["existing_fallback_selected"] += 1
            elif len(existing_ids) > 1:
                counts["ambiguous"] += 1
            else:
                all_in_dates = pd.concat(
                    [
                        pd.to_datetime(
                            supplement_rows["in_date"], errors="coerce"
                        ),
                        pd.to_datetime(
                            supplement_all_rows["in_date"], errors="coerce"
                        ),
                        pd.to_datetime(existing_rows["in_date"], errors="coerce"),
                    ],
                    ignore_index=True,
                ).dropna()
                if len(all_in_dates) and pd.Timestamp(day) < all_in_dates.min():
                    counts["taxonomy_not_yet_effective"] += 1
                    taxonomy_not_effective_codes.add(str(stock_code))
                else:
                    counts["missing"] += 1
                    missing_codes.add(str(stock_code))
    covered = counts["supplement_selected"] + counts["existing_fallback_selected"]
    raw_integrity_passed = not errors
    release_eligible = raw_integrity_passed and not counts["missing"] and not counts["ambiguous"]
    report = {
        "contract": INDUSTRY_SUPPLEMENT_AUDIT_CONTRACT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_role": "train",
        "official_test_queried": False,
        "raw_integrity_passed": raw_integrity_passed,
        "feature_release_eligible": release_eligible,
        "release_blockers": [
            name
            for name, blocked in (
                ("missing_train_date_membership", counts["missing"] > 0),
                ("ambiguous_train_date_membership", counts["ambiguous"] > 0),
                ("cross_source_l1_disagreement", counts["cross_source_disagree"] > 0),
                ("raw_integrity_error", bool(errors)),
            )
            if blocked
        ],
        "source_contract_sha256": contract.source_sha256,
        "state_sha256": sha256_file(state_path),
        "samples_sha256": sha256_file(samples_path),
        "universe_codes": len(codes),
        "classification_rows": len(classification),
        "raw_membership_rows": sum(len(frame) for frame in memberships),
        "mapped_l1_membership_rows": len(membership),
        "mapped_l1_codes": int(membership["con_code"].nunique()),
        "coverage": {
            **counts,
            "covered": covered,
            "covered_fraction": covered / len(samples) if len(samples) else 0.0,
            "state_covered": covered + counts["taxonomy_not_yet_effective"],
            "state_coverage_fraction": (
                (covered + counts["taxonomy_not_yet_effective"]) / len(samples)
                if len(samples)
                else 0.0
            ),
            "newly_covered_days": newly_covered_days,
            "newly_covered_codes": sorted(newly_covered_codes),
            "missing_codes": sorted(missing_codes),
            "taxonomy_not_yet_effective_codes": sorted(taxonomy_not_effective_codes),
        },
        "errors": errors,
    }
    return report

"""Build train-only v3 features from audited industry and PIT financial sources."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.prediction.tabular.catalog_v3 import TabularFeatureCatalogV3
from src.prediction.tabular.industry_supplement import _active_identities

INDUSTRY_AUDIT_CONTRACT = "tabular-industry-supplement-audit-v1"
FINANCIAL_AUDIT_CONTRACT = "tabular-financial-revision-audit-v1"
V3_DATASET_CONTRACT = "causal-tabular-train-dataset-v3"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _available_at(day: date) -> pd.Timestamp:
    return pd.Timestamp(
        datetime.combine(day, time(8, 0), tzinfo=ZoneInfo("Asia/Shanghai"))
    ).tz_convert("UTC")


def _load_industry_rows(
    root: Path, *, existing_root: Path | None = None
) -> tuple[dict[str, pd.DataFrame], set[str]]:
    classification = pd.concat(
        [pd.read_parquet(path) for path in (root / "raw" / "classification").glob("*.parquet")],
        ignore_index=True,
    )
    l1_codes = set(
        classification.loc[classification["level"].eq("L1"), "index_code"].astype(str)
    )
    rows: dict[str, pd.DataFrame] = {}
    for path in (root / "raw" / "membership").glob("*.parquet"):
        rows[path.stem] = pd.read_parquet(path)
    for path in (root / "raw" / "membership_all").glob("*.parquet"):
        rows.setdefault(f"all:{path.stem}", pd.read_parquet(path))
    if existing_root is not None:
        for path in (
            existing_root / "raw" / "company_industry" / "sw_membership"
        ).glob("*.parquet"):
            frame = pd.read_parquet(path).rename(
                columns={"l1_code": "index_code", "ts_code": "con_code"}
            )
            rows[f"existing:{path.stem}"] = frame
            l1_codes.update(frame["index_code"].dropna().astype(str))
    return rows, l1_codes


def _industry_features(
    samples: pd.DataFrame, *, root: Path, existing_root: Path | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows, l1_codes = _load_industry_rows(root, existing_root=existing_root)
    codebook = {code: index for index, code in enumerate(sorted(l1_codes), start=1)}
    output = []
    for stock_code, group in samples.groupby("stock_code", sort=False):
        legacy = rows.get(stock_code, pd.DataFrame())
        fallback = rows.get(f"all:{stock_code}", pd.DataFrame())
        existing = rows.get(f"existing:{stock_code}", pd.DataFrame())
        legacy = legacy.loc[
            legacy.get("index_code", pd.Series(dtype=str)).astype(str).isin(l1_codes)
        ]
        fallback = fallback.loc[
            fallback.get("index_code", pd.Series(dtype=str)).astype(str).isin(l1_codes)
        ]
        for row_index, row in group.iterrows():
            day = pd.Timestamp(row["published_date"]).date()
            identities = _active_identities(legacy, day, "index_code")
            if not identities:
                identities = _active_identities(fallback, day, "index_code")
            if not identities:
                identities = _active_identities(existing, day, "index_code")
            ambiguous = len(identities) > 1
            selected = sorted(identities)[0] if len(identities) == 1 else None
            all_dates = pd.concat(
                [
                    pd.to_datetime(
                        legacy.get("in_date", pd.Series(dtype="datetime64[ns]")),
                        errors="coerce",
                    ),
                    pd.to_datetime(
                        fallback.get("in_date", pd.Series(dtype="datetime64[ns]")),
                        errors="coerce",
                    ),
                    pd.to_datetime(
                        existing.get("in_date", pd.Series(dtype="datetime64[ns]")),
                        errors="coerce",
                    ),
                ],
                ignore_index=True,
            ).dropna()
            not_yet_effective = (
                not identities and len(all_dates) > 0 and pd.Timestamp(day) < all_dates.min()
            )
            output.append(
                {
                    "__row_index": row_index,
                    "industry_l1_code_id": codebook.get(selected, 0),
                    "industry_taxonomy_not_yet_effective": bool(not_yet_effective),
                    "industry_membership_ambiguous": bool(ambiguous),
                    "industry_features_available_at_utc": _available_at(day),
                }
            )
    return pd.DataFrame(output).set_index("__row_index").reindex(samples.index), codebook


def _load_financial_frames(root: Path, endpoint: str) -> pd.DataFrame:
    frames = [
        pd.read_parquet(path)
        for path in (root / "raw" / "fundamentals" / endpoint).glob("*.parquet")
    ]
    result = pd.concat(frames, ignore_index=True)
    result["actual_disclosure_date"] = pd.to_datetime(
        result["actual_disclosure_date"], errors="coerce"
    ).dt.date
    return result


def _financial_features(samples: pd.DataFrame, *, root: Path) -> pd.DataFrame:
    endpoint_fields = {
        "income": ("basic_eps", "total_revenue", "n_income"),
        "balancesheet": ("total_assets", "total_liab"),
        "cashflow": ("n_cashflow_act", "free_cashflow"),
    }
    prefixes = {
        "income": "pit_income",
        "balancesheet": "pit_balancesheet",
        "cashflow": "pit_cashflow",
    }
    grouped = {
        endpoint: _load_financial_frames(root, endpoint).sort_values(
            ["ts_code", "actual_disclosure_date", "f_ann_date", "update_flag"],
            kind="mergesort",
        ).groupby("ts_code", sort=False)
        for endpoint in endpoint_fields
    }
    output = []
    for row_index, row in samples.iterrows():
        day = pd.Timestamp(row["published_date"]).date()
        values: dict[str, object] = {"__row_index": row_index}
        for endpoint, fields in endpoint_fields.items():
            try:
                stock_frame = grouped[endpoint].get_group(row["stock_code"])
            except KeyError:
                stock_frame = None
            latest = None
            if stock_frame is not None and not stock_frame.empty:
                disclosure_days = stock_frame["actual_disclosure_date"].to_numpy(
                    dtype="datetime64[D]"
                )
                position = (
                    int(
                        np.searchsorted(
                            disclosure_days, np.datetime64(day), side="left"
                        )
                    )
                    - 1
                )
                if position >= 0:
                    latest = stock_frame.iloc[position]
            prefix = prefixes[endpoint]
            available = (
                _available_at(latest["actual_disclosure_date"])
                if latest is not None
                else pd.NaT
            )
            values[f"{prefix}_features_available_at_utc"] = available
            values[f"{prefix}_observation_date"] = (
                latest["actual_disclosure_date"] if latest is not None else pd.NaT
            )
            for field in fields:
                values[f"{prefix}_{field}"] = (
                    latest[field] if latest is not None else np.nan
                )
        output.append(values)
    return pd.DataFrame(output).set_index("__row_index").reindex(samples.index)


def build_tabular_train_dataset_v3(
    *,
    v2_root: Path,
    industry_root: Path,
    financial_root: Path,
    industry_audit_path: Path,
    financial_audit_path: Path,
    catalog: TabularFeatureCatalogV3,
    existing_industry_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build v3 without mutating v2 and require both source audits to pass."""
    v2_manifest = json.loads((v2_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        v2_manifest.get("dataset_role") != "train"
        or v2_manifest.get("official_test_queried") is not False
    ):
        raise PermissionError("v2 base is not sealed train-only")
    industry_audit = json.loads(industry_audit_path.read_text(encoding="utf-8"))
    financial_audit = json.loads(financial_audit_path.read_text(encoding="utf-8"))
    if (
        industry_audit.get("contract") != INDUSTRY_AUDIT_CONTRACT
        or not industry_audit.get("feature_release_eligible")
    ):
        raise PermissionError("industry source audit has not passed")
    if (
        financial_audit.get("contract") != FINANCIAL_AUDIT_CONTRACT
        or not financial_audit.get("feature_release_eligible")
    ):
        raise PermissionError("financial source audit has not passed")
    features = pd.read_parquet(v2_root / "features.parquet")
    labels = pd.read_parquet(v2_root / "labels.parquet")
    sample_keys = features[["stock_code", "published_date"]].drop_duplicates(
        ["stock_code", "published_date"], ignore_index=True
    )
    industry, codebook = _industry_features(
        sample_keys, root=industry_root, existing_root=existing_industry_root
    )
    financial = _financial_features(sample_keys, root=financial_root)
    additions = pd.concat(
        [
            sample_keys.reset_index(drop=True),
            industry.reset_index(drop=True),
            financial.reset_index(drop=True),
        ],
        axis=1,
    )
    result = features.merge(
        additions,
        on=["stock_code", "published_date"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    for item in catalog.features:
        if item.name not in result.columns:
            raise ValueError(f"v3 feature missing from built frame: {item.name}")
    manifest = {
        "contract": V3_DATASET_CONTRACT,
        "dataset_role": "train",
        "official_test_queried": False,
        "rows": len(result),
        "feature_columns": list(catalog.feature_names),
        "base_v2_manifest_sha256": _sha256_file(v2_root / "manifest.json"),
        "catalog_sha256": catalog.source_sha256,
        "industry_audit_sha256": _sha256_file(industry_audit_path),
        "financial_audit_sha256": _sha256_file(financial_audit_path),
        "industry_l1_codebook": codebook,
        "point_in_time_rule": (
            "financial actual_disclosure_date < published_date; "
            "industry interval [in_date,out_date)"
        ),
        "feature_schema_sha256": hashlib.sha256(
            "\n".join(catalog.feature_names).encode()
        ).hexdigest(),
    }
    return result, labels, manifest

"""Resumable raw archives for historical company/industry and PIT financial data."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from datetime import time as datetime_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from src.config import PROJECT_ROOT, config
from src.pipeline.fetcher import TushareClient
from src.prediction.development_contract import CausalDevelopmentContract
from src.prediction.splits import PredictionSplitContract
from src.prediction.tabular.contract import TabularModelContract

SHANGHAI = ZoneInfo("Asia/Shanghai")
PIT_SOURCE_CONTRACT = "tabular-point-in-time-source-backfill-v1"
PIT_CONFIG_CONTRACT = "tabular-point-in-time-sources-v1"
DEFAULT_PIT_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "tabular_point_in_time_sources_v1.yaml"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", index=False)
    temporary.replace(path)
    return sha256_file(path)


@dataclass(frozen=True)
class PointInTimeSourceContract:
    dataset_role: str
    universe_path: Path
    stock_basic_statuses: tuple[str, ...]
    stock_basic_fields: tuple[str, ...]
    industry_fields: tuple[str, ...]
    financial_start: date
    financial_end: date
    financial_identity_fields: tuple[str, ...]
    financial_fields: dict[str, tuple[str, ...]]
    source_path: Path
    source_sha256: str


def load_point_in_time_source_contract(
    path: Path = DEFAULT_PIT_CONFIG_PATH,
) -> PointInTimeSourceContract:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    if raw.get("contract") != PIT_CONFIG_CONTRACT:
        raise ValueError("unexpected point-in-time source contract")
    if raw.get("dataset_role") != "train" or raw.get("official_test_allowed") is not False:
        raise PermissionError("point-in-time sources must remain train-only")
    fundamentals = raw["fundamentals"]
    if fundamentals.get("availability_rule") != (
        "f_ann_date_fallback_ann_date_end_of_day_asia_shanghai"
    ):
        raise ValueError("unexpected financial availability rule")
    endpoints = fundamentals["endpoints"]
    if set(endpoints) != {"income", "balancesheet", "cashflow"}:
        raise ValueError("financial endpoint set changed")
    financial_fields = {
        str(name): tuple(str(value) for value in item["fields"])
        for name, item in endpoints.items()
    }
    identity_fields = tuple(
        str(value) for value in fundamentals.get("identity_fields", [])
    )
    required_identity = {
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "report_type",
        "comp_type",
        "end_type",
        "update_flag",
    }
    if set(identity_fields) != required_identity:
        raise ValueError("unexpected financial identity/version fields")
    if any(not required_identity.issubset(fields) for fields in financial_fields.values()):
        raise ValueError("financial endpoint is missing identity/version fields")
    config = PointInTimeSourceContract(
        dataset_role="train",
        universe_path=(PROJECT_ROOT / raw["universe"]["source"]).resolve(),
        stock_basic_statuses=tuple(
            str(value)
            for value in raw["company_industry"]["stock_basic"]["list_statuses"]
        ),
        stock_basic_fields=tuple(
            str(value)
            for value in raw["company_industry"]["stock_basic"]["fields"]
        ),
        industry_fields=tuple(
            str(value)
            for value in raw["company_industry"]["sw_industry_membership"]["fields"]
        ),
        financial_start=date.fromisoformat(str(fundamentals["source_start"])),
        financial_end=date.fromisoformat(str(fundamentals["source_end"])),
        financial_identity_fields=identity_fields,
        financial_fields=financial_fields,
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    if config.financial_end > date(2024, 12, 31):
        raise PermissionError("financial source contract crosses the train cutoff")
    return config


def load_train_universe(path: Path) -> list[str]:
    state = json.loads(path.read_text(encoding="utf-8"))
    selection = state.get("selection", {})
    if selection.get("dataset_role") != "train":
        raise PermissionError("point-in-time universe is not train-only")
    if state.get("official_test_queried") is not False:
        raise PermissionError("point-in-time universe touched official test")
    codes = sorted({str(key).split(":", 1)[0] for key in state.get("completed", {})})
    expected_hash = hashlib.sha256("\n".join(codes).encode()).hexdigest()
    if len(codes) != selection.get("codes"):
        raise ValueError("point-in-time universe count mismatch")
    if expected_hash != selection.get("codes_sha256"):
        raise ValueError("point-in-time universe hash mismatch")
    return codes


class TusharePointInTimeClient:
    def __init__(self, *, minimum_interval_seconds: float = 0.15) -> None:
        self.pro = TushareClient().pro
        configured_floor = 60.0 / max(config.tushare.rate_limit_per_minute, 1)
        self.minimum_interval_seconds = max(
            float(minimum_interval_seconds), configured_floor * 1.05
        )
        self._last_call: float | None = None

    def fetch(self, endpoint: str, **kwargs: Any) -> pd.DataFrame:
        if self._last_call is not None:
            remaining = self.minimum_interval_seconds - (
                time.monotonic() - self._last_call
            )
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()
        result = getattr(self.pro, endpoint)(**kwargs)
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame()


def _state_base(
    *,
    selection: dict,
    state_path: Path,
    contract: PointInTimeSourceContract,
    split: PredictionSplitContract,
    development: CausalDevelopmentContract,
    tabular: TabularModelContract,
) -> dict:
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    if state.get("selection") not in (None, selection):
        raise ValueError("point-in-time source state selection differs from this run")
    state.setdefault("started_at_utc", datetime.now(UTC).isoformat())
    state.update(
        {
            "contract": PIT_SOURCE_CONTRACT,
            "source_contract": PIT_CONFIG_CONTRACT,
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
    return state


def _record_frame(
    *, frame: pd.DataFrame, target: Path, relative: Path, status_extra: dict | None = None
) -> dict:
    if frame.empty:
        return {"status": "empty", "rows": 0, "path": None, "sha256": None}
    digest = write_parquet_atomic(frame, target)
    return {
        "status": "downloaded",
        "rows": len(frame),
        "path": relative.as_posix(),
        "sha256": digest,
        **(status_extra or {}),
    }


def _fetch_with_retries(
    client: TusharePointInTimeClient,
    endpoint: str,
    *,
    maximum_attempts: int,
    **kwargs: Any,
) -> pd.DataFrame:
    error: Exception | None = None
    for attempt in range(1, maximum_attempts + 1):
        try:
            return client.fetch(endpoint, **kwargs)
        except Exception as exc:  # pragma: no cover - live service behavior
            error = exc
            if attempt < maximum_attempts:
                time.sleep(min(2**attempt, 15))
    assert error is not None
    raise error


def _normalize_stock_basic(
    frame: pd.DataFrame,
    *,
    fields: tuple[str, ...],
    universe: set[str],
    fetched_at: datetime,
) -> pd.DataFrame:
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ValueError(f"stock_basic response missing columns: {sorted(missing)}")
    result = frame.loc[frame["ts_code"].astype(str).isin(universe), fields].copy()
    result = result.drop_duplicates().sort_values("ts_code", kind="mergesort")
    for column in ("list_date", "delist_date"):
        result[column] = pd.to_datetime(
            result[column], format="%Y%m%d", errors="coerce"
        ).dt.date
    result["source"] = "tushare_stock_basic_current_snapshot"
    result["retrieved_at_utc"] = pd.Timestamp(fetched_at).tz_convert("UTC")
    result["historical_release_status"] = "provisional_current_snapshot"
    return result.reset_index(drop=True)


def _normalize_industry(
    frame: pd.DataFrame,
    *,
    stock_code: str,
    fields: tuple[str, ...],
    query_is_new: str,
    fetched_at: datetime,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*fields, "query_is_new"])
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ValueError(f"industry response missing columns: {sorted(missing)}")
    result = frame.loc[:, fields].copy()
    if set(result["ts_code"].astype(str).unique()) != {stock_code}:
        raise ValueError("industry response contains another stock code")
    if not set(result["is_new"].astype(str)).issubset({"Y", "N"}):
        raise ValueError("industry response has invalid is_new")
    for column in ("in_date", "out_date"):
        result[column] = pd.to_datetime(
            result[column], format="%Y%m%d", errors="coerce"
        ).dt.date
    result["query_is_new"] = query_is_new
    result["source"] = "tushare_index_member_all"
    result["retrieved_at_utc"] = pd.Timestamp(fetched_at).tz_convert("UTC")
    return result.drop_duplicates().sort_values(
        ["in_date", "out_date", "l1_code"], kind="mergesort", na_position="last"
    )


def download_company_industry_sources(
    *,
    codes: list[str],
    output_root: Path,
    state_path: Path,
    client: TusharePointInTimeClient,
    contract: PointInTimeSourceContract,
    split: PredictionSplitContract,
    development: CausalDevelopmentContract,
    tabular: TabularModelContract,
    maximum_attempts: int = 4,
) -> dict:
    selected_codes = sorted(set(codes))
    selection = {
        "source_group": "company_industry",
        "dataset_role": "train",
        "codes": len(selected_codes),
        "codes_sha256": hashlib.sha256("\n".join(selected_codes).encode()).hexdigest(),
        "stock_basic_statuses": list(contract.stock_basic_statuses),
        "stock_basic_fields": list(contract.stock_basic_fields),
        "industry_fields": list(contract.industry_fields),
        "industry_queries": ["Y", "N"],
    }
    state = _state_base(
        selection=selection,
        state_path=state_path,
        contract=contract,
        split=split,
        development=development,
        tabular=tabular,
    )
    completed = dict(state.get("completed", {}))
    failures = dict(state.get("failures", {}))
    stock_key = "stock_basic:all_statuses"
    if stock_key not in completed:
        try:
            fetched_at = datetime.now(UTC)
            frames = [
                _fetch_with_retries(
                    client,
                    "stock_basic",
                    maximum_attempts=maximum_attempts,
                    list_status=status,
                    fields=",".join(contract.stock_basic_fields),
                )
                for status in contract.stock_basic_statuses
            ]
            combined = pd.concat(frames, ignore_index=True)
            normalized = _normalize_stock_basic(
                combined,
                fields=contract.stock_basic_fields,
                universe=set(selected_codes),
                fetched_at=fetched_at,
            )
            relative = Path("raw") / "company_industry" / "stock_basic.parquet"
            completed[stock_key] = _record_frame(
                frame=normalized,
                target=output_root / relative,
                relative=relative,
                status_extra={"covered_codes": int(normalized["ts_code"].nunique())},
            )
            failures.pop(stock_key, None)
        except Exception as exc:  # pragma: no cover - live service behavior
            failures[stock_key] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "updated_at_utc": datetime.now(UTC).isoformat(),
            }
        state.update({"completed": completed, "failures": failures})
        save_json_atomic(state_path, state)
    for index, stock_code in enumerate(selected_codes, 1):
        key = f"industry:{stock_code}"
        if key in completed:
            continue
        try:
            fetched_at = datetime.now(UTC)
            frames = []
            for is_new in ("Y", "N"):
                raw = _fetch_with_retries(
                    client,
                    "index_member_all",
                    maximum_attempts=maximum_attempts,
                    ts_code=stock_code,
                    is_new=is_new,
                )
                frames.append(
                    _normalize_industry(
                        raw,
                        stock_code=stock_code,
                        fields=contract.industry_fields,
                        query_is_new=is_new,
                        fetched_at=fetched_at,
                    )
                )
            normalized = pd.concat(frames, ignore_index=True)
            relative = (
                Path("raw") / "company_industry" / "sw_membership" / f"{stock_code}.parquet"
            )
            completed[key] = _record_frame(
                frame=normalized,
                target=output_root / relative,
                relative=relative,
                status_extra={
                    "current_rows": int((normalized.get("is_new") == "Y").sum()),
                    "historical_rows": int((normalized.get("is_new") == "N").sum()),
                },
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
            f"company_industry progress {index}/{len(selected_codes)}; "
            f"completed={len(completed)}; failures={len(failures)}",
            flush=True,
        )
    if not failures and len(completed) == len(selected_codes) + 1:
        state.setdefault("finished_at_utc", datetime.now(UTC).isoformat())
        save_json_atomic(state_path, state)
    return state


def _normalize_financial(
    frame: pd.DataFrame,
    *,
    stock_code: str,
    endpoint: str,
    fields: tuple[str, ...],
    start: date,
    end: date,
    fetched_at: datetime,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=fields)
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ValueError(f"{endpoint} response missing columns: {sorted(missing)}")
    result = frame.loc[:, fields].copy()
    if set(result["ts_code"].astype(str).unique()) != {stock_code}:
        raise ValueError(f"{endpoint} response contains another stock code")
    parsed_dates = {}
    for column in ("ann_date", "f_ann_date", "end_date"):
        parsed_dates[column] = pd.to_datetime(
            result[column], format="%Y%m%d", errors="coerce"
        ).dt.date
        result[column] = parsed_dates[column]
    announcement = parsed_dates["ann_date"]
    if announcement.isna().any():
        raise ValueError(f"{endpoint} contains no usable announcement date")
    if announcement.min() < start or announcement.max() > end:
        raise ValueError(f"{endpoint} response leaves the configured query window")
    actual = parsed_dates["f_ann_date"].where(
        parsed_dates["f_ann_date"].notna(), parsed_dates["ann_date"]
    )
    if actual.isna().any():
        raise ValueError(f"{endpoint} contains no usable disclosure date")
    result["actual_disclosure_date"] = actual
    result["availability_date_source"] = pd.Series(
        [
            "f_ann_date"
            if value is not None and not pd.isna(value)
            else "ann_date"
            for value in parsed_dates["f_ann_date"]
        ],
        index=result.index,
    )
    result["available_at_utc"] = pd.to_datetime(
        [datetime.combine(value, datetime_time.max, tzinfo=SHANGHAI) for value in actual],
        utc=True,
    )
    result["source"] = f"tushare_{endpoint}"
    result["retrieved_at_utc"] = pd.Timestamp(fetched_at).tz_convert("UTC")
    result["dataset_role"] = "train"
    return result.sort_values(
        ["actual_disclosure_date", "end_date", "report_type"],
        kind="mergesort",
    ).reset_index(drop=True)


def download_financial_sources(
    *,
    codes: list[str],
    output_root: Path,
    state_path: Path,
    client: TusharePointInTimeClient,
    contract: PointInTimeSourceContract,
    split: PredictionSplitContract,
    development: CausalDevelopmentContract,
    tabular: TabularModelContract,
    maximum_attempts: int = 4,
) -> dict:
    selected_codes = sorted(set(codes))
    selection = {
        "source_group": "fundamentals",
        "dataset_role": "train",
        "codes": len(selected_codes),
        "codes_sha256": hashlib.sha256("\n".join(selected_codes).encode()).hexdigest(),
        "source_start": contract.financial_start.isoformat(),
        "source_end": contract.financial_end.isoformat(),
        "endpoints": {
            name: list(fields) for name, fields in contract.financial_fields.items()
        },
        "availability_rule": (
            "f_ann_date_fallback_ann_date_end_of_day_asia_shanghai"
        ),
    }
    state = _state_base(
        selection=selection,
        state_path=state_path,
        contract=contract,
        split=split,
        development=development,
        tabular=tabular,
    )
    completed = dict(state.get("completed", {}))
    failures = dict(state.get("failures", {}))
    work = [
        (stock_code, endpoint)
        for stock_code in selected_codes
        for endpoint in contract.financial_fields
    ]
    for index, (stock_code, endpoint) in enumerate(work, 1):
        key = f"{endpoint}:{stock_code}"
        if key in completed:
            continue
        try:
            fetched_at = datetime.now(UTC)
            fields = contract.financial_fields[endpoint]
            raw = _fetch_with_retries(
                client,
                endpoint,
                maximum_attempts=maximum_attempts,
                ts_code=stock_code,
                start_date=contract.financial_start.strftime("%Y%m%d"),
                end_date=contract.financial_end.strftime("%Y%m%d"),
                fields=",".join(fields),
            )
            normalized = _normalize_financial(
                raw,
                stock_code=stock_code,
                endpoint=endpoint,
                fields=fields,
                start=contract.financial_start,
                end=contract.financial_end,
                fetched_at=fetched_at,
            )
            relative = Path("raw") / "fundamentals" / endpoint / f"{stock_code}.parquet"
            extra = {}
            if not normalized.empty:
                extra = {
                    "first_actual_disclosure_date": str(
                        normalized["actual_disclosure_date"].min()
                    ),
                    "last_actual_disclosure_date": str(
                        normalized["actual_disclosure_date"].max()
                    ),
                }
            completed[key] = _record_frame(
                frame=normalized,
                target=output_root / relative,
                relative=relative,
                status_extra=extra,
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
            f"fundamentals progress {index}/{len(work)}; "
            f"completed={len(completed)}; failures={len(failures)}",
            flush=True,
        )
    if not failures and len(completed) == len(work):
        state.setdefault("finished_at_utc", datetime.now(UTC).isoformat())
        save_json_atomic(state_path, state)
    return state

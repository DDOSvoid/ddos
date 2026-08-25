"""Deterministic plan and CDP evidence audit for financial revision chains."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.config import PROJECT_ROOT
from src.pipeline.announcement_content import (
    assemble_content_pages,
    assemble_pdf_extracted_content,
)
from src.pipeline.fetcher import EastmoneyClient
from src.prediction.tabular.point_in_time_sources import (
    PointInTimeSourceContract,
    sha256_file,
)

FINANCIAL_REVISION_CONFIG_CONTRACT = "tabular-financial-revision-evidence-v1"
FINANCIAL_REVISION_PLAN_CONTRACT = "tabular-financial-revision-plan-v1"
FINANCIAL_REVISION_BACKFILL_CONTRACT = "tabular-financial-revision-backfill-v1"
FINANCIAL_REVISION_AUDIT_CONTRACT = "tabular-financial-revision-audit-v1"
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "tabular_financial_revision_evidence_v1.yaml"
)


@dataclass(frozen=True)
class FinancialRevisionEvidenceContract:
    selection_seed: str
    sample_per_endpoint: int
    announcement_date_start: date
    announcement_date_end: date
    group_fields: tuple[str, ...]
    derived_metric_fields: dict[str, tuple[str, ...]]
    excluded_announcement_ids: dict[str, str]
    database_path: Path
    minimum_verified_groups: int
    minimum_verified_groups_per_endpoint: int
    provider_documentation_urls: tuple[str, ...]
    source_path: Path
    source_sha256: str


def load_financial_revision_evidence_contract(
    path: Path = DEFAULT_CONFIG_PATH,
) -> FinancialRevisionEvidenceContract:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    if raw.get("contract") != FINANCIAL_REVISION_CONFIG_CONTRACT:
        raise ValueError("unexpected financial revision evidence contract")
    if raw.get("dataset_role") != "train" or raw.get("official_test_allowed") is not False:
        raise PermissionError("financial revision evidence must remain train-only")
    if raw["evidence"].get("changed_metric_match_required") is not True:
        raise ValueError("changed financial metrics must be matched to source evidence")
    return FinancialRevisionEvidenceContract(
        selection_seed=str(raw["selection_seed"]),
        sample_per_endpoint=int(raw["sample_per_endpoint"]),
        announcement_date_start=date.fromisoformat(str(raw["announcement_date_start"])),
        announcement_date_end=date.fromisoformat(str(raw["announcement_date_end"])),
        group_fields=tuple(str(value) for value in raw["revision_group_fields"]),
        derived_metric_fields={
            str(endpoint): tuple(str(field) for field in fields)
            for endpoint, fields in raw.get("derived_metric_fields", {}).items()
        },
        excluded_announcement_ids={
            str(announcement_id): str(reason)
            for announcement_id, reason in raw.get(
                "excluded_announcement_ids", {}
            ).items()
        },
        database_path=(PROJECT_ROOT / raw["evidence"]["metadata_source"]).resolve(),
        minimum_verified_groups=int(raw["evidence"]["minimum_verified_revision_groups"]),
        minimum_verified_groups_per_endpoint=int(
            raw["evidence"]["minimum_verified_groups_per_endpoint"]
        ),
        provider_documentation_urls=tuple(
            str(value)
            for value in raw["evidence"]["provider_revision_documentation"]
        ),
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _load_financial_endpoint(
    *, root: Path, endpoint: str, source_contract: PointInTimeSourceContract
) -> pd.DataFrame:
    frames = [
        pd.read_parquet(path)
        for path in (root / "raw" / "fundamentals" / endpoint).glob("*.parquet")
    ]
    if not frames:
        return pd.DataFrame(columns=source_contract.financial_fields[endpoint])
    frame = pd.concat(frames, ignore_index=True)
    for column in ("ann_date", "f_ann_date", "end_date", "actual_disclosure_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise").dt.date
    return frame


def _period_terms(end_type: str, end_date: date) -> tuple[str, ...]:
    year = end_date.year
    mapping = {
        "1": (f"{year}年第一季度报告", f"{year}年一季度报告"),
        "2": (f"{year}年半年度报告", f"{year}年中期报告"),
        "3": (f"{year}年第三季度报告", f"{year}年三季度报告"),
        "4": (f"{year}年年度报告",),
    }
    return mapping.get(str(end_type), (str(year),))


def _title_score(
    title: str, *, end_type: str, end_date: date, revision_index: int
) -> int:
    normalized = "".join(str(title).split())
    score = 0
    period_match = any(
        term in normalized for term in _period_terms(end_type, end_date)
    )
    if period_match:
        score += 120
    if "报告全文" in normalized or re.search(r"报告(?:\(更正后\))?$", normalized):
        score += 30
    if "摘要" in normalized:
        score -= 80
    if "审计报告" in normalized or "工作报告" in normalized:
        score -= 40
    revision_terms = ("更正", "追溯", "重述", "会计差错", "调整财务报表")
    revision_match = any(term in normalized for term in revision_terms)
    statement_match = "财务报表" in normalized
    # A generic prior-error/audit report without a quarter or half-year marker
    # cannot establish a point-in-time match for an interim statement. Keep it
    # eligible only for annual (`end_type=4`) groups, where the report's annual
    # correction scope can be independently checked.
    if revision_index > 0 and revision_match and not period_match and str(end_type) != "4":
        return -1000
    if not period_match and not revision_match and not statement_match:
        return -1000
    if revision_match:
        score += 100 if revision_index > 0 else -40
    if revision_index > 0 and any(
        term in normalized for term in ("年度报告", "季度报告", "半年度报告")
    ):
        score += 20
    return score


def _load_announcement_candidates(
    *, database_path: Path, stock_codes: set[str], dates: set[date]
) -> dict[tuple[str, date], list[dict[str, Any]]]:
    if not stock_codes or not dates:
        return {}
    start = min(dates).isoformat()
    end = max(dates).isoformat()
    placeholders = ",".join("?" for _ in stock_codes)
    query = f"""
        SELECT c.stock_code, a.published_date, a.announcement_id, a.title,
               a.raw_response
        FROM announcements AS a
        JOIN companies AS c ON c.id = a.company_id
        WHERE c.stock_code IN ({placeholders})
          AND a.published_date BETWEEN ? AND ?
          AND a.announcement_id IS NOT NULL
        ORDER BY c.stock_code, a.published_date, a.id
    """
    parameters = [*sorted(stock_codes), start, end]
    result: dict[tuple[str, date], list[dict[str, Any]]] = {}
    with sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute(query, parameters):
            published = date.fromisoformat(str(row["published_date"]))
            if published not in dates:
                continue
            item = dict(row)
            item["published_date"] = published
            result.setdefault((str(row["stock_code"]), published), []).append(item)
    return result


def _canonical_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _candidate_group(
    *,
    endpoint: str,
    key: tuple[Any, ...],
    group: pd.DataFrame,
    candidates: dict[tuple[str, date], list[dict[str, Any]]],
    source_contract: PointInTimeSourceContract,
    group_fields: tuple[str, ...],
) -> dict[str, Any] | None:
    ordered = group.sort_values(
        ["actual_disclosure_date", "ann_date", "update_flag"], kind="mergesort"
    )
    versions_by_date = []
    for _, same_day in ordered.groupby("actual_disclosure_date", sort=True):
        latest = same_day.loc[same_day["update_flag"].astype(str).eq("1")]
        selected = latest if len(latest) else same_day
        if len(selected) != 1:
            return None
        versions_by_date.append(selected.iloc[0])
    versions = []
    for revision_index, row in enumerate(versions_by_date):
        stock_code = str(row["ts_code"])
        disclosure_date = row["actual_disclosure_date"]
        options = candidates.get((stock_code, disclosure_date), [])
        ranked = sorted(
            options,
            key=lambda item: (
                -_title_score(
                    str(item["title"]),
                    end_type=str(row["end_type"]),
                    end_date=row["end_date"],
                    revision_index=revision_index,
                ),
                str(item["announcement_id"]),
            ),
        )
        if not ranked:
            return None
        best = ranked[0]
        if _title_score(
            str(best["title"]),
            end_type=str(row["end_type"]),
            end_date=row["end_date"],
            revision_index=revision_index,
        ) <= 0:
            return None
        values = {
            field: _canonical_value(row[field])
            for field in source_contract.financial_fields[endpoint]
        }
        versions.append(
            {
                "actual_disclosure_date": disclosure_date.isoformat(),
                "update_flag": str(row["update_flag"]),
                "announcement_id": str(best["announcement_id"]),
                "announcement_title": str(best["title"]),
                "announcement_title_sha256": hashlib.sha256(
                    str(best["title"]).encode()
                ).hexdigest(),
                "financial_values": values,
            }
        )
    identity = {
        name: _canonical_value(value)
        for name, value in zip(group_fields, key, strict=True)
    }
    identity_text = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    return {
        "endpoint": endpoint,
        "identity": identity,
        "identity_sha256": hashlib.sha256(identity_text.encode()).hexdigest(),
        "flag_pattern": [item["update_flag"] for item in versions],
        "versions": versions,
    }


def build_financial_revision_plan(
    *,
    financial_root: Path,
    output_path: Path,
    contract: FinancialRevisionEvidenceContract,
    source_contract: PointInTimeSourceContract,
    split_contract: str,
    split_source_sha256: str,
    development_contract: str,
    development_source_sha256: str,
    tabular_contract: str,
    tabular_source_sha256: str,
) -> dict[str, Any]:
    revision_groups: list[tuple[str, tuple[Any, ...], pd.DataFrame]] = []
    all_codes: set[str] = set()
    all_dates: set[date] = set()
    total_groups = {}
    for endpoint in source_contract.financial_fields:
        frame = _load_financial_endpoint(
            root=financial_root, endpoint=endpoint, source_contract=source_contract
        )
        endpoint_groups = []
        for key, group in frame.groupby(list(contract.group_fields), dropna=False):
            if group["actual_disclosure_date"].nunique() <= 1:
                continue
            dates = set(group["actual_disclosure_date"])
            if (
                min(dates) < contract.announcement_date_start
                or max(dates) > contract.announcement_date_end
            ):
                continue
            endpoint_groups.append((endpoint, key, group))
            all_codes.add(str(group.iloc[0]["ts_code"]))
            all_dates.update(dates)
        total_groups[endpoint] = len(endpoint_groups)
        revision_groups.extend(endpoint_groups)
    candidates = _load_announcement_candidates(
        database_path=contract.database_path,
        stock_codes=all_codes,
        dates=all_dates,
    )
    eligible = []
    for endpoint, key, group in revision_groups:
        item = _candidate_group(
            endpoint=endpoint,
            key=key,
            group=group,
            candidates=candidates,
            source_contract=source_contract,
            group_fields=contract.group_fields,
        )
        if item is not None:
            eligible.append(item)
    excluded_ids = set(contract.excluded_announcement_ids)
    eligible = [
        item
        for item in eligible
        if not any(
            version["announcement_id"] in excluded_ids
            for version in item["versions"]
        )
    ]
    selected = []
    for endpoint in source_contract.financial_fields:
        endpoint_items = [item for item in eligible if item["endpoint"] == endpoint]
        derived_fields = set(contract.derived_metric_fields.get(endpoint, ()))

        def selection_key(item: dict[str, Any]) -> tuple[int, str]:
            changed_direct = 0
            for field in source_contract.financial_fields[endpoint]:
                if field in source_contract.financial_identity_fields or field in derived_fields:
                    continue
                values = [version["financial_values"].get(field) for version in item["versions"]]
                nonnull = [value for value in values if value is not None]
                changed_direct += int(
                    len(nonnull) >= 2 and len({str(value) for value in nonnull}) > 1
                )
            digest = hashlib.sha256(
                f"{contract.selection_seed}|{item['identity_sha256']}".encode()
            ).hexdigest()
            return (-changed_direct, digest)

        endpoint_items.sort(key=selection_key)
        endpoint_selected = []
        selected_codes: set[str] = set()
        for item in endpoint_items:
            stock_code = str(item["identity"]["ts_code"])
            if stock_code in selected_codes:
                continue
            endpoint_selected.append(item)
            selected_codes.add(stock_code)
            if len(endpoint_selected) >= contract.sample_per_endpoint:
                break
        if len(endpoint_selected) < contract.sample_per_endpoint:
            used = {item["identity_sha256"] for item in endpoint_selected}
            endpoint_selected.extend(
                item
                for item in endpoint_items
                if item["identity_sha256"] not in used
            )
        selected.extend(endpoint_selected[: contract.sample_per_endpoint])
    announcement_ids = sorted(
        {
            version["announcement_id"]
            for item in selected
            for version in item["versions"]
        }
    )
    plan = {
        "contract": FINANCIAL_REVISION_PLAN_CONTRACT,
        "dataset_role": "train",
        "official_test_queried": False,
        "market_targets_queried": False,
        "source_contract_sha256": source_contract.source_sha256,
        "revision_contract_sha256": contract.source_sha256,
        "split_contract": split_contract,
        "split_source_sha256": split_source_sha256,
        "development_contract": development_contract,
        "development_source_sha256": development_source_sha256,
        "tabular_contract": tabular_contract,
        "tabular_source_sha256": tabular_source_sha256,
        "financial_root": financial_root.resolve().as_posix(),
        "financial_state_sha256": sha256_file(financial_root / "fundamentals.state.json"),
        "eligible_revision_groups_by_endpoint": total_groups,
        "matched_revision_groups_by_endpoint": {
            endpoint: sum(item["endpoint"] == endpoint for item in eligible)
            for endpoint in source_contract.financial_fields
        },
        "excluded_announcement_ids": dict(contract.excluded_announcement_ids),
        "selected_revision_groups_by_endpoint": {
            endpoint: sum(item["endpoint"] == endpoint for item in selected)
            for endpoint in source_contract.financial_fields
        },
        "announcement_ids": announcement_ids,
        "revision_groups": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return plan


def _write_bytes_atomic(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return sha256_file(path)


def _save_json_atomic(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return sha256_file(path)


def _verification_title(title: str) -> str:
    """Return the invariant report title commonly printed on revised PDF covers."""
    comparable = str(title).strip()
    for separator in (":", "："):
        prefix, found, remainder = comparable.partition(separator)
        if found and len(prefix) <= 20:
            comparable = remainder.strip()
            break
    return re.sub(
        r"[（(][^（）()]{0,40}(?:更正|修订|稿|版)[^（）()]{0,20}[）)]\s*$",
        "",
        comparable,
    ).strip()


def _pdf_title_match_score(title: str, pages: list[dict[str, Any]]) -> float:
    comparable = _verification_title(title)
    if "关于" in comparable:
        comparable = comparable[comparable.index("关于") :]
    normalized_title = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", comparable)
    normalized_first_pages = re.sub(
        r"[^0-9A-Za-z\u4e00-\u9fff]+",
        "",
        "".join(str(page.get("notice_content") or "") for page in pages[:3]),
    )
    if len(normalized_title) < 6:
        return 0.0
    if normalized_title in normalized_first_pages:
        return 1.0
    title_bigrams = {
        normalized_title[index : index + 2]
        for index in range(len(normalized_title) - 1)
    }
    matched = {value for value in title_bigrams if value in normalized_first_pages}
    return len(matched) / max(len(title_bigrams), 1)


def _assemble_financial_pdf(
    data: bytes,
    *,
    art_code: str,
    title: str,
    notice_date: str,
    attachment_url: str,
) -> dict[str, Any]:
    """Extract a report PDF while recording isolated blank pages as provenance."""
    verification_title = _verification_title(title)
    try:
        return assemble_pdf_extracted_content(
            data,
            art_code=art_code,
            title=verification_title,
            notice_date=notice_date,
            source_eitime=None,
            attachment_url=attachment_url,
        )
    except ValueError as exc:
        if not re.fullmatch(r"PDF page \d+ has no extractable text", str(exc)):
            raise

    from pypdf import PdfReader
    from pypdf import __version__ as pypdf_version

    reader = PdfReader(BytesIO(data))
    if not reader.pages:
        raise ValueError("PDF contains no pages")
    pages: list[dict[str, Any]] = []
    blank_pages: list[int] = []
    for page_index, page in enumerate(reader.pages, 1):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            blank_pages.append(page_index)
        page_data: dict[str, Any] = {
            "art_code": art_code,
            "page_size": len(reader.pages),
            "notice_content": page_text,
            "pdf_page_index": page_index,
            "pdf_text_extractor": f"pypdf-{pypdf_version}",
        }
        if page_index == 1:
            page_data.update(
                {
                    "notice_title": verification_title,
                    "notice_date": notice_date,
                    "eitime": None,
                    "attach_url_web": attachment_url,
                    "attach_type": "0",
                    "content_source": "eastmoney_static_pdf_financial_tolerant",
                }
            )
        pages.append(page_data)
    if len(blank_pages) == len(pages):
        raise ValueError("financial evidence PDF has no extractable text")
    title_score = _pdf_title_match_score(verification_title, pages)
    if title_score < 0.8:
        raise ValueError("stored announcement title not found in first three PDF pages")
    combined = assemble_content_pages(pages, expected_pages=len(pages))
    combined["pdf_title_verified"] = True
    combined["pdf_title_match_score"] = round(title_score, 6)
    combined["pdf_text_extractor"] = f"pypdf-{pypdf_version}"
    combined["financial_pdf_blank_pages"] = blank_pages
    return combined


def archive_financial_revision_evidence(
    *,
    plan_path: Path,
    output_root: Path,
    state_path: Path,
    contract: FinancialRevisionEvidenceContract,
    rate_limit_per_minute: int = 30,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("contract") != FINANCIAL_REVISION_PLAN_CONTRACT:
        raise ValueError("unexpected financial revision plan")
    if plan.get("dataset_role") != "train" or plan.get("official_test_queried") is not False:
        raise PermissionError("financial revision plan is not train-only")
    if plan.get("revision_contract_sha256") != contract.source_sha256:
        raise ValueError("financial revision plan contract hash mismatch")
    plan_sha256 = sha256_file(plan_path)
    selection = {
        "plan_sha256": plan_sha256,
        "announcement_ids": len(plan["announcement_ids"]),
        "announcement_ids_sha256": hashlib.sha256(
            "\n".join(plan["announcement_ids"]).encode()
        ).hexdigest(),
    }
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if state.get("selection") not in (None, selection):
        raise ValueError("financial revision archive selection differs from this run")
    state.setdefault("started_at_utc", datetime.now(UTC).isoformat())
    state.update(
        {
            "contract": FINANCIAL_REVISION_BACKFILL_CONTRACT,
            "revision_contract_sha256": contract.source_sha256,
            "dataset_role": "train",
            "official_test_queried": False,
            "market_targets_queried": False,
            "selection": selection,
        }
    )
    expected = {}
    for group in plan["revision_groups"]:
        for version in group["versions"]:
            announcement_id = str(version["announcement_id"])
            identity = (
                version["announcement_title"], version["actual_disclosure_date"]
            )
            if announcement_id in expected and expected[announcement_id] != identity:
                raise ValueError("one announcement ID has conflicting expected identity")
            expected[announcement_id] = identity
    completed = dict(state.get("completed", {}))
    failures = dict(state.get("failures", {}))
    client = EastmoneyClient()
    client._min_interval = 60.0 / max(rate_limit_per_minute, 1)
    for index, announcement_id in enumerate(plan["announcement_ids"], 1):
        if announcement_id in completed:
            continue
        try:
            title, notice_date = expected[announcement_id]
            url = f"https://pdf.dfcfw.com/pdf/H2_{announcement_id}_1.pdf"
            response = None
            for attempt in range(1, 5):
                client._rate_limit()
                try:
                    response = client.session.get(url, timeout=90)
                    response.raise_for_status()
                    if not response.content.startswith(b"%PDF"):
                        raise ValueError("financial evidence response is not a PDF")
                    break
                except Exception:
                    if attempt >= 4:
                        raise
                    time.sleep(5 * attempt)
            assert response is not None
            pdf_relative = Path("raw") / "pdf" / f"{announcement_id}.pdf"
            pdf_sha256 = _write_bytes_atomic(
                output_root / pdf_relative, response.content
            )
            content = _assemble_financial_pdf(
                response.content,
                art_code=announcement_id,
                title=title,
                notice_date=notice_date,
                attachment_url=url,
            )
            text = str(content.get("notice_content") or "")
            if not text:
                raise ValueError("financial evidence PDF text is empty")
            content_record = {
                "announcement_id": announcement_id,
                "title": title,
                "notice_date": notice_date,
                "content_text": text,
                "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "pages_expected": content.get("_content_pages_expected"),
                "pages_fetched": content.get("_content_pages_fetched"),
                "content_complete": content.get("_content_complete"),
                "page_sha256": content.get("_content_page_sha256"),
                "pdf_title_verified": content.get("pdf_title_verified"),
                "pdf_title_match_score": content.get("pdf_title_match_score"),
                "pdf_text_extractor": content.get("pdf_text_extractor"),
                "financial_pdf_blank_pages": content.get(
                    "financial_pdf_blank_pages", []
                ),
                "evidence_status": "pdf_text_verified",
                "attachment_url": url,
                "pdf_sha256": pdf_sha256,
                "retrieved_at_utc": datetime.now(UTC).isoformat(),
                "dataset_role": "train",
            }
            content_relative = Path("raw") / "content" / f"{announcement_id}.json"
            content_file_sha256 = _save_json_atomic(
                output_root / content_relative, content_record
            )
            completed[announcement_id] = {
                "status": "archived",
                "title_sha256": hashlib.sha256(title.encode()).hexdigest(),
                "notice_date": notice_date,
                "pdf_path": pdf_relative.as_posix(),
                "pdf_bytes": len(response.content),
                "pdf_sha256": pdf_sha256,
                "content_path": content_relative.as_posix(),
                "content_chars": len(text),
                "content_sha256": content_record["content_sha256"],
                "content_file_sha256": content_file_sha256,
                "pages": content_record["pages_fetched"],
                "blank_pages": content_record["financial_pdf_blank_pages"],
            }
            failures.pop(announcement_id, None)
        except Exception as exc:  # pragma: no cover - live service behavior
            failures[announcement_id] = {
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
        _save_json_atomic(state_path, state)
        print(
            f"financial revision evidence {index}/{len(plan['announcement_ids'])}; "
            f"completed={len(completed)}; failures={len(failures)}",
            flush=True,
        )
    if not failures and len(completed) == len(plan["announcement_ids"]):
        state.setdefault("finished_at_utc", datetime.now(UTC).isoformat())
        _save_json_atomic(state_path, state)
    return state


def _numeric_match(value: object, content: str) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if pd.isna(number):
        return False
    compact = re.sub(r"[\s,，]", "", content).replace("−", "-").replace("—", "-")
    # Tushare returns yuan; Chinese reports frequently print the same table
    # in thousand/ten-thousand/million/hundred-million yuan.
    candidates: set[str] = set()
    for scale in (1.0, 1_000.0, 10_000.0, 1_000_000.0, 100_000_000.0):
        scaled = abs(number) / scale
        candidates.update({f"{scaled:.2f}", f"{scaled:.1f}", f"{scaled:g}"})
    if number < 0:
        return any(
            f"-{candidate}" in compact or f"({candidate})" in compact
            for candidate in candidates
        )
    return any(candidate in compact for candidate in candidates)


def audit_financial_revision_evidence(
    *,
    plan_path: Path,
    root: Path,
    state_path: Path,
    contract: FinancialRevisionEvidenceContract,
    source_contract: PointInTimeSourceContract,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if plan.get("contract") != FINANCIAL_REVISION_PLAN_CONTRACT:
        errors.append("unexpected financial revision plan")
    if plan.get("revision_contract_sha256") != contract.source_sha256:
        errors.append("plan revision contract hash mismatch")
    if plan.get("source_contract_sha256") != source_contract.source_sha256:
        errors.append("plan point-in-time source contract hash mismatch")
    if plan.get("official_test_queried") is not False:
        errors.append("financial revision plan touched official test")
    if plan.get("market_targets_queried") is not False:
        errors.append("financial revision plan touched market targets")
    if state.get("selection", {}).get("plan_sha256") != sha256_file(plan_path):
        errors.append("archive state plan hash mismatch")
    if state.get("revision_contract_sha256") != contract.source_sha256:
        errors.append("archive state revision contract hash mismatch")
    if state.get("official_test_queried") is not False:
        errors.append("archive state touched official test")
    if state.get("market_targets_queried") is not False:
        errors.append("archive state touched market targets")
    if state.get("failures"):
        errors.append("archive state contains failures")
    completed = state.get("completed", {})
    if set(completed) != set(plan["announcement_ids"]):
        errors.append("archived announcement IDs differ from plan")
    contents = {}
    unresolved_announcement_ids: list[str] = []
    for announcement_id, item in completed.items():
        try:
            pdf_path = (root / item["pdf_path"]).resolve()
            content_path = (root / item["content_path"]).resolve()
            pdf_path.relative_to(root.resolve())
            content_path.relative_to(root.resolve())
            if sha256_file(pdf_path) != item["pdf_sha256"]:
                raise ValueError("PDF SHA-256 mismatch")
            if sha256_file(content_path) != item["content_file_sha256"]:
                raise ValueError("content file SHA-256 mismatch")
            record = json.loads(content_path.read_text(encoding="utf-8"))
            text = str(record["content_text"])
            if hashlib.sha256(text.encode()).hexdigest() != item["content_sha256"]:
                raise ValueError("content SHA-256 mismatch")
            if record.get("announcement_id") != announcement_id:
                raise ValueError("content announcement identity mismatch")
            if record.get("evidence_status") == "pdf_unextractable_cdp_metadata":
                unresolved_announcement_ids.append(announcement_id)
                continue
            if record.get("pdf_title_verified") is not True:
                raise ValueError("content PDF title was not verified")
            if record.get("content_complete") is not True:
                raise ValueError("content PDF extraction is incomplete")
            contents[announcement_id] = text
        except Exception as exc:
            errors.append(f"{announcement_id}: {type(exc).__name__}: {exc}")
    group_results = []
    verified_by_endpoint = {endpoint: 0 for endpoint in source_contract.financial_fields}
    for group in plan["revision_groups"]:
        endpoint = group["endpoint"]
        identity_fields = set(source_contract.financial_identity_fields)
        metric_fields = [
            field
            for field in source_contract.financial_fields[endpoint]
            if field not in identity_fields
        ]
        derived_fields = set(contract.derived_metric_fields.get(endpoint, ()))
        changed = []
        derived_changed = []
        for field in metric_fields:
            values = [version["financial_values"].get(field) for version in group["versions"]]
            nonnull = [value for value in values if value is not None]
            if len(nonnull) >= 2 and len({str(value) for value in nonnull}) > 1:
                (derived_changed if field in derived_fields else changed).append(field)
        matches = {}
        for field in changed:
            field_matches = []
            for version in group["versions"]:
                value = version["financial_values"].get(field)
                text = contents.get(version["announcement_id"], "")
                field_matches.append(_numeric_match(value, text))
            matches[field] = field_matches
        verified = bool(changed) and all(all(values) for values in matches.values())
        verified_by_endpoint[endpoint] += int(verified)
        group_results.append(
            {
                "endpoint": endpoint,
                "identity": group["identity"],
                "identity_sha256": group["identity_sha256"],
                "changed_metric_fields": changed,
                "derived_changed_metric_fields": derived_changed,
                "derived_metric_evidence_status": (
                    "not_directly_reported_in_statement"
                    if derived_changed
                    else "not_applicable"
                ),
                "changed_metric_matches_by_version": matches,
                "verified": verified,
            }
        )
    verified_groups = sum(item["verified"] for item in group_results)
    evidence_threshold_passed = (
        verified_groups >= contract.minimum_verified_groups
        and all(
            value >= contract.minimum_verified_groups_per_endpoint
            for value in verified_by_endpoint.values()
        )
    )
    raw_integrity_passed = not errors
    feature_release_eligible = (
        raw_integrity_passed
        and evidence_threshold_passed
        and not unresolved_announcement_ids
    )
    release_blockers = [
        name
        for name, blocked in (
            ("raw_integrity_error", bool(errors)),
            ("insufficient_verified_revision_groups", not evidence_threshold_passed),
            (
                "unresolved_pdf_text_evidence",
                bool(unresolved_announcement_ids),
            ),
        )
        if blocked
    ]
    return {
        "contract": FINANCIAL_REVISION_AUDIT_CONTRACT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_role": "train",
        "official_test_queried": False,
        "market_targets_queried": False,
        "raw_integrity_passed": raw_integrity_passed,
        "evidence_threshold_passed": evidence_threshold_passed,
        "feature_release_eligible": feature_release_eligible,
        "release_blockers": release_blockers,
        "revision_contract_sha256": contract.source_sha256,
        "plan_sha256": sha256_file(plan_path),
        "state_sha256": sha256_file(state_path),
        "archived_announcements": len(completed),
        "selected_revision_groups": len(group_results),
        "verified_revision_groups": verified_groups,
        "verified_revision_groups_by_endpoint": verified_by_endpoint,
        "minimum_verified_revision_groups": contract.minimum_verified_groups,
        "minimum_verified_groups_per_endpoint": contract.minimum_verified_groups_per_endpoint,
        "provider_revision_documentation_status": (
            "official_documentation_plus_sampled_pdf_evidence"
        ),
        "provider_revision_documentation": list(
            contract.provider_documentation_urls
        ),
        "unresolved_announcement_ids": unresolved_announcement_ids,
        "group_results": group_results,
        "errors": errors,
    }

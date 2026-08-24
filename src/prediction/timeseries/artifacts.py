"""Build separate train-only daily-v1 feature and label artifacts."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.prediction.development_contract import load_causal_development_contract
from src.prediction.timeseries.alignment import (
    daily_bar_available_at,
    daily_v1_prediction_as_of,
    require_daily_v1_bar_usable,
)
from src.prediction.timeseries.contract import (
    TimeseriesModelContract,
    load_timeseries_model_contract,
)
from src.prediction.timeseries.coverage import _read_only_connection
from src.prediction.timeseries.features import (
    PRICE_COLUMNS,
    build_daily_feature_panel,
    summary_feature_names,
)


def _aware_utc(raw: object) -> datetime:
    value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _stable_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def _write_json_atomic(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_train_labels(
    connection: sqlite3.Connection,
    *,
    contract: TimeseriesModelContract,
    split_source_sha256: str,
) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT
            a.id,
            a.company_id,
            c.stock_code,
            a.published_date,
            x.needs_review,
            t.horizon_sessions,
            t.actual_direction,
            t.excess_return,
            t.outcome_available_at
        FROM announcement_market_targets AS t
        JOIN announcements AS a ON a.id = t.announcement_id
        JOIN companies AS c ON c.id = a.company_id
        JOIN classifications AS x ON x.announcement_id = a.id
        WHERE
            t.dataset_role = ?
            AND t.split_contract = ?
            AND t.split_source_sha256 = ?
        ORDER BY t.horizon_sessions, a.published_date, a.company_id, a.id
        """,
        ("train", contract.split_contract, split_source_sha256),
    ).fetchall()
    grouped: dict[tuple[int, str, str, int], dict[str, object]] = defaultdict(
        lambda: {
            "announcement_ids": [],
            "directions": set(),
            "excess_returns": [],
            "outcome_available_at": [],
            "has_accepted_classification": False,
        }
    )
    for row in rows:
        (
            announcement_id,
            company_id,
            stock_code,
            published_date,
            needs_review,
            horizon,
            actual_direction,
            excess_return,
            outcome_available_at,
        ) = row
        key = (int(company_id), str(stock_code), str(published_date), int(horizon))
        group = grouped[key]
        group["announcement_ids"].append(int(announcement_id))
        group["directions"].add(1 if int(actual_direction) > 0 else 0)
        group["excess_returns"].append(float(excess_return))
        group["outcome_available_at"].append(_aware_utc(outcome_available_at))
        group["has_accepted_classification"] |= not bool(needs_review)

    labels = []
    for (company_id, stock_code, raw_day, horizon), group in grouped.items():
        if not group["has_accepted_classification"]:
            continue
        directions = group["directions"]
        if len(directions) != 1:
            raise ValueError(
                "inconsistent company-day direction target: "
                f"{company_id}/{raw_day}/{horizon}"
            )
        returns = np.asarray(group["excess_returns"], dtype=float)
        if not np.allclose(returns, returns[0], rtol=0.0, atol=1e-12):
            raise ValueError(
                "inconsistent company-day excess return: "
                f"{company_id}/{raw_day}/{horizon}"
            )
        announcement_ids = sorted(set(group["announcement_ids"]))
        identity = {
            "contract": contract.contract,
            "split_source_sha256": split_source_sha256,
            "company_id": company_id,
            "stock_code": stock_code,
            "published_date": raw_day,
            "horizon_sessions": horizon,
            "announcement_ids": announcement_ids,
        }
        labels.append(
            {
                "sample_id": _stable_sha256(identity),
                "company_id": company_id,
                "stock_code": stock_code,
                "published_date": date.fromisoformat(raw_day),
                "horizon_sessions": horizon,
                "target": next(iter(directions)),
                "excess_return": float(returns[0]),
                "outcome_available_at": max(group["outcome_available_at"]),
                "dataset_role": "train",
                "announcement_ids": announcement_ids,
            }
        )
    return labels


def _load_price_frames(
    connection: sqlite3.Connection,
    *,
    instrument_codes: set[str],
    exclusive_end: date,
) -> tuple[dict[str, pd.DataFrame], str]:
    placeholders = ",".join("?" for _ in instrument_codes)
    parameters = [*sorted(instrument_codes), exclusive_end.isoformat()]
    rows = connection.execute(
        f"""
        SELECT
            instrument_code,
            trade_date,
            open_price,
            high_price,
            low_price,
            close_price,
            pre_close,
            volume,
            amount
        FROM daily_prices
        WHERE instrument_code IN ({placeholders}) AND trade_date < ?
        ORDER BY instrument_code, trade_date
        """,
        parameters,
    ).fetchall()
    digest = hashlib.sha256()
    grouped: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        code = str(row[0])
        values = tuple(row[1:])
        grouped[code].append(values)
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    frames = {
        code: pd.DataFrame(values, columns=PRICE_COLUMNS)
        for code, values in grouped.items()
    }
    return frames, digest.hexdigest()


def _require_output_under_data_boundary(
    output_directory: Path,
    *,
    contract: TimeseriesModelContract,
) -> Path:
    output = output_directory.resolve()
    if not output.is_relative_to(contract.data_directory):
        raise ValueError("time-series artifacts must remain under data/timeseries")
    return output


def build_train_artifacts(
    database_path: Path,
    output_directory: Path,
    *,
    contract: TimeseriesModelContract,
    force: bool = False,
) -> dict[str, object]:
    """Build causal train features and labels without touching sealed roles."""
    output = _require_output_under_data_boundary(
        output_directory,
        contract=contract,
    )
    feature_path = output / "features" / "train_daily_v1_features.csv"
    label_path = output / "labels" / "train_daily_v1_labels.csv"
    manifest_path = output / "manifests" / "train_daily_v1_manifest.json"
    targets = (feature_path, label_path, manifest_path)
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(f"train artifact already exists: {existing[0]}")

    development = load_causal_development_contract()
    split_source_sha256 = PROJECT_ROOT.joinpath(
        "config", "prediction_splits.yaml"
    ).read_bytes()
    split_source_sha256 = hashlib.sha256(split_source_sha256).hexdigest()
    with _read_only_connection(database_path) as connection:
        labels = _load_train_labels(
            connection,
            contract=contract,
            split_source_sha256=split_source_sha256,
        )
        instrument_codes = {str(item["stock_code"]) for item in labels}
        instrument_codes.add("000300.SH")
        price_frames, price_source_sha256 = _load_price_frames(
            connection,
            instrument_codes=instrument_codes,
            exclusive_end=contract.training_end + timedelta(days=1),
        )
    if not labels:
        raise ValueError("no eligible train labels")
    benchmark = price_frames.get("000300.SH")
    if benchmark is None or benchmark.empty:
        raise ValueError("CSI300 train bars are missing")

    panel_by_stock: dict[str, pd.DataFrame] = {}
    exclusions: defaultdict[str, int] = defaultdict(int)
    features = []
    included_labels = []
    grouped_labels: dict[tuple[str, date], list[dict[str, object]]] = defaultdict(list)
    for label in labels:
        grouped_labels[(str(label["stock_code"]), label["published_date"])].append(label)

    feature_names = summary_feature_names(contract)
    for (stock_code, published_date), event_labels in sorted(
        grouped_labels.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        stock_bars = price_frames.get(stock_code)
        if stock_bars is None or stock_bars.empty:
            exclusions["missing_stock_bars"] += len(event_labels)
            continue
        if stock_code not in panel_by_stock:
            panel_by_stock[stock_code] = build_daily_feature_panel(
                stock_bars=stock_bars,
                benchmark_bars=benchmark,
                contract=contract,
            )
        panel = panel_by_stock[stock_code]
        panel_dates = panel["trade_date"].tolist()
        endpoint_index = bisect.bisect_left(panel_dates, published_date) - 1
        if endpoint_index < 0:
            exclusions["no_prepublication_market_session"] += len(event_labels)
            continue
        endpoint = panel.iloc[endpoint_index]
        if int(endpoint["calendar_history_sessions"]) < contract.minimum_history_sessions:
            exclusions["insufficient_market_history"] += len(event_labels)
            continue
        if (
            int(endpoint["valid_stock_sessions_max_window"])
            < contract.minimum_valid_stock_sessions
        ):
            exclusions["insufficient_valid_stock_sessions"] += len(event_labels)
            continue
        prediction_as_of = daily_v1_prediction_as_of(
            published_date,
            development=development,
        )
        last_bar_trade_date = endpoint["trade_date"]
        last_bar_available_at = daily_bar_available_at(
            last_bar_trade_date,
            contract=contract,
            development=development,
        )
        require_daily_v1_bar_usable(
            trade_date=last_bar_trade_date,
            published_date=published_date,
            prediction_as_of=prediction_as_of,
            bar_available_at=last_bar_available_at,
            development=development,
        )
        for label in event_labels:
            features.append(
                {
                    "sample_id": label["sample_id"],
                    "stock_code": stock_code,
                    "published_date": published_date,
                    "prediction_as_of": prediction_as_of,
                    "last_bar_trade_date": last_bar_trade_date,
                    "last_bar_available_at": last_bar_available_at,
                    "dataset_role": "train",
                    **{name: endpoint[name] for name in feature_names},
                }
            )
            included_labels.append(
                {
                    key: label[key]
                    for key in (
                        "sample_id",
                        "horizon_sessions",
                        "target",
                        "excess_return",
                        "outcome_available_at",
                        "dataset_role",
                    )
                }
            )

    feature_frame = pd.DataFrame(features)
    label_frame = pd.DataFrame(included_labels)
    if feature_frame.empty or label_frame.empty:
        raise ValueError("no time-series train artifacts survived causal eligibility")
    feature_frame = feature_frame.sort_values(
        ["published_date", "stock_code", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)
    label_frame = label_frame.sort_values(
        ["horizon_sessions", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)
    if set(feature_frame["sample_id"]) != set(label_frame["sample_id"]):
        raise ValueError("feature and label sample identities differ before write")

    _write_csv_atomic(feature_frame, feature_path)
    _write_csv_atomic(label_frame, label_path)
    normalized_label_source = [
        {
            "sample_id": item["sample_id"],
            "horizon_sessions": item["horizon_sessions"],
            "target": item["target"],
            "excess_return": item["excess_return"],
            "outcome_available_at": item["outcome_available_at"].isoformat(),
        }
        for item in sorted(labels, key=lambda item: str(item["sample_id"]))
    ]
    manifest = {
        "artifact": "causal-timeseries-daily-v1-train",
        "created_at": datetime.now(UTC).isoformat(),
        "contract": contract.contract,
        "contract_sha256": contract.source_sha256,
        "split_contract": contract.split_contract,
        "split_source_sha256": split_source_sha256,
        "dataset_role_read": "train",
        "official_test_read": False,
        "quarantine_read": False,
        "forward_validation_read": False,
        "feature_label_storage": "physically_separate_csv",
        "input_train_samples": len(labels),
        "included_samples": len(feature_frame),
        "excluded_samples": len(labels) - len(feature_frame),
        "exclusions": dict(sorted(exclusions.items())),
        "feature_columns": list(feature_names),
        "features_file": feature_path.relative_to(output).as_posix(),
        "features_sha256": _file_sha256(feature_path),
        "labels_file": label_path.relative_to(output).as_posix(),
        "labels_sha256": _file_sha256(label_path),
        "train_label_source_sha256": _stable_sha256(normalized_label_source),
        "train_price_source_sha256": price_source_sha256,
    }
    _write_json_atomic(manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "ddos.db",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "timeseries" / "daily_v1",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    contract = load_timeseries_model_contract()
    manifest = build_train_artifacts(
        args.database,
        args.output_dir,
        contract=contract,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

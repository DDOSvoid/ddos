#!/usr/bin/env python
"""冻结分类器并建立不可回看的时间外人工验证批次。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import subprocess
import sys
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.training.label_contract import SCHEMA_VERSION, validate_label_item
from src.utils.text_utils import clean_chinese_text

CONTRACT = "temporal-classification-v1"
FROZEN_FILES = (
    "src/pipeline/classification_rules.py",
    "config/event_types.yaml",
    "src/ml/classifier_wrapper.py",
    "src/pipeline/classifier.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def freeze(db_path: Path, output_dir: Path) -> Path:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"冻结批次已存在，拒绝覆盖: {manifest_path}")

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT MAX(published_date), COUNT(*) FROM announcements"
        ).fetchone()
    if not row or not row[0]:
        raise ValueError("数据库没有可用于冻结的公告")

    cutoff = date.fromisoformat(row[0])
    file_hashes = {
        name: _sha256(PROJECT_ROOT / name)
        for name in FROZEN_FILES
    }
    manifest = {
        "contract": CONTRACT,
        "batch_id": output_dir.name,
        "status": "frozen",
        "frozen_at": datetime.now(UTC).isoformat(),
        "database": str(db_path.resolve()),
        "data_cutoff": cutoff.isoformat(),
        "validation_start": (cutoff + timedelta(days=1)).isoformat(),
        "announcements_at_freeze": int(row[1]),
        "git_head": _git_head(),
        "frozen_file_sha256": file_hashes,
        "rules_must_not_change_before_prediction_lock": True,
    }
    _write_json(manifest_path, manifest)
    print(f"冻结完成: {manifest_path}")
    print(f"数据截止: {manifest['data_cutoff']}; 时间外起点: {manifest['validation_start']}")
    return manifest_path


def _verify_frozen_files(manifest: dict) -> None:
    changed = []
    for name, expected in manifest["frozen_file_sha256"].items():
        actual = _sha256(PROJECT_ROOT / name)
        if actual != expected:
            changed.append(name)
    if changed:
        raise ValueError("冻结后分类代码发生变化，拒绝锁定预测: " + ", ".join(changed))


def lock_predictions(db_path: Path, output_dir: Path, end_date: date) -> int:
    manifest_path = output_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "frozen":
        raise ValueError(f"批次状态不是 frozen: {manifest.get('status')}")
    _verify_frozen_files(manifest)

    prediction_path = output_dir / "predictions_locked.jsonl"
    blind_path = output_dir / "human_review_blind.jsonl"
    if prediction_path.exists() or blind_path.exists():
        raise FileExistsError("预测或盲审文件已存在，拒绝覆盖")

    start_date = date.fromisoformat(manifest["validation_start"])
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = list(
            connection.execute(
                """
                SELECT a.announcement_id, a.published_date, a.title, a.full_text,
                       c.major_category, c.sub_category, c.confidence,
                       c.classification_source, c.rule_id, c.document_type,
                       c.relevance, c.needs_review
                FROM announcements a
                JOIN classifications c ON c.announcement_id = a.id
                WHERE a.published_date >= ? AND a.published_date <= ?
                ORDER BY a.published_date, a.announcement_id
                """,
                (start_date.isoformat(), end_date.isoformat()),
            )
        )
    if not rows:
        raise ValueError(f"{start_date} 至 {end_date} 没有已分类公告")

    with (
        open(prediction_path, "w", encoding="utf-8") as predictions,
        open(blind_path, "w", encoding="utf-8") as blind,
    ):
        for row in rows:
            text = f"{row['title'] or ''}\n{row['full_text'] or ''}".strip()
            prediction = {
                "contract": CONTRACT,
                "announcement_id": row["announcement_id"],
                "published_at": row["published_date"],
                "predicted_major_category": row["major_category"],
                "predicted_sub_category": row["sub_category"],
                "confidence": row["confidence"],
                "classification_source": row["classification_source"],
                "rule_id": row["rule_id"],
                "document_type": row["document_type"],
                "relevance": row["relevance"],
                "abstained": bool(row["needs_review"]),
            }
            review_item = {
                "schema_version": SCHEMA_VERSION,
                "announcement_id": row["announcement_id"],
                "published_at": row["published_date"],
                "text": text,
                "major_category": "",
                "sub_category": "",
                "label_source": "human",
                "label_status": "candidate",
                "labeled_at": "",
                "reviewer": "",
                "dataset_role": "test",
                "evidence": "",
            }
            predictions.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            blind.write(json.dumps(review_item, ensure_ascii=False) + "\n")

    manifest.update(
        {
            "status": "predictions_locked",
            "validation_end": end_date.isoformat(),
            "prediction_count": len(rows),
            "predictions_locked_at": datetime.now(UTC).isoformat(),
            "predictions_sha256": _sha256(prediction_path),
            "blind_review_sha256_at_creation": _sha256(blind_path),
        }
    )
    _write_json(manifest_path, manifest)
    print(f"已锁定 {len(rows)} 条预测: {prediction_path}")
    print(f"人工盲审文件: {blind_path}")
    return len(rows)


def enrich_blind_review(
    db_path: Path,
    output_dir: Path,
    rate_limit_per_minute: int,
    output_name: str,
) -> int:
    """预测锁定后补抓正文，另存盲审文件；绝不改写锁定预测。"""
    from src.config import config
    from src.pipeline.fetcher import EastmoneyClient

    manifest_path = output_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "predictions_locked":
        raise ValueError("必须先锁定预测，才能补抓盲审正文")

    source_path = output_dir / "human_review_blind.jsonl"
    output_path = output_dir / output_name
    if output_path.exists():
        raise FileExistsError(f"正文增强盲审文件已存在，拒绝覆盖: {output_path}")

    with open(source_path, encoding="utf-8") as file:
        items = [json.loads(line) for line in file if line.strip()]
    ids = [str(item["announcement_id"]) for item in items]
    placeholders = ",".join("?" for _ in ids)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"SELECT announcement_id, title, full_text FROM announcements "
            f"WHERE announcement_id IN ({placeholders})",
            ids,
        ).fetchall()
    current = {str(row["announcement_id"]): row for row in rows}

    config.eastmoney.rate_limit_per_minute = rate_limit_per_minute
    client = EastmoneyClient()
    fetched = 0
    failed = 0
    with sqlite3.connect(db_path) as connection:
        for index, item in enumerate(items, 1):
            announcement_id = str(item["announcement_id"])
            row = current.get(announcement_id)
            title = row["title"] if row else item["text"].splitlines()[0]
            body = row["full_text"] if row else None
            body_is_placeholder = (
                not body
                or body.strip() == (title or "").strip()
                or len(body.strip()) < 100
            )
            if body_is_placeholder:
                content = client.fetch_announcement_content(announcement_id)
                body = clean_chinese_text(content.get("notice_content", ""))
                if body:
                    connection.execute(
                        "UPDATE announcements SET full_text = ?, pdf_url = COALESCE(?, pdf_url) "
                        "WHERE announcement_id = ?",
                        (
                            body,
                            content.get("attach_url_web") or content.get("attach_url"),
                            announcement_id,
                        ),
                    )
                    fetched += 1
                else:
                    failed += 1
            item["text"] = f"{title or ''}\n{body or ''}".strip()
            if index % 50 == 0 or index == len(items):
                print(f"正文补抓进度 {index}/{len(items)}; fetched={fetched}; failed={failed}")
        connection.commit()

    with open(output_path, "w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest.update(
        {
            "blind_review_enriched_at": datetime.now(UTC).isoformat(),
            "blind_review_enriched_sha256": _sha256(output_path),
            "blind_review_full_text_fetched": fetched,
            "blind_review_full_text_failed": failed,
        }
    )
    _write_json(manifest_path, manifest)
    print(f"正文增强盲审文件: {output_path}")
    return fetched


def prepare_review_batches(output_dir: Path, accuracy_sample_size: int) -> dict:
    """Create prediction-blind review batches without changing locked predictions."""
    manifest_path = output_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    prediction_path = output_dir / "predictions_locked.jsonl"
    blind_path = output_dir / "human_review_blind.jsonl"
    if manifest.get("status") != "predictions_locked":
        raise ValueError("predictions must be locked before preparing review batches")
    if _sha256(prediction_path) != manifest.get("predictions_sha256"):
        raise ValueError("locked prediction hash mismatch")

    predictions = {
        str(item["announcement_id"]): item for item in _read_jsonl(prediction_path)
    }
    blind_items = {
        str(item["announcement_id"]): item for item in _read_jsonl(blind_path)
    }
    if set(predictions) != set(blind_items):
        raise ValueError("locked predictions and blind-review items do not match")

    accepted_ids = [key for key, item in predictions.items() if not item["abstained"]]
    abstained_ids = [key for key, item in predictions.items() if item["abstained"]]
    if accuracy_sample_size <= 0 or accuracy_sample_size > len(accepted_ids):
        raise ValueError(
            f"accuracy sample size must be between 1 and {len(accepted_ids)}"
        )

    # Hash ordering is deterministic, auditable, and independent of category/prediction.
    sample_salt = f"{manifest['batch_id']}:accepted-accuracy-v1"
    accepted_ids.sort(
        key=lambda key: hashlib.sha256(f"{sample_salt}:{key}".encode()).hexdigest()
    )
    accuracy_ids = accepted_ids[:accuracy_sample_size]
    abstained_ids.sort()

    outputs = {
        "accuracy": output_dir / "human_review_accuracy_sample.jsonl",
        "coverage_gaps": output_dir / "human_review_coverage_gaps.jsonl",
    }
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"review batch already exists; refusing overwrite: {path}")

    for group, ids in (("accuracy", accuracy_ids), ("coverage_gaps", abstained_ids)):
        with open(outputs[group], "w", encoding="utf-8") as file:
            for announcement_id in ids:
                file.write(
                    json.dumps(blind_items[announcement_id], ensure_ascii=False) + "\n"
                )

    batch_manifest = {
        "contract": CONTRACT,
        "batch_id": manifest["batch_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "selection": {
            "accuracy": {
                "population": "accepted predictions only",
                "population_size": len(accepted_ids),
                "sample_size": len(accuracy_ids),
                "method": "ascending SHA-256 hash order",
                "salt": sample_salt,
                "purpose": "estimate selective accuracy",
                "sha256": _sha256(outputs["accuracy"]),
            },
            "coverage_gaps": {
                "population": "all abstained predictions",
                "population_size": len(abstained_ids),
                "sample_size": len(abstained_ids),
                "method": "complete census",
                "purpose": "diagnose taxonomy/rule coverage; never estimate accuracy",
                "sha256": _sha256(outputs["coverage_gaps"]),
            },
        },
    }
    batch_manifest_path = output_dir / "review_batches_manifest.json"
    if batch_manifest_path.exists():
        raise FileExistsError(
            f"review batch manifest already exists; refusing overwrite: {batch_manifest_path}"
        )
    _write_json(batch_manifest_path, batch_manifest)
    print(f"accuracy sample: {outputs['accuracy']} ({len(accuracy_ids)} items)")
    print(f"coverage gaps: {outputs['coverage_gaps']} ({len(abstained_ids)} items)")
    return batch_manifest


def _wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = correct / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def evaluate_review_batch(output_dir: Path, labels_path: Path, batch: str) -> dict:
    """Evaluate a completed human batch without treating AI candidates as truth."""
    if batch not in {"accuracy", "coverage_gaps"}:
        raise ValueError("batch must be accuracy or coverage_gaps")
    manifest = _read_json(output_dir / "manifest.json")
    review_manifest = _read_json(output_dir / "review_batches_manifest.json")
    prediction_path = output_dir / "predictions_locked.jsonl"
    if _sha256(prediction_path) != manifest.get("predictions_sha256"):
        raise ValueError("locked prediction hash mismatch")

    review_filename = {
        "accuracy": "human_review_accuracy_sample.jsonl",
        "coverage_gaps": "human_review_coverage_gaps.jsonl",
    }[batch]
    review_path = output_dir / review_filename
    expected_hash = review_manifest["selection"][batch]["sha256"]
    if _sha256(review_path) != expected_hash:
        raise ValueError(f"{batch} review-batch hash mismatch")

    expected_ids = {
        str(item["announcement_id"]) for item in _read_jsonl(review_path)
    }
    predictions = {
        str(item["announcement_id"]): item for item in _read_jsonl(prediction_path)
    }
    labels: dict[str, dict] = {}
    for line_number, item in enumerate(_read_jsonl(labels_path), 1):
        validate_label_item(item, production=True)
        announcement_id = str(item["announcement_id"])
        if announcement_id in labels:
            raise ValueError(f"duplicate announcement_id on line {line_number}")
        labels[announcement_id] = item
    missing = sorted(expected_ids - set(labels))
    extra = sorted(set(labels) - expected_ids)
    if missing or extra:
        raise ValueError(
            f"review labels do not match {batch} batch: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    report: dict = {
        "contract": CONTRACT,
        "batch_id": manifest["batch_id"],
        "review_batch": batch,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "samples": len(labels),
        "label_source": "verified human labels only",
        "labels_sha256": _sha256(labels_path),
    }
    if batch == "accuracy":
        sub_correct = sum(
            predictions[key]["predicted_sub_category"] == item["sub_category"]
            for key, item in labels.items()
        )
        major_correct = sum(
            predictions[key]["predicted_major_category"] == item["major_category"]
            for key, item in labels.items()
        )
        sub_low, sub_high = _wilson_interval(sub_correct, len(labels))
        major_low, major_high = _wilson_interval(major_correct, len(labels))
        report.update(
            {
                "subcategory_correct": sub_correct,
                "subcategory_accuracy": sub_correct / len(labels),
                "subcategory_accuracy_wilson_95pct": [sub_low, sub_high],
                "major_category_correct": major_correct,
                "major_category_accuracy": major_correct / len(labels),
                "major_category_accuracy_wilson_95pct": [major_low, major_high],
                "errors": [
                    {
                        "announcement_id": key,
                        "predicted_major_category": predictions[key][
                            "predicted_major_category"
                        ],
                        "predicted_sub_category": predictions[key][
                            "predicted_sub_category"
                        ],
                        "actual_major_category": item["major_category"],
                        "actual_sub_category": item["sub_category"],
                        "rule_id": predictions[key]["rule_id"],
                        "evidence": item.get("evidence", ""),
                    }
                    for key, item in labels.items()
                    if predictions[key]["predicted_sub_category"]
                    != item["sub_category"]
                ],
            }
        )
        output_path = output_dir / "evaluation_accuracy_sample.json"
    else:
        report.update(
            {
                "purpose": "taxonomy and coverage diagnosis; not an accuracy estimate",
                "actual_category_counts": dict(
                    Counter(item["sub_category"] for item in labels.values())
                ),
            }
        )
        output_path = output_dir / "evaluation_coverage_gaps.json"

    _write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def fill_ai_review_candidates(output_dir: Path) -> dict:
    """Fill review batches as auditable AI candidates, never as human truth."""
    accuracy_path = output_dir / "human_review_accuracy_sample.jsonl"
    gaps_path = output_dir / "human_review_coverage_gaps.jsonl"
    prediction_path = output_dir / "predictions_locked.jsonl"
    abstained_candidates_path = output_dir / "ai_review_abstained.jsonl"
    overrides_path = output_dir / "ai_review_accuracy_overrides.json"
    outputs = {
        "accuracy": output_dir / "ai_review_accuracy_sample_filled.jsonl",
        "coverage_gaps": output_dir / "ai_review_coverage_gaps_filled.jsonl",
    }
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"AI review output already exists: {path}")

    predictions = {
        str(item["announcement_id"]): item for item in _read_jsonl(prediction_path)
    }
    accuracy_items = _read_jsonl(accuracy_path)
    gap_items = _read_jsonl(gaps_path)
    gap_candidates = {
        str(item["announcement_id"]): item
        for item in _read_jsonl(abstained_candidates_path)
    }
    overrides = _read_json(overrides_path)
    labeled_at = datetime.now(UTC).isoformat()

    accuracy_filled = []
    for item in accuracy_items:
        announcement_id = str(item["announcement_id"])
        prediction = predictions[announcement_id]
        decision = overrides.get(announcement_id)
        if decision:
            major = decision["major_category"]
            sub = decision["sub_category"]
            evidence = decision["evidence"]
        else:
            major = prediction["predicted_major_category"]
            sub = prediction["predicted_sub_category"]
            evidence = (
                "AI逐条语义初审：标题及锁定时可见正文与该预测标签一致；"
                "仍需真人独立复核。"
            )
        filled = {
            **item,
            "major_category": major,
            "sub_category": sub,
            "label_source": "ai_candidate",
            "label_status": "candidate",
            "labeled_at": labeled_at,
            "reviewer": "Codex AI semantic audit",
            "evidence": evidence,
        }
        validate_label_item(filled, production=False)
        accuracy_filled.append(filled)

    gaps_filled = []
    for item in gap_items:
        announcement_id = str(item["announcement_id"])
        candidate = gap_candidates.get(announcement_id)
        if not candidate:
            raise ValueError(f"missing AI gap candidate: {announcement_id}")
        filled = {
            **item,
            "major_category": candidate["ai_major_category"],
            "sub_category": candidate["ai_sub_category"],
            "label_source": "ai_candidate",
            "label_status": "candidate",
            "labeled_at": labeled_at,
            "reviewer": "Codex AI semantic audit",
            "evidence": candidate["evidence"],
        }
        validate_label_item(filled, production=False)
        gaps_filled.append(filled)

    for group, items in (("accuracy", accuracy_filled), ("coverage_gaps", gaps_filled)):
        with open(outputs[group], "w", encoding="utf-8") as file:
            for item in items:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "created_at": labeled_at,
        "label_source": "ai_candidate",
        "label_status": "candidate",
        "production_eligible": False,
        "accuracy_items": len(accuracy_filled),
        "accuracy_overrides": len(overrides),
        "coverage_gap_items": len(gaps_filled),
        "accuracy_sha256": _sha256(outputs["accuracy"]),
        "coverage_gaps_sha256": _sha256(outputs["coverage_gaps"]),
    }
    _write_json(output_dir / "ai_review_filled_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def evaluate(output_dir: Path, labels_path: Path) -> dict:
    manifest = _read_json(output_dir / "manifest.json")
    prediction_path = output_dir / "predictions_locked.jsonl"
    if _sha256(prediction_path) != manifest.get("predictions_sha256"):
        raise ValueError("锁定预测文件哈希不一致，拒绝评估")

    predictions = {}
    with open(prediction_path, encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            predictions[str(item["announcement_id"])] = item

    labels = {}
    with open(labels_path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            item = json.loads(line)
            validate_label_item(item, production=True)
            announcement_id = str(item["announcement_id"])
            if announcement_id not in predictions:
                raise ValueError(f"第 {line_number} 行不属于本时间外批次")
            labels[announcement_id] = item
    missing = sorted(set(predictions) - set(labels))
    if missing:
        raise ValueError(f"还有 {len(missing)} 条预测未完成人工盲审")

    accepted = [item for item in predictions.values() if not item["abstained"]]
    sub_correct = sum(
        item["predicted_sub_category"] == labels[key]["sub_category"]
        for key, item in predictions.items()
        if not item["abstained"]
    )
    major_correct = sum(
        item["predicted_major_category"] == labels[key]["major_category"]
        for key, item in predictions.items()
        if not item["abstained"]
    )
    errors = [
        {
            "announcement_id": key,
            "predicted": item["predicted_sub_category"],
            "actual": labels[key]["sub_category"],
            "rule_id": item["rule_id"],
            "evidence": labels[key].get("evidence", ""),
        }
        for key, item in predictions.items()
        if not item["abstained"]
        and item["predicted_sub_category"] != labels[key]["sub_category"]
    ]
    report = {
        "contract": CONTRACT,
        "batch_id": manifest["batch_id"],
        "evaluated_at": datetime.now(UTC).isoformat(),
        "samples": len(predictions),
        "accepted": len(accepted),
        "coverage": len(accepted) / len(predictions),
        "subcategory_accuracy": sub_correct / len(accepted) if accepted else 0.0,
        "major_category_accuracy": major_correct / len(accepted) if accepted else 0.0,
        "actual_category_counts": dict(Counter(item["sub_category"] for item in labels.values())),
        "errors": errors,
        "labels_sha256": _sha256(labels_path),
    }
    _write_json(output_dir / "evaluation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="时间外分类人工验证")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "ddos.db")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--output-dir", type=Path, required=True)

    lock_parser = subparsers.add_parser("lock")
    lock_parser.add_argument("--output-dir", type=Path, required=True)
    lock_parser.add_argument("--end-date", type=date.fromisoformat, required=True)

    enrich_parser = subparsers.add_parser("enrich")
    enrich_parser.add_argument("--output-dir", type=Path, required=True)
    enrich_parser.add_argument("--rate-limit", type=int, default=120)
    enrich_parser.add_argument(
        "--output-name",
        default="human_review_blind_enriched.jsonl",
    )

    batches_parser = subparsers.add_parser("prepare-review-batches")
    batches_parser.add_argument("--output-dir", type=Path, required=True)
    batches_parser.add_argument("--accuracy-sample-size", type=int, default=120)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--labels", type=Path, required=True)

    batch_evaluate_parser = subparsers.add_parser("evaluate-review-batch")
    batch_evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    batch_evaluate_parser.add_argument("--labels", type=Path, required=True)
    batch_evaluate_parser.add_argument(
        "--batch",
        choices=("accuracy", "coverage_gaps"),
        required=True,
    )

    ai_fill_parser = subparsers.add_parser("fill-ai-review")
    ai_fill_parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    db_path = args.db.resolve()
    output_dir = args.output_dir.resolve()
    if args.command == "freeze":
        freeze(db_path, output_dir)
    elif args.command == "lock":
        lock_predictions(db_path, output_dir, args.end_date)
    elif args.command == "enrich":
        enrich_blind_review(
            db_path,
            output_dir,
            max(args.rate_limit, 1),
            args.output_name,
        )
    elif args.command == "prepare-review-batches":
        prepare_review_batches(output_dir, args.accuracy_sample_size)
    elif args.command == "evaluate-review-batch":
        evaluate_review_batch(output_dir, args.labels.resolve(), args.batch)
    elif args.command == "fill-ai-review":
        fill_ai_review_candidates(output_dir)
    else:
        evaluate(output_dir, args.labels.resolve())


if __name__ == "__main__":
    main()

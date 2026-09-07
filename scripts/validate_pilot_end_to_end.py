"""Real-data wiring diagnostic, NOT qualification of the production experts.

Uses only cached pre-2025 bars and frozen pilot announcement metadata. All
preprocessors are fitted inside temporal folds; stacking uses earlier OOF only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.prediction.fusion.baselines import run_equal_weight_baseline  # noqa: E402
from src.prediction.fusion.contract import load_expert_fusion_contract  # noqa: E402
from src.prediction.fusion.ranking import (  # noqa: E402
    evaluate_daily_ranking,
    rank_company_day_predictions,
)
from src.prediction.fusion.schema import IDENTITY_COLUMNS, build_joint_expert_frame  # noqa: E402


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def build_samples(database, plan):
    ids = plan["announcement_ids"]
    if plan.get("dataset_role") != "train" or plan.get("market_targets_queried") is not False:
        raise PermissionError("Need a train-only outcome-blind selection plan")
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as con:
        anns = pd.read_sql_query(
            "SELECT a.announcement_id,a.published_date,a.title,co.stock_code,c.major_category "
            "FROM announcements a JOIN companies co ON co.id=a.company_id "
            "JOIN classifications c ON c.announcement_id=a.id "
            "WHERE a.published_date BETWEEN '2023-01-01' AND '2024-12-31' "
            "AND c.needs_review=0 AND c.relevance='core_event'",
            con,
        )
        anns = anns[anns.announcement_id.isin(ids)].copy()
        bars = pd.read_sql_query(
            "SELECT instrument_code,trade_date,open_price,high_price,low_price,close_price,"
            "volume,amount FROM daily_prices "
            "WHERE trade_date BETWEEN '2022-09-01' AND '2024-12-31' "
            "ORDER BY instrument_code,trade_date",
            con,
        )
    prices = {code: df.reset_index(drop=True) for code, df in bars.groupby("instrument_code")}
    benchmark = prices["000300.SH"].set_index("trade_date")
    anns["entry"] = None
    for code, rows in anns.groupby("stock_code"):
        if code not in prices:
            continue
        ds = prices[code].trade_date.to_numpy()
        idx = np.searchsorted(ds, rows.published_date.to_numpy(), side="right")
        valid = idx < len(ds)
        anns.loc[rows.index[valid], "entry"] = ds[idx[valid]]
    features, labels = [], []
    for (code, entry), group in anns.dropna(subset=["entry"]).groupby(["stock_code", "entry"]):
        # Several publication dates can map to the same tradable open.
        pub = group.published_date.max()
        history = prices[code][prices[code].trade_date < group.published_date.min()]
        if len(history) < 31:
            continue
        future = prices[code][prices[code].trade_date >= entry]
        daily_returns = history.close_price.pct_change().iloc[-20:].to_numpy()
        if not np.isfinite(daily_returns).all():
            continue
        last = history.iloc[-1]
        tab = [
            len(group),
            np.log1p(max(float(last.amount or 0), 0)),
            (last.high_price - last.low_price) / last.close_price,
        ]
        tab += [int((group.major_category == cat).sum()) for cat in "ABCDEFG"]
        if not np.isfinite(tab).all():
            continue
        title = "\n".join(group.sort_values("announcement_id").title)
        evidence = digest(title + json.dumps(tab) + json.dumps(daily_returns.tolist()))
        for horizon in (1, 3, 5):
            if len(future) < horizon:
                continue
            end = future.iloc[horizon - 1]
            if entry not in benchmark.index or end.trade_date not in benchmark.index:
                continue
            sid = digest(f"{code}:{entry}:{horizon}")
            base = dict(
                sample_id=sid,
                company_day_id=digest(f"{code}:{entry}"),
                stock_code=code,
                published_date=pub,
                eligible_entry_date=entry,
                prediction_as_of=pd.Timestamp(entry, tz="Asia/Shanghai")
                + pd.Timedelta(hours=8, minutes=30),
                horizon_sessions=horizon,
                data_as_of=pd.Timestamp(pub, tz="Asia/Shanghai")
                + pd.Timedelta(hours=23, minutes=59),
                evidence_sha256=evidence,
                title=title,
            )
            base.update({f"tab_{i}": value for i, value in enumerate(tab)})
            base.update({f"seq_{i}": value for i, value in enumerate(daily_returns)})
            features.append(base)
            target = end.close_price / future.iloc[0].open_price - 1
            target -= (
                benchmark.loc[end.trade_date].close_price / benchmark.loc[entry].open_price - 1
            )
            labels.append(
                dict(
                    sample_id=sid,
                    future_excess_return=target,
                    outcome_date=end.trade_date,
                    dataset_role="train",
                    published_date=pub,
                )
            )
    return pd.DataFrame(features), pd.DataFrame(labels), len(anns)


def run(database, plan_path, output):
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    features, labels, planned_resolved = build_samples(database, plan)
    if features.empty:
        raise ValueError("No real cached samples available")
    features.to_parquet(output / "features.parquet", index=False)
    labels.to_parquet(output / "labels.parquet", index=False)
    data = features.merge(
        labels[["sample_id", "future_excess_return", "outcome_date"]],
        on="sample_id",
        validate="one_to_one",
    )
    contract = load_expert_fusion_contract()
    components = {name: [] for name in contract.expert_names}
    fold_audit = []
    folds = [
        ("2023-07-01", "2023-12-31"),
        ("2024-01-01", "2024-06-30"),
        ("2024-07-01", "2024-12-31"),
    ]
    for horizon in (1, 3, 5):
        horizon_data = data[data.horizon_sessions == horizon]
        for start, end in folds:
            train = horizon_data[
                (horizon_data.eligible_entry_date < start) & (horizon_data.outcome_date < start)
            ]
            valid = horizon_data[
                horizon_data.eligible_entry_date.between(start, end)
                & (horizon_data.outcome_date <= end)
            ]
            if len(train) < 20 or valid.empty:
                continue
            fold_audit.append(
                dict(
                    horizon=horizon,
                    start=start,
                    end=end,
                    train_rows=len(train),
                    validation_rows=len(valid),
                    latest_training_outcome=train.outcome_date.max(),
                )
            )
            for name in components:
                if name == "text":
                    model = make_pipeline(
                        TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=3000),
                        Ridge(alpha=10, solver="lsqr"),
                    )
                    xtrain, xvalid = train.title, valid.title
                else:
                    prefix = "tab_" if name == "tabular" else "seq_"
                    columns = [c for c in features if c.startswith(prefix)]
                    model = make_pipeline(StandardScaler(), Ridge(alpha=10))
                    xtrain, xvalid = train[columns], valid[columns]
                model.fit(xtrain, train.future_excess_return)
                predicted = model.predict(xvalid)
                frame = valid[IDENTITY_COLUMNS + ["data_as_of", "evidence_sha256"]].copy()
                frame["expert_signal"] = frame["expected_excess_return"] = predicted
                frame["risk_scale"] = float(train.future_excess_return.abs().mean())
                frame["quality_score"] = 1.0
                frame["available"] = True
                frame["abstain_reason"] = None
                frame["expert_version"] = f"wiring-only-{name}-ridge-v1"
                frame["fold_start"] = start
                components[name].append(frame)
    if not all(components.values()):
        raise ValueError("Insufficient history for real temporal OOF")
    components = {name: pd.concat(frames, ignore_index=True) for name, frames in components.items()}
    joint = build_joint_expert_frame(components, contract=contract)
    joint.to_parquet(output / "joint_oof.parquet", index=False)
    for name, frame in components.items():
        frame.to_parquet(output / f"{name}_oof.parquet", index=False)
    predictions = {"equal_weight": run_equal_weight_baseline(joint, contract=contract)}
    for name in components:
        predictions[name] = run_equal_weight_baseline(
            build_joint_expert_frame({name: components[name]}, contract=contract), contract=contract
        )
        others = {k: v for k, v in components.items() if k != name}
        predictions[f"drop_{name}"] = run_equal_weight_baseline(
            build_joint_expert_frame(others, contract=contract), contract=contract
        )
    stack_frames, stack_audit = [], []
    columns = [f"{name}__expected_excess_return" for name in components]
    aligned = joint.merge(
        labels[["sample_id", "future_excess_return", "outcome_date"]],
        on="sample_id",
        validate="one_to_one",
    )
    for horizon in (1, 3, 5):
        frame = aligned[aligned.horizon_sessions == horizon].copy()
        ds = frame.eligible_entry_date.astype(str)
        for start, end in folds:
            prior = frame[(ds < start) & (frame.outcome_date < start)]
            valid = frame[ds.between(start, end)]
            if len(prior) < 20 or valid.empty:
                continue
            weights, _ = nnls(prior[columns].to_numpy(), prior.future_excess_return.to_numpy())
            if not np.isfinite(weights).all():
                raise ValueError("Nonfinite stacking weights")
            pred = (
                predictions["equal_weight"]
                .set_index("sample_id")
                .loc[valid.sample_id]
                .reset_index()
            )
            pred["expected_excess_return"] = valid[columns].to_numpy() @ weights
            pred["fusion_version"] = "wiring-only-prior-oof-nnls-v1"
            stack_frames.append(
                rank_company_day_predictions(
                    pred,
                    risk_aversion=contract.risk_aversion,
                    round_trip_cost_bps=contract.round_trip_cost_bps,
                )
            )
            stack_audit.append(
                dict(
                    horizon=horizon,
                    start=start,
                    prior_oof_rows=len(prior),
                    latest_training_outcome=prior.outcome_date.max(),
                    weights=weights.tolist(),
                )
            )
    if stack_frames:
        predictions["static_stack"] = pd.concat(stack_frames, ignore_index=True)
    # Persist predictions before their evaluation, and bind output hashes.
    hashes = {}
    for name, pred in predictions.items():
        path = output / f"{name}_rankings.parquet"
        pred.to_parquet(path, index=False)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    common = set.intersection(*(set(p.sample_id) for p in predictions.values()))
    metrics = {}
    for name, pred in predictions.items():
        evaluated = pred[pred.sample_id.isin(common)].merge(
            labels[["sample_id", "future_excess_return"]], on="sample_id", validate="one_to_one"
        )
        # Small cross sections cannot support Top-20 or a remainder comparison.
        sizes = evaluated.groupby(["eligible_entry_date", "horizon_sessions"]).sample_id.transform(
            "size"
        )
        evaluated = evaluated[sizes > 5]
        metrics[name] = evaluate_daily_ranking(evaluated, top_k=(5,), round_trip_cost_bps=20)
    report = dict(
        status="ENGINEERING_E2E_PASS",
        production_qualification="NOT_EVALUATED",
        source="real_cached_training_data",
        created_at=datetime.now(UTC).isoformat(),
        planned_announcements_resolved=planned_resolved,
        feature_rows=len(features),
        companies=int(features.stock_code.nunique()),
        company_days=int(features.company_day_id.nunique()),
        oof_rows=len(joint),
        matched_comparison_rows=len(common),
        folds=fold_audit,
        stack_folds=stack_audit,
        prediction_sha256=hashes,
        ranking_metrics=metrics,
        strict_oos_2026_read=False,
        official_test_read=False,
        provenance={
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "features_sha256": hashlib.sha256(
                (output / "features.parquet").read_bytes()
            ).hexdigest(),
            "labels_sha256": hashlib.sha256((output / "labels.parquet").read_bytes()).hexdigest(),
            "fusion_contract_sha256": contract.source_sha256,
        },
        limitations=[
            "Text is title TF-IDF, not the approved LLM extractor.",
            "Tabular is metadata and lagged bars; PIT financials are absent.",
            "Time series is daily return Ridge, not the intended sequence expert.",
            "Cached-stock subset is incomplete and not representative.",
            "Corporate actions, tradability and source timing need qualification.",
            "Sparse event-date turnover is not a continuous portfolio backtest.",
        ],
    )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("ranking_metrics", "folds", "stack_folds", "prediction_sha256")
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / "data/ddos.db")
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "data/backfills/event_ranking_pilot_electrical_v1/body_plan.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.database, args.plan, args.output)

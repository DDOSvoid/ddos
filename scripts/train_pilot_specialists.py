"""Frozen CPU-only pilot LightGBM/LSTM research, with explicit text abstention."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_pilot_end_to_end import build_samples
from src.prediction.fusion.contract import load_expert_fusion_contract
from src.prediction.fusion.schema import IDENTITY_COLUMNS, build_joint_expert_frame
from src.prediction.tabular.point_in_time_sources import save_json_atomic

BASE = Path("data/tabular/event_ranking_pilot_electrical_v1")
PREPARED = BASE / "expert_inputs_20260907_01"
CONFIG = Path("config/pilot_expert_training_20260907.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def financial_asof(frame, asof):
    eligible = frame.loc[(frame.available_at_utc < asof) & frame.report_type.astype(str).eq("1")]
    if eligible.empty:
        return None
    return eligible.sort_values(["end_date", "available_at_utc", "update_flag"]).iloc[-1]


def prepare():
    audit_path = BASE / "financial_revision_evidence/audit.report.json"
    audit = json.loads(audit_path.read_text())
    if not audit.get("feature_release_eligible"):
        raise PermissionError("financial revision evidence gate failed")
    # Bind the release decision to the archived evidence and the frozen plan.
    evidence_root = BASE / "financial_revision_evidence"
    if sha(evidence_root / "state.json") != audit["state_sha256"]:
        raise PermissionError("financial evidence state changed")
    manifest = json.loads((PREPARED / "manifest.json").read_text())
    for rel, digest in manifest["source_sha256"].items():
        if sha(BASE / "point_in_time_sources" / rel) != digest:
            raise PermissionError("raw source changed after preparation")
    plan_path = Path("data/backfills/event_ranking_pilot_electrical_v1/body_plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    f, labels, _ = build_samples(Path("data/ddos.db"), plan)
    eligibility = pd.read_parquet(PREPARED / "company_day_eligibility.parquet")
    f = f.merge(
        eligibility[["stock_code", "published_date", "stock_eligible", "industry_available"]],
        on=["stock_code", "published_date"],
        validate="many_to_one",
    )
    f = f[f.stock_eligible].reset_index(drop=True)
    with sqlite3.connect("file:data/ddos.db?mode=ro", uri=True) as con:
        bars = pd.read_sql_query(
            "SELECT instrument_code,trade_date,open_price,close_price,high_price,low_price,"
            "volume,amount "
            "FROM daily_prices WHERE trade_date BETWEEN '2022-09-01' AND '2024-12-31' "
            "ORDER BY instrument_code,trade_date",
            con,
        )
    prices = {code: g.copy() for code, g in bars.groupby("instrument_code")}
    benchmark = prices["000300.SH"].set_index("trade_date").close_price.pct_change()
    sequences = {}
    for code, g in prices.items():
        g = g.set_index("trade_date")
        ret = g.close_price.pct_change()
        bench = benchmark.reindex(g.index)
        values = pd.DataFrame(
            dict(
                stock_return=ret,
                benchmark_return=bench,
                excess_return=ret - bench,
                range=(g.high_price - g.low_price) / g.close_price,
                gap=g.open_price / g.close_price.shift(1) - 1,
                log_volume=np.log1p(g.volume.clip(lower=0).fillna(0)),
                log_amount=np.log1p(g.amount.clip(lower=0).fillna(0)),
                tradable=(g.volume.fillna(0) > 0).astype(float),
                volume_mask=g.volume.notna().astype(float),
                amount_mask=g.amount.notna().astype(float),
            )
        )
        sequences[code] = values
    finances = {}
    fields = {
        "income": ["revenue", "n_income_attr_p", "operate_profit"],
        "balancesheet": ["total_assets", "total_liab", "money_cap"],
        "cashflow": ["n_cashflow_act", "n_cashflow_inv_act"],
    }
    for endpoint in fields:
        value = pd.read_parquet(PREPARED / f"{endpoint}.train_candidate.parquet")
        value.available_at_utc = pd.to_datetime(value.available_at_utc, utc=True)
        finances[endpoint] = {code: g for code, g in value.groupby("ts_code")}
    daily = {}
    daily_fields = [
        "turnover_rate",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "total_mv",
        "circ_mv",
    ]
    ds = json.loads((BASE / "daily_basic/state.json").read_text())
    for item in ds["completed"].values():
        if item["status"] != "downloaded":
            continue
        path = BASE / "daily_basic" / item["path"]
        if sha(path) != item["sha256"]:
            raise PermissionError("daily source changed")
        value = pd.read_parquet(path)
        value.available_at_utc = pd.to_datetime(value.available_at_utc, utc=True)
        daily[str(value.ts_code.iloc[0])] = value
    extras, seq = [], []
    for row in f.itertuples():
        asof = pd.Timestamp(row.prediction_as_of).tz_convert("UTC")
        asof = min(asof, pd.Timestamp(row.data_as_of).tz_convert("UTC"))
        hist = sequences[row.stock_code].loc[lambda x: x.index < row.published_date].tail(20)
        seq.append(hist.to_numpy(dtype=np.float32))
        extra = {}
        for endpoint, names in fields.items():
            frame = finances[endpoint].get(row.stock_code)
            selected = None if frame is None else financial_asof(frame, asof)
            for name in names:
                extra[f"fin_{name}"] = np.nan if selected is None else selected[name]
            extra[f"fin_{endpoint}_available"] = int(selected is not None)
        frame = daily.get(row.stock_code)
        # Conservative exclusion of publication-day bars, even though decision is later.
        past = (
            None
            if frame is None
            else frame.loc[
                (frame.available_at_utc < asof)
                & (frame.trade_date.astype(str) < row.published_date)
            ]
        )
        selected = None if past is None or past.empty else past.sort_values("trade_date").iloc[-1]
        for name in daily_fields:
            extra[f"daily_{name}"] = np.nan if selected is None else selected[name]
        extras.append(extra)
    f = pd.concat([f, pd.DataFrame(extras)], axis=1)
    xseq = np.stack(seq)
    if not np.isfinite(xseq).all() or xseq.shape[1:] != (20, 10):
        raise ValueError("invalid sequence; do not silently forward-fill")
    return (
        f,
        labels[labels.sample_id.isin(f.sample_id)],
        xseq,
        {
            "financial_audit_sha256": sha(audit_path),
            "prepared_manifest_sha256": sha(PREPARED / "manifest.json"),
            "plan_sha256": sha(plan_path),
            "daily_state_sha256": sha(BASE / "daily_basic/state.json"),
        },
    )


def main():
    import lightgbm as lgb
    import torch
    from torch import nn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    save_json_atomic(out / "frozen_config.json", cfg)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    f, labels, sequence, provenance = prepare()
    f.to_parquet(out / "features.parquet", index=False)
    labels.to_parquet(out / "labels.parquet", index=False)
    np.save(out / "sequences.npy", sequence)
    data = f.merge(
        labels[["sample_id", "future_excess_return", "outcome_date"]],
        on="sample_id",
        validate="one_to_one",
    )
    columns = [c for c in f if c.startswith(("tab_", "fin_", "daily_"))]
    parts = {name: [] for name in ("tabular", "timeseries", "text")}
    folds = []
    for horizon in cfg["horizons"]:
        for start, end in cfg["folds"]:
            train = data[
                (data.horizon_sessions == horizon)
                & (data.eligible_entry_date < start)
                & (data.outcome_date < start)
            ]
            valid = data[
                (data.horizon_sessions == horizon) & data.eligible_entry_date.between(start, end)
            ]
            if len(train) < 20 or valid.empty:
                continue
            token = f"h{horizon}_{start}"
            torch.manual_seed(cfg["seed"])
            np.random.seed(cfg["seed"])
            good = train[train.industry_available]
            model = lgb.LGBMRegressor(
                **{k: v for k, v in cfg["tabular"].items() if k != "family"},
                n_jobs=1,
                random_state=cfg["seed"],
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
            )
            model.fit(good[columns], good.future_excess_return)
            tab_pred = model.predict(valid[columns])
            model.booster_.save_model(str(out / f"tabular_{token}.txt"))
            xtrain, xvalid = sequence[train.index], sequence[valid.index]
            mean, std = xtrain.mean(axis=(0, 1)), xtrain.std(axis=(0, 1)).clip(1e-6)
            x = torch.tensor((xtrain - mean) / std)
            y = torch.tensor(train.future_excess_return.to_numpy(np.float32) * 100)
            lstm = nn.LSTM(10, cfg["timeseries"]["hidden_size"], batch_first=True)
            head = nn.Linear(cfg["timeseries"]["hidden_size"], 1)
            optimizer = torch.optim.Adam(
                list(lstm.parameters()) + list(head.parameters()),
                lr=cfg["timeseries"]["learning_rate"],
            )
            for _ in range(cfg["timeseries"]["epochs"]):
                order = torch.randperm(len(x))
                for idx in order.split(cfg["timeseries"]["batch_size"]):
                    optimizer.zero_grad()
                    output, _ = lstm(x[idx])
                    loss = nn.functional.mse_loss(head(output[:, -1]).flatten(), y[idx])
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(lstm.parameters()) + list(head.parameters()), 1.0)
                    optimizer.step()
            with torch.no_grad():
                output, _ = lstm(torch.tensor((xvalid - mean) / std))
                ts_pred = head(output[:, -1]).flatten().numpy() / 100
            torch.save(
                dict(
                    lstm=lstm.state_dict(),
                    head=head.state_dict(),
                    mean=mean.tolist(),
                    std=std.tolist(),
                ),
                out / f"timeseries_{token}.pt",
            )
            for name, pred in (
                ("tabular", tab_pred),
                ("timeseries", ts_pred),
                ("text", np.full(len(valid), np.nan)),
            ):
                frame = valid[IDENTITY_COLUMNS + ["data_as_of"]].copy()
                frame["expected_excess_return"] = pred
                frame["expert_signal"] = pred
                frame["available"] = (
                    valid.industry_available.to_numpy() if name == "tabular" else name != "text"
                )
                frame["abstain_reason"] = np.where(
                    frame.available,
                    "",
                    "missing_LLM_extraction"
                    if name == "text"
                    else "historical_industry_unavailable",
                )
                frame.loc[~frame.available, ["expert_signal", "expected_excess_return"]] = np.nan
                frame["quality_score"] = frame.available.astype(float)
                frame["expert_version"] = cfg["contract"] + ":" + name
                frame["evidence_sha256"] = hashlib.sha256(
                    (sha(out / "features.parquet") + sha(CONFIG) + token + name).encode()
                ).hexdigest()
                frame["fold_start"] = start
                parts[name].append(frame)
            folds.append(
                dict(
                    horizon=horizon,
                    start=start,
                    train_rows=len(train),
                    valid_rows=len(valid),
                    latest_training_outcome=train.outcome_date.max(),
                )
            )
            save_json_atomic(out / "progress.json", dict(status="training", completed_folds=folds))
            print(f"trained {token}: train={len(train)} valid={len(valid)}", flush=True)
    components = {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}
    for name, frame in components.items():
        frame.to_parquet(out / f"{name}_oof.parquet", index=False)
    joint = build_joint_expert_frame(components, contract=load_expert_fusion_contract())
    joint.to_parquet(out / "joint_oof.parquet", index=False)
    save_json_atomic(
        out / "report.json",
        dict(
            status="TWO_EXPERT_CANDIDATES_TRAINED_TEXT_BLOCKED",
            folds=folds,
            rows=len(f),
            oof_rows=len(joint),
            provenance=provenance,
            config_sha256=sha(CONFIG),
            runner_sha256=sha(Path(__file__)),
            artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file()},
            torch_version=torch.__version__,
            lightgbm_version=lgb.__version__,
            full_three_expert_training_complete=False,
            production_release=False,
            strict_oos_2026_read=False,
            official_test_read=False,
            limitations=[
                "Text extraction unavailable",
                "Corporate actions and tradability not qualified",
                "Financial evidence is sampled, not exhaustive",
                "Legacy LSTM contract unchanged; independent pilot candidate",
                "No model selection or profitability claim",
            ],
        ),
    )
    print("two specialist candidates trained; text unavailable; no production release", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

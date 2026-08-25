# Time-series data boundary

Only causal time-series inputs, manifests, labels, out-of-fold predictions, and
reports belong here. Generated artifacts remain local and are ignored by Git.

Feature artifacts and label artifacts must be stored separately. The sealed
official-test, quarantine, and forward-validation roles may not be used for
fitting or model selection.

The daily-v1 train coverage audit is reproducible with:

```powershell
python -m src.prediction.timeseries.coverage `
  --output data/timeseries/reports/train_daily_v1_coverage_20260821.json
```

The audit reads train-role identities and pre-2025 price dates only. It does
not select outcome values or any sealed-role row.

Build physically separated train artifacts and run the diagnostic OOF baseline:

```powershell
python -m src.prediction.timeseries.artifacts
python -m src.prediction.timeseries.train_baseline
```

Both commands are train-only. The baseline result is diagnostic until the
transaction-cost gate and identical-sample comparison with table-model OOF are
implemented and passed.

Build the train-only complete-day aggregates derived from the audited 15-minute
snapshot, then run the matched-sample diagnostic comparison:

```powershell
python -m src.prediction.timeseries.intraday_artifacts
python -m src.prediction.timeseries.train_intraday_baseline
```

This version uses only complete trading days strictly before `published_date`.
Publication-day bars and minute-level decisions remain blocked until authoritative
announcement timestamps are available under a separately versioned timing contract.

## Direct-sequence LSTM v1

The accepted implementation contract and archived plan are:

- `config/timeseries_lstm_daily_v1.yaml`
- `docs/TIMESERIES_LSTM_IMPLEMENTATION_PLAN_20260824.md`

Build and audit the physically separated train-only direct-sequence artifacts:

```powershell
python -m src.prediction.timeseries.sequence_artifacts
python -m src.prediction.timeseries.sequence_labels
python -m src.prediction.timeseries.audit_lstm_artifacts
```

Use `--force` only after an intentional source-data or registered-contract change.
The sequence builder never queries an outcome table. The label builder independently
recomputes train labels from cached daily prices. The audit verifies content hashes,
canonical identities, bar timing, train-only roles, prehistory coverage, and optional
agreement with the tabular train labels. None of these commands may read the official
test, quarantine, or forward-validation partitions.

Run the preregistered constant and pooled direct-sequence Logistic/Ridge OOF
baselines after the artifact audit passes:

```powershell
python -m src.prediction.timeseries.sequence_baselines
```

This reports both 20-session and 60-session candidates without selecting a
lookback on outer OOF. Its output is a research baseline for the later LSTM
increment test, not a model eligible for freezing or official-test access.

Run all preregistered small single-layer LSTM candidates only after rebuilding
and auditing artifacts and regenerating the matching simple baseline:

```powershell
python -m src.prediction.timeseries.train_lstm
```

The command performs inner temporal model-fit/early-stop/calibration partitions
inside every outer fold, saves research checkpoints, and evaluates absolute and
simple-sequence incremental gates. Table OOF is skipped unless a candidate first
passes those gates, and the official test is never read by this command.

The immutable v1 OOF is diagnosed read-only with:

```powershell
python -m src.prediction.timeseries.diagnose_lstm_v1
```

After v1 failed, v2 was registered as an independent train-only research
contract. Validate its boundaries with:

```powershell
python -c "from src.prediction.timeseries.lstm_research_contract import load_lstm_v2_research_contract; print(load_lstm_v2_research_contract().candidate_names())"
```

The v2 contract does not authorize official-test, quarantine, forward-validation,
or table-OOF reads. Its hypotheses and execution order are archived in
`docs/TIMESERIES_LSTM_V2_RESEARCH_PLAN_20260825.md`.

The v2 objective is range-based: return bounds are fit from each outer-fold
training segment (`q10`/`q90` by default), while coverage and error checks use
inclusive acceptable ranges. No project-wide fixed return boundary is used.

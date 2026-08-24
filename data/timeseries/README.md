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

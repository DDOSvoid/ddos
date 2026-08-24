# Tabular data boundary

This directory is reserved for generated tabular feature and label artifacts.

- Feature artifacts must contain only point-in-time inputs plus per-feature availability metadata.
- Labels are stored separately and may contain post-announcement return/risk outcomes.
- Only 2023–2024 rows are eligible for fitting and expanding-window model selection.
- Official-test labels (2025-01-01 through 2025-08-19) remain sealed until a frozen candidate is evaluated once.
- 2025-08-20 through 2025-12-31 is quarantined; 2026 onward is forward-validation only.

Generated files are intentionally ignored by Git. Their manifests must record the split, development, and tabular contract hashes.

`intraday_15m/` contains train-period raw bars and resumable source-specific state files. AmazingData labels each 15-minute bar by its start, so `available_at_utc` is the label plus 15 minutes. Raw minute prices are unadjusted and may only be converted into within-day, lagged daily aggregates. They must not replace adjusted daily prices for cross-day returns.

The completed 2023–2024 AmazingData snapshot contains 2,307,216 raw bars and 144,201 complete stock-day aggregates for 343 stocks. `intraday_15m/audit.report.json` is the machine-readable recomputation audit; it records zero errors and proves that the official-test partition was not queried.

`snapshots/ddos.train.sqlite` is a physically train-only database with no sealed tables. `train_v1/features.parquet` and `train_v1/labels.parquet` contain 50,802 aligned rows but remain physically separate; `train_v1/manifest.json` binds both files, their schemas, and all source/contract hashes.

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

The current `train_v1` model schema has 63 inputs: one horizon field, 16 title/category aggregate fields, 31 strictly lagged daily-market fields, and 15 complete-day intraday aggregate fields. It does not yet contain structured announcement facts, point-in-time financial statements, historical company/industry attributes, or any `daily_basic` field.

`daily_basic/` contains a completed train-only Tushare backfill request for 343 selected stock partitions over 2023–2024: 325 partitions contain data, 18 are empty, 151,674 rows were stored, and no request remains failed. `daily_basic/audit.report.json` independently rechecks every partition and hash, reports zero close mismatches against the train-only snapshot, explains that all 18 empty codes also have no eligible train-period daily-price rows, and records `official_test_queried=false`. It covers 97.74% of eligible snapshot stock-days. The legacy artifact split hash differs from the current split-file hash; the audit records a semantic requalification based on the unchanged train role/range rather than claiming exact hash equality.

`train_v2/` is an immutable extension rather than an in-place rewrite of `train_v1`. It contains the same 50,802 aligned, physically separated feature/label rows and 78 model inputs: the 63 v1 inputs plus 15 strictly lagged `daily_basic` valuation/liquidity inputs. The sample-level `daily_basic` availability is 99.21% (50,403 available, 399 explicitly missing). Its manifest records feature groups, availability, units, transformations, missingness, provenance, source hashes, current contract hashes, and the legacy v1 hashes used to establish lineage. Structured announcement facts, point-in-time fundamentals, and historical company/industry context remain explicitly blocked until their own time-evidence audits pass.

`point_in_time_sources/` is the resumable, train-only raw archive for the next two table feature groups. It contains 343 `stock_basic` rows, 470 Shenwan membership rows, and 26,847 financial-statement rows across `income`, `balancesheet`, and `cashflow`. All 1,373 requested partitions completed with zero failures, all stored hashes and version identities passed the independent raw-integrity audit, and `official_test_queried=false`. The initial snapshot covered 16,117 of 17,017 training company-days for industry and 16,953 for all three financial statements; those are raw-archive figures, not feature-release conclusions. The current supplement and revision-evidence results are recorded below and in `docs/TABULAR_POINT_IN_TIME_SOURCE_AUDIT_20260824.md`.

`industry_supplement_v1/` adds historical `SW2014/SW2021` classifications and dated `index_member` intervals. Its raw audit passes with 349/349 partitions, and the combined point-in-time industry coverage is 16,204/17,017 company-days (95.22%); 813 company-days across 58 codes remain explicitly missing and are not backfilled from current attributes. `financial_revision_evidence_v1/` archives 17 train-period announcement PDFs and a deterministic revision plan: 9/12 revision groups pass direct-field cross-checks (3 per financial endpoint), while one all-scanned PDF is retained as CDP metadata only. Both groups remain blocked from `train_v2` until their release blockers are cleared.
`train_v3/` is the train-only candidate table built after both source audits passed. It has the same 50,802 rows as `train_v2`, 88 catalog features, and never queries the official test partition. Industry context uses dated membership intervals and records 813 pre-effective company-days as the explicit `industry_taxonomy_not_yet_effective` state; it never fills them from a current snapshot. PIT financial features select the latest row with `actual_disclosure_date < published_date` and retain endpoint-specific availability timestamps. This table is not a frozen model input until same-sample ablation and OOF gates pass.

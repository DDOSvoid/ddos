# Time-series 15-minute complete-day aggregate baseline — 2026-08-21

## Scope and isolation

- Contract: `causal-timeseries-intraday-aggregates-v1`.
- Read-only source: the audited 2023–2024 15-minute train snapshot.
- Source audit: 608 aggregate files, 144,201 complete stock-day rows, 308 stocks
  with usable aggregates; the 343-stock requested universe includes symbols with no
  usable train-period bars.
- Each feature row uses only complete trading days with
  `observation_date < published_date` and availability no later than the decision
  timestamp.
- Publication-day bars, official test, quarantine, and forward-validation data were
  not read.
- Labels remain physically separate and reuse the daily-v1 sample identity.
- Median imputation and standardization are fitted inside each expanding fold only.

Of 68,023 eligible daily-v1 train samples, 64,099 have at least 20 complete intraday
history days. The explicit exclusions are 1,659 samples with insufficient complete
intraday history and 2,265 samples whose stock has no usable intraday aggregate.

## Preregistered comparison

The same samples and folds were evaluated with fixed unweighted Logistic Regression
`C=0.1`:

1. matched daily-only features;
2. 15-minute complete-day aggregate features only;
3. daily plus 15-minute aggregate features.

The incremental gate requires the combined model to lower pooled Brier and to lower
Brier in at least two of three folds relative to matched daily-only.

## Train-period OOF result

| Horizon | OOF samples | Daily Brier | Intraday Brier | Combined Brier | Intraday AUC | Combined AUC | Combined gain vs daily |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 session | 16,547 | 0.268843 | 0.254139 | 0.273350 | 0.514966 | 0.499731 | -0.004507 |
| 3 sessions | 16,247 | 0.277164 | 0.257641 | 0.282185 | 0.513119 | 0.499367 | -0.005020 |
| 5 sessions | 16,023 | 0.299674 | 0.256746 | 0.300452 | 0.519618 | 0.509840 | -0.000778 |

The intraday-only model is the lowest-Brier feature set at all three horizons, but
its Brier scores remain worse than their fold-train constant baselines
(0.248689/0.247996/0.247649). Its bullish hit rates are
47.35%/46.16%/46.60%, with Wilson 95% lower bounds
46.22%/44.99%/45.38%.

The combined model is worse than matched daily-only in pooled Brier and in every
fold at every horizon. It also fails the general Wilson, fold hit-rate, worst-fold,
and constant-Brier gates.

## Decision

**FAIL; TCN remains blocked.** Finer 15-minute-derived data contain a small amount of
standalone ranking information, but do not provide stable incremental OOF value when
combined with the daily state. Increasing model complexity is not justified by this
evidence.

Do not freeze a model and do not read the official test. Direct intraday sequences
remain out of scope until authoritative announcement timestamps support a separately
versioned minute-decision contract. Machine-readable evidence is stored locally at
`data/timeseries/intraday_aggregates_v1/reports/train_intraday_aggregates_v1_matched_oof.json`.

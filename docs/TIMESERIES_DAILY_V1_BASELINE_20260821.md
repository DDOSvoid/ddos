# Time-series daily-v1 train-only baseline — 2026-08-21

## Scope and isolation

- Contract: `causal-timeseries-daily-v1`.
- Train only: 2023-01-01 through 2024-12-31.
- Three chronological expanding-window folds from the causal development contract.
- Official test, quarantine, and forward-validation roles were not read.
- Features and labels were materialized as separate files and joined only after role,
  timestamp, sample-identity, and SHA-256 validation.
- Every input bar satisfies `trade_date < published_date`; fold preprocessing consists
  of a median imputer and standard scaler fitted on fold-train only.

The source audit found 69,904 accepted company-publication-day-horizon train samples.
68,023 passed the minimum requirement of 20 market sessions and 15 valid stock
sessions; 1,881 were explicitly excluded for insufficient history. No future filling
or full-sample normalization was used.

## Preregistered baseline

The baseline used 5/20/60-session stock, CSI300, and excess momentum; historical
volatility; range and gap; log volume and amount; and tradability fractions. For each
horizon it compared Logistic Regression `C` values 0.1/1.0/10.0 with unweighted and
balanced-class variants. Selection used the minimum pooled train-period OOF Brier
score. All three horizons selected the unweighted `C=0.1` candidate.

## Train-period OOF result

| Horizon | OOF samples | AUC | Brier | Fold-train constant Brier | Bullish signals | Bullish hit rate | Wilson 95% lower |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 session | 17,142 | 0.4995 | 0.2647 | 0.2489 | 8,429 | 46.09% | 45.03% |
| 3 sessions | 16,833 | 0.4966 | 0.2698 | 0.2485 | 8,117 | 44.83% | 43.75% |
| 5 sessions | 16,603 | 0.5039 | 0.2879 | 0.2483 | 8,505 | 44.69% | 43.64% |

## Decision

**FAIL.** Every horizon fails both the Wilson-lower-bound requirement and the Brier
requirement. The cost-adjusted return and table-model incremental gates do not need
to be consumed to reject this candidate because earlier mandatory gates already
fail.

Do not freeze this model, do not read the official test, and do not start a TCN from
the same daily inputs. The next permitted diagnostic is the already archived
train-only 15-minute snapshot, used only as complete pre-publication-day aggregates.
Raw publication-day intraday bars and intraday decision timing remain blocked until
an authoritative announcement timestamp and a versioned timing contract exist.

Machine-readable evidence is stored locally at
`data/timeseries/daily_v1/reports/train_daily_v1_logistic_oof.json`; its manifest and
OOF file carry source and artifact hashes.

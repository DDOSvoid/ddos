# Strict temporal split checkpoint — 2026-08-21

## Enforced partitions

- Train: 2023-01-01 through 2024-12-31.
- One-time official test: 2025-01-01 through 2025-08-19.
- Quarantine: 2025-08-20 through 2025-12-31 because the rejected v1 experiment used part of this period.
- Forward validation: 2026-01-01 onward. It cannot be used for fitting or tuning.
- A train or test target that matures after its partition cutoff is excluded from that eligible role.

The source contract is `config/prediction_splits.yaml`. Every generated target stores the contract name and SHA-256 of that file.

## Completed data work

- Added 118,989 announcement records for 2023-01-01 through 2025-08-19.
- Rule-classified all 118,989 records: 94,733 accepted and 24,256 abstained (79.61% coverage).
- Cached 205,536 stock daily bars and 649 CSI300 bars for 2022-12-15 through 2025-08-19.
- Generated 496,397 real, non-random excess-return targets across 1/3/5 sessions.
- The sealed target report exposes partition counts but no direction distribution for test, quarantine, or forward validation.

The 12 securities without bars were all listed after the 2025-08-19 price cutoff. Their absence is expected; no synthetic prices or outcomes were created.

## Train-only model result

The locked TF-IDF logistic candidate used only eligible 2023–2024 samples and three expanding-window folds. Its bullish results were:

| Horizon | Train samples | CV bullish hit rate | Wilson 95% lower bound |
|---|---:|---:|---:|
| 1 session | 23,409 | 49.94% | 46.66% |
| 3 sessions | 23,288 | 50.29% | 47.47% |
| 5 sessions | 23,207 | 51.79% | 48.51% |

All horizons failed the pre-registered requirement that the historical bullish hit-rate lower bound exceed 50%. A train-only causal market-feature diagnostic also failed. Consequently, the 2025 official test remains unread and the model is not production eligible.

## Current decision

Do not output tradable bullish rankings from either candidate. Preserve the one-time official test for a later candidate that first demonstrates robust training-period walk-forward accuracy.

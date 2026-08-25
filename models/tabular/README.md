# Tabular model boundary

This directory is reserved for generated LightGBM and CatBoost artifacts.

Before an official-test read, a candidate manifest must freeze the feature schema, target definitions, preprocessing, hyperparameters, random seeds, training-data hashes, and all three governing contract hashes. Model binaries are generated locally and ignored by Git.

`research_v1/` contains train-only expanding-window OOF evidence. Its selection review is `KEEP_RESEARCH_ONLY_DO_NOT_FREEZE`; no candidate model binary is frozen and the official test remains unread.

The current OOF artifact has 111,588 rows across linear, LightGBM, and CatBoost candidates. It contains raw bullish probabilities and expected-excess-return predictions for 1/3/5 sessions, while realized-volatility risk predictions are populated only for 3/5 sessions. Probabilities are not yet causally calibrated, and the artifact is not the versioned availability/refusal/provenance interface required by the fusion roadmap.

All direction/return candidates fail the mandatory gates. The CatBoost 3/5-session risk improvements remain research evidence only; they do not authorize freezing a partial component or reading the official test.

`research_v2/` contains 148,784 train-only OOF rows generated from the audited 78-feature `train_v2` schema. It adds chronological fold-train Platt calibration, 1/3/5-session absolute-excess-return risk predictions, availability/refusal fields, model/data versions, and evidence hashes. It also contains a same-sample CatBoost ablation without the 15-feature valuation/liquidity group.

All nine full-v2 model/horizon candidates still fail the mandatory combined gates. Calibration improves each model's raw Brier score but does not beat the fold-train constant-probability baseline; return MAE and cost-adjusted-return gates also fail. CatBoost aggregate risk MAE beats the median baseline at all three horizons, but only 1/3 sessions meet the required fold stability. Valuation/liquidity has small positive Brier and return-MAE increments at all horizons, but its risk increment is not consistent. The v2 selection review therefore remains `KEEP_RESEARCH_ONLY_DO_NOT_FREEZE`, and `official_test_action=DO_NOT_READ`.

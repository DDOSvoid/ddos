# Tabular model boundary

This directory is reserved for generated LightGBM and CatBoost artifacts.

Before an official-test read, a candidate manifest must freeze the feature schema, target definitions, preprocessing, hyperparameters, random seeds, training-data hashes, and all three governing contract hashes. Model binaries are generated locally and ignored by Git.

`research_v1/` contains train-only expanding-window OOF evidence. Its selection review is `KEEP_RESEARCH_ONLY_DO_NOT_FREEZE`; no candidate model binary is frozen and the official test remains unread.

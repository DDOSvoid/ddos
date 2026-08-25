# Time-series model boundary

Frozen causal time-series models and their manifests belong here. Generated
weights are ignored by Git. Every candidate must record the configuration,
data, feature, preprocessing, threshold, code, and model hashes before the
one-time official test can be read.

The registered direct-sequence branch is `lstm_daily_v1`. Its code skeleton and
train artifacts are research-ready, but no candidate is frozen yet. Model weights
must remain under `models/timeseries/lstm_daily_v1/research/` until the constant,
simple-sequence, LSTM, calibration, cost, risk, and tabular-increment OOF gates all
pass. The current decision is `KEEP_RESEARCH_ONLY_DO_NOT_FREEZE`.

The preregistered pooled-sequence Logistic/Ridge baselines have been run and all
six lookback/horizon combinations failed the required training-period gates. They
are evidence and comparison baselines only; they are not frozen model candidates.

LSTM v1 research produced 36 fold checkpoints for 12 preregistered candidates.
All candidates failed the complete train-period gate, so every checkpoint remains
research-only under `lstm_daily_v1/research/`; `lstm_daily_v1/frozen/` must remain
empty and the official test must remain sealed.

The v1 failure diagnosis is stored at
`data/timeseries/lstm_daily_v1/reports/v1_diagnostic_report.json`. The independent
v2 registration is `config/timeseries_lstm_daily_v2_research.yaml`; it remains
hypothesis-only and must not write to the v1 research or frozen directories.
Its objective uses fold-local return ranges and inclusive metric ranges rather
than a single project-wide return boundary.

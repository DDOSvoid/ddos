"""Causal tabular return/risk modelling with strict temporal isolation."""

from src.prediction.tabular.aggregates import (
    INTRADAY_FEATURE_COLUMNS,
    aggregate_intraday_days,
    intraday_feature_specs,
)
from src.prediction.tabular.contract import (
    TabularModelContract,
    load_tabular_model_contract,
)
from src.prediction.tabular.features import (
    FeatureSpec,
    validate_and_select_features,
)
from src.prediction.tabular.intraday import (
    IntradayChunk,
    normalize_intraday_frame,
    yearly_chunks,
)
from src.prediction.tabular.walk_forward import (
    WalkForwardFrames,
    iter_expanding_window_frames,
)

__all__ = [
    "FeatureSpec",
    "INTRADAY_FEATURE_COLUMNS",
    "IntradayChunk",
    "TabularModelContract",
    "WalkForwardFrames",
    "aggregate_intraday_days",
    "intraday_feature_specs",
    "iter_expanding_window_frames",
    "load_tabular_model_contract",
    "normalize_intraday_frame",
    "validate_and_select_features",
    "yearly_chunks",
]

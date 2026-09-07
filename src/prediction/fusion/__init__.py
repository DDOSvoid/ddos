"""Company-day expert fusion and cross-sectional ranking."""

from src.prediction.fusion.adapters import (
    adapt_tabular_v2_oof,
    adapt_timeseries_lstm_oof,
)
from src.prediction.fusion.contract import (
    ExpertFusionContract,
    load_expert_fusion_contract,
)
from src.prediction.fusion.ranking import (
    compute_rank_score,
    evaluate_daily_ranking,
    rank_company_day_predictions,
)
from src.prediction.fusion.schema import (
    build_joint_expert_frame,
    validate_expert_signal_frame,
)

__all__ = [
    "ExpertFusionContract",
    "adapt_tabular_v2_oof",
    "adapt_timeseries_lstm_oof",
    "build_joint_expert_frame",
    "compute_rank_score",
    "evaluate_daily_ranking",
    "load_expert_fusion_contract",
    "rank_company_day_predictions",
    "validate_expert_signal_frame",
]

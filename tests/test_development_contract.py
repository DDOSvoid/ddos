"""Causal multimodel development-contract tests."""

from datetime import UTC, date, datetime

import pytest

from src.prediction.development_contract import load_causal_development_contract


def test_contract_uses_expanding_folds_inside_train_only():
    contract = load_causal_development_contract()
    assert contract.split_contract == "causal-impact-split-v2"
    assert [fold.name for fold in contract.folds] == ["fold_1", "fold_2", "fold_3"]
    assert contract.folds[0].train_start == date(2023, 1, 1)
    assert contract.folds[-1].validation_end == date(2024, 12, 31)
    assert contract.official_test_read_limit == 1


def test_date_only_history_rejects_same_day_and_future_market_bars():
    contract = load_causal_development_contract()
    announcement_day = date(2024, 4, 25)
    assert contract.historical_daily_bar_allowed(
        trade_date=date(2024, 4, 24),
        announcement_published_date=announcement_day,
    )
    assert not contract.historical_daily_bar_allowed(
        trade_date=announcement_day,
        announcement_published_date=announcement_day,
    )
    assert not contract.historical_daily_bar_allowed(
        trade_date=date(2024, 4, 26),
        announcement_published_date=announcement_day,
    )


def test_future_feature_is_rejected():
    contract = load_causal_development_contract()
    cutoff = datetime(2024, 4, 25, 0, tzinfo=UTC)
    contract.require_available(
        feature_name="prior_close",
        available_at=datetime(2024, 4, 24, 7, tzinfo=UTC),
        prediction_as_of=cutoff,
    )
    with pytest.raises(ValueError, match="future feature rejected"):
        contract.require_available(
            feature_name="future_close",
            available_at=datetime(2024, 4, 25, 7, tzinfo=UTC),
            prediction_as_of=cutoff,
        )


def test_fold_training_requires_outcome_to_mature_by_train_cutoff():
    contract = load_causal_development_contract()
    assert contract.fold_role(
        fold_name="fold_2",
        published_date=date(2023, 12, 29),
        outcome_available_at=datetime(2023, 12, 31, 10, tzinfo=UTC),
    ) == "fold_train"
    assert contract.fold_role(
        fold_name="fold_2",
        published_date=date(2023, 12, 29),
        outcome_available_at=datetime(2024, 1, 2, 7, tzinfo=UTC),
    ) == "fold_train_outcome_after_cutoff"


def test_module_gates_keep_memory_ranking_statistical_only():
    contract = load_causal_development_contract()
    memory = contract.module_gates["memory_ranking"]
    assert memory["minimum_matured_predictions_per_key"] == 50
    assert memory["rank_score"] == "historical_hit_rate_wilson_lower"
    assert memory["subjective_model_confidence_in_rank_forbidden"] is True

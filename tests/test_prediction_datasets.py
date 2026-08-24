"""Tests for role-gated prediction datasets."""

import pytest

from src.prediction.datasets import load_company_day_samples
from src.prediction.splits import load_prediction_split_contract


def test_sealed_roles_require_explicit_authorization(db_session):
    split = load_prediction_split_contract()
    with pytest.raises(PermissionError, match="test labels are sealed"):
        load_company_day_samples(db_session, role="test", split=split)


def test_cutoff_roles_cannot_be_loaded_as_eligible_data(db_session):
    split = load_prediction_split_contract()
    with pytest.raises(ValueError, match="unsupported eligible dataset role"):
        load_company_day_samples(
            db_session,
            role="train_outcome_after_cutoff",
            split=split,
        )

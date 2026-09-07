import sqlite3

from src.prediction.strict_oos import (
    audit_current_database,
    load_strict_oos_contract,
)


def test_strict_oos_2026_contract_is_sealed_before_prediction_lock():
    contract = load_strict_oos_contract()
    assert contract.feature_rows_allowed_pre_freeze is False
    assert contract.outcomes_allowed_pre_freeze is False
    assert contract.outcomes_allowed_before_prediction_lock is False
    assert contract.evaluation_count == 1
    assert contract.tuning_from_results_allowed is False
    assert contract.repository_wide_pristine_claim_allowed is False


def test_strict_oos_database_audit_reads_counts_only(tmp_path):
    database = tmp_path / "test.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE daily_prices (trade_date DATE);"
            "CREATE TABLE announcements (id INTEGER, published_date DATE);"
            "CREATE TABLE announcement_market_targets (announcement_id INTEGER);"
            "CREATE TABLE fused_stock_predictions (id INTEGER, eligible_entry_date DATE);"
            "CREATE TABLE fused_prediction_outcomes (prediction_id INTEGER);"
        )
    report = audit_current_database(database)
    assert report["current_architecture_store_clean"] is True
    assert report["daily_prices_2026"] == 0
    assert report["market_targets_2026"] == 0
    assert report["fused_outcomes_2026"] == 0

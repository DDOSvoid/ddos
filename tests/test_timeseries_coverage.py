"""Train-only daily sequence coverage audit tests."""

from src.prediction.timeseries.contract import load_timeseries_model_contract
from src.prediction.timeseries.coverage import summarize_train_coverage


def test_coverage_uses_only_prepublication_calendar_sessions():
    contract = load_timeseries_model_contract()
    benchmark = [
        ("000300.SH", f"2023-01-{day:02d}") for day in range(1, 32)
    ]
    stock = [("000001.SZ", raw_day) for _, raw_day in benchmark]
    # The publication-day price must not help a short-history sample reach 20 bars.
    samples = [("000001.SZ", "2023-01-20", 1)]
    report = summarize_train_coverage(
        samples,
        [*benchmark, *stock],
        contract=contract,
    )
    assert report["totals"]["samples"] == 1
    assert report["totals"]["eligible_minimum"] == 0


def test_coverage_reports_full_sequence_without_reading_labels():
    contract = load_timeseries_model_contract()
    benchmark = [
        ("000300.SH", f"2023-{month:02d}-{day:02d}")
        for month in (1, 2, 3)
        for day in range(1, 29)
    ]
    stock = [("000001.SZ", raw_day) for _, raw_day in benchmark]
    report = summarize_train_coverage(
        [("000001.SZ", "2023-04-01", 5)],
        [*benchmark, *stock],
        contract=contract,
    )
    assert report["totals"]["eligible_minimum"] == 1
    assert report["totals"]["full_sequence"] == 1

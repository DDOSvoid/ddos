"""Train-only, resumable intraday data tests."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from src.prediction.splits import load_prediction_split_contract
from src.prediction.tabular.contract import load_tabular_model_contract
from src.prediction.tabular.intraday import (
    IntradayChunk,
    download_intraday_train,
    normalize_intraday_frame,
    require_train_only_range,
    yearly_chunks,
)


class FakeIntradayClient:
    source = "fake_intraday"
    bar_timestamp_semantics = "bar_start"

    def __init__(self) -> None:
        self.calls = []

    def fetch(self, chunk: IntradayChunk, *, frequency: str) -> pd.DataFrame:
        self.calls.append((chunk, frequency))
        return pd.DataFrame(
            {
                "ts_code": [chunk.stock_code, chunk.stock_code],
                "trade_time": [
                    f"{chunk.start.isoformat()} 09:45:00",
                    f"{chunk.start.isoformat()} 09:30:00",
                ],
                "open": [10.1, 10.0],
                "high": [10.3, 10.2],
                "low": [10.0, 9.9],
                "close": [10.2, 10.1],
                "vol": [200.0, 100.0],
                "amount": [2040.0, 1010.0],
            }
        )


def test_yearly_chunks_never_cross_year_boundary():
    chunks = yearly_chunks(
        "300750.SZ", start=date(2023, 6, 1), end=date(2024, 5, 1)
    )
    assert [(item.start, item.end) for item in chunks] == [
        (date(2023, 6, 1), date(2023, 12, 31)),
        (date(2024, 1, 1), date(2024, 5, 1)),
    ]


def test_intraday_download_range_cannot_leave_training_partition():
    split = load_prediction_split_contract()
    require_train_only_range(
        start=date(2023, 1, 1), end=date(2024, 12, 31), split=split
    )
    with pytest.raises(PermissionError, match="2023-2024 train partition"):
        require_train_only_range(
            start=date(2023, 1, 1), end=date(2025, 1, 2), split=split
        )


def test_normalization_sorts_and_stamps_point_in_time_metadata():
    chunk = IntradayChunk("300750.SZ", date(2024, 1, 2), date(2024, 1, 2))
    raw = FakeIntradayClient().fetch(chunk, frequency="15min")
    normalized = normalize_intraday_frame(
        raw,
        chunk=chunk,
        frequency="15min",
        fetched_at=datetime(2026, 8, 21, tzinfo=UTC),
        source="amazingdata_query_kline",
        bar_timestamp_semantics="bar_start",
    )
    assert normalized["trade_time"].is_monotonic_increasing
    assert str(normalized["available_at_utc"].dt.tz) == "UTC"
    assert normalized["frequency"].unique().tolist() == ["15min"]
    assert normalized["price_adjustment"].unique().tolist() == ["raw"]
    assert normalized["bar_timestamp_semantics"].unique().tolist() == ["bar_start"]
    assert str(normalized.iloc[0]["available_at_utc"]) == "2024-01-02 01:45:00+00:00"


def test_download_is_resumable_and_writes_hashed_parquet(tmp_path):
    client = FakeIntradayClient()
    state_path = tmp_path / "state.json"
    kwargs = {
        "codes": ["300750.SZ"],
        "start": date(2023, 1, 1),
        "end": date(2024, 12, 31),
        "output_root": tmp_path / "intraday_15m",
        "state_path": state_path,
        "client": client,
        "split": load_prediction_split_contract(),
        "tabular": load_tabular_model_contract(),
    }
    first = download_intraday_train(**kwargs)
    assert len(first["completed"]) == 2
    assert not first["failures"]
    assert len(client.calls) == 2
    for item in first["completed"].values():
        assert item["status"] == "downloaded"
        assert len(item["sha256"]) == 64
        assert (kwargs["output_root"] / item["path"]).exists()

    second = download_intraday_train(**kwargs)
    assert second["completed"] == first["completed"]
    assert len(client.calls) == 2

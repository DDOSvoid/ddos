import pandas as pd

from scripts.seed_database import _download_universe, _parse_date


class _FakePro:
    def __init__(self):
        self.statuses = []

    def stock_basic(self, *, list_status, **kwargs):
        self.statuses.append(list_status)
        if list_status == "L":
            return pd.DataFrame([{"ts_code": "000001.SZ", "name": "A"}])
        if list_status == "D":
            return pd.DataFrame([{"ts_code": "000002.SZ", "name": "B"}])
        return pd.DataFrame([{"ts_code": "000001.SZ", "name": "duplicate"}])


def test_universe_download_includes_ldp_and_deduplicates_codes():
    client = _FakePro()
    frame = _download_universe(client, "")
    assert client.statuses == ["L", "D", "P"]
    assert set(frame["ts_code"]) == {"000001.SZ", "000002.SZ"}
    assert frame.loc[frame["ts_code"] == "000001.SZ", "list_status"].iloc[0] == "L"


def test_tushare_dates_are_parsed_without_filling_missing_values():
    assert str(_parse_date("20230102")) == "2023-01-02"
    assert _parse_date("") is None
    assert _parse_date(float("nan")) is None

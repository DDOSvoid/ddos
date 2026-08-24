"""Historical announcement backfill helpers."""

from datetime import date
from types import SimpleNamespace

from scripts.backfill_historical_announcements import (
    _records_from_items,
    month_ranges,
)


def test_month_ranges_keep_exact_boundaries():
    assert month_ranges(date(2025, 8, 20), date(2025, 10, 3)) == [
        (date(2025, 8, 20), date(2025, 8, 31)),
        (date(2025, 9, 1), date(2025, 9, 30)),
        (date(2025, 10, 1), date(2025, 10, 3)),
    ]


def test_records_use_item_stock_code_and_real_notice_date():
    company = SimpleNamespace(id=7, stock_code="300750.SZ")
    records, unmatched = _records_from_items(
        [
            {
                "art_code": "AN1",
                "title": "真实公告",
                "notice_date": "2025-09-08 00:00:00",
                "codes": [{"stock_code": "300750"}],
            },
            {
                "art_code": "AN2",
                "title": "范围外公司",
                "notice_date": "2025-09-08",
                "codes": [{"stock_code": "000001"}],
            },
        ],
        {"300750": company},
    )
    assert unmatched == 1
    assert len(records) == 1
    assert records[0]["company_id"] == 7
    assert records[0]["announcement_id"] == "AN1"
    assert records[0]["published_date"] == date(2025, 9, 8)
    assert records[0]["processing_status"] == "fetched"

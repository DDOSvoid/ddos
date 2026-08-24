"""Announcement content-probe selection tests."""

from scripts.probe_announcement_content_api import select_evenly


def test_select_evenly_keeps_boundaries_and_requested_size():
    rows = list(range(100))
    selected = select_evenly(rows, 5)
    assert len(selected) == 5
    assert selected[0] == 0
    assert selected[-1] == 99


def test_select_evenly_handles_empty_and_short_inputs():
    assert select_evenly([], 10) == []
    assert select_evenly([1, 2], 10) == [1, 2]
    assert select_evenly([1, 2], 0) == []

import json
from unittest.mock import MagicMock

from src.pipeline.cdp_fetcher import CdpEastmoneyClient
from src.pipeline.content_page_cache import ContentPageCache


def page(index, version="v1"):
    return dict(art_code="A1", page_size=3, notice_content=f"{version}-page-{index}")


def test_resume_after_interrupted_third_page_refreshes_first_and_reuses_second(tmp_path):
    client = CdpEastmoneyClient(page_cache_root=tmp_path, content_max_retries=1)
    client._fetch_json = MagicMock(
        side_effect=[{"data": page(1)}, {"data": page(2)}, RuntimeError("network")]
    )
    assert client.fetch_announcement_content("A1")["_content_complete"] is False
    resumed = CdpEastmoneyClient(page_cache_root=tmp_path, content_max_retries=1)
    resumed._fetch_json = MagicMock(side_effect=[{"data": page(1)}, {"data": page(3)}])
    result = resumed.fetch_announcement_content("A1")
    assert result["_content_complete"] is True
    assert result["notice_content"] == "v1-page-1v1-page-2v1-page-3"
    assert resumed._fetch_json.call_count == 2
    assert "page_index=3" in resumed._fetch_json.call_args_list[1].args[0]


def test_changed_first_page_cannot_reuse_previous_generation(tmp_path):
    old = ContentPageCache(tmp_path, "A1", page(1))
    old.write(2, page(2))
    new = ContentPageCache(tmp_path, "A1", page(1, "v2"))
    assert new.read(2) is None


def test_corrupted_payload_and_expired_checkpoints_are_cache_misses(tmp_path):
    cache = ContentPageCache(tmp_path, "A1", page(1))
    cache.write(2, page(2))
    path = cache.directory / "0002.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["payload"]["notice_content"] = "tampered"
    path.write_text(json.dumps(row), encoding="utf-8")
    assert cache.read(2) is None
    cache.write(2, page(2))
    row = json.loads(path.read_text(encoding="utf-8"))
    row["saved_at"] = 0
    path.write_text(json.dumps(row), encoding="utf-8")
    assert cache.read(2) is None

"""真实分类标注契约测试。"""

import pytest

from src.training.label_contract import validate_label_item, validate_temporal_roles


def _item(**overrides):
    item = {
        "schema_version": "real-label-v1",
        "announcement_id": "ANN-REAL-001",
        "published_at": "2026-01-01T09:00:00+08:00",
        "text": "某公司关于回购股份的公告\n公司拟回购股份。",
        "major_category": "C",
        "sub_category": "buyback",
        "label_source": "human",
        "label_status": "verified",
        "labeled_at": "2026-01-02T10:00:00+08:00",
        "reviewer": "tester",
        "dataset_role": "train",
    }
    item.update(overrides)
    return item


def test_verified_real_label_is_accepted():
    sample = validate_label_item(_item(), production=True)
    assert sample.sub_category == "buyback"
    assert sample.dataset_role == "train"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("label_source", "synthetic"),
        ("label_source", "ai_candidate"),
        ("label_source", "rule_candidate"),
        ("label_status", "candidate"),
    ],
)
def test_non_verified_targets_are_rejected(field, value):
    with pytest.raises(ValueError):
        validate_label_item(_item(**{field: value}), production=True)


def test_label_time_cannot_precede_announcement():
    with pytest.raises(ValueError, match="不能早于"):
        validate_label_item(
            _item(labeled_at="2025-12-31T10:00:00+08:00"),
            production=True,
        )


def test_temporal_roles_must_not_overlap():
    train = validate_label_item(_item(), production=True)
    validation = validate_label_item(
        _item(
            announcement_id="ANN-REAL-002",
            published_at="2026-01-02T09:00:00+08:00",
            labeled_at="2026-01-03T10:00:00+08:00",
            dataset_role="validation",
        ),
        production=True,
    )
    test = validate_label_item(
        _item(
            announcement_id="ANN-REAL-003",
            published_at="2026-01-03T09:00:00+08:00",
            labeled_at="2026-01-04T10:00:00+08:00",
            dataset_role="test",
        ),
        production=True,
    )
    validate_temporal_roles([train, validation, test])

    overlapping_validation = validate_label_item(
        _item(
            announcement_id="ANN-REAL-004",
            published_at="2026-01-01T08:00:00+08:00",
            labeled_at="2026-01-03T10:00:00+08:00",
            dataset_role="validation",
        ),
        production=True,
    )
    with pytest.raises(ValueError, match="train 与 validation"):
        validate_temporal_roles([train, overlapping_validation, test])

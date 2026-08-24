"""轻量数据库迁移回归测试。"""

from sqlalchemy import create_engine, text

from src.database.engine import _run_lightweight_migrations


def test_classification_v2_migration_preserves_legacy_category(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE classifications ("
                "id INTEGER PRIMARY KEY, announcement_id INTEGER, "
                "major_category VARCHAR(2), sub_category VARCHAR(50), "
                "confidence FLOAT, industry VARCHAR(50), industry_group VARCHAR(20))"
            )
        )
        conn.execute(
            text(
                "INSERT INTO classifications "
                "(id, announcement_id, major_category, sub_category, confidence) "
                "VALUES (1, 10, 'A', 'earnings_q1', 0.42)"
            )
        )

    _run_lightweight_migrations(engine)
    _run_lightweight_migrations(engine)  # 幂等

    with engine.connect() as conn:
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(classifications)"))}
        row = conn.execute(
            text(
                "SELECT major_category, sub_category, confidence, "
                "classification_source, review_status, needs_review, taxonomy_version "
                "FROM classifications WHERE id = 1"
            )
        ).one()

    assert {"document_type", "relevance", "model_margin", "reviewed_at"} <= columns
    assert row[:3] == ("A", "earnings_q1", 0.42)
    assert row[3:] == ("legacy", "pending", 1, "v1")


def test_market_target_split_columns_are_added_idempotently(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy-target.db'}")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE announcement_market_targets ("
                "id INTEGER PRIMARY KEY, actual_direction INTEGER)"
            )
        )

    _run_lightweight_migrations(engine)
    _run_lightweight_migrations(engine)

    with engine.connect() as conn:
        columns = {
            row[1]
            for row in conn.execute(
                text("PRAGMA table_info(announcement_market_targets)")
            )
        }

    assert {
        "dataset_role",
        "split_contract",
        "split_source_sha256",
    } <= columns

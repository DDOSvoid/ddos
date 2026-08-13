"""Web 前端测试 — TestClient + StaticPool 内存库 + dependency_overrides。

关键点：TestClient 在独立线程跑同步依赖，普通 `:memory:` 引擎每个线程拿到
独立空库，必须用 StaticPool + check_same_thread=False 共享同一连接，否则
fixture 里 create_all 的种子数据对请求线程不可见。
"""

from datetime import date, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src.config import Config
from src.database.models import (
    Announcement,
    Base,
    Classification,
    Company,
    DailyReport,
    ExtractedField,
    PipelineRun,
    Score,
)
from src.web.app import create_app
from src.web.deps import get_db


def _fake_resolve_path(tmp_path):
    """把 config.resolve_path 指到临时目录（pydantic 实例不可 setattr，需 patch 类方法）。"""
    return lambda self, rel: tmp_path / Path(rel).name


def _make_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine


def _make_client(engine):
    """构建绑定到指定引擎的测试客户端。"""
    app = create_app()

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def _seed(session: Session) -> None:
    """造最小数据集：2 公告（A/F 分类，利好/利空）、字段、日报、管线记录。"""
    comp = Company(
        stock_code="000001.SZ",
        stock_name="平安银行",
        exchange="SZSE",
        industry="银行",
        annual_revenue=15_000_000,
        net_assets=5_000_000,
        is_tracked=True,
    )
    session.add(comp)
    session.flush()

    ann_a = Announcement(
        company_id=comp.id,
        announcement_id="ANN-A",
        title="2024年第一季度报告",
        full_text="公司2024年Q1实现营收100亿元，同比增长15%，净利润20亿元。",
        published_date=date(2024, 4, 25),
        source_url="http://example.com/ann-a",
        processing_status="reported",
    )
    ann_b = Announcement(
        company_id=comp.id,
        announcement_id="ANN-B",
        title="关于收到行政处罚决定书的公告",
        full_text="公司因违规收到监管处罚，罚没款项合计500万元。",
        published_date=date(2024, 4, 26),
        source_url="http://example.com/ann-b",
        processing_status="scored",
    )
    session.add_all([ann_a, ann_b])
    session.flush()

    session.add(
        Classification(
            announcement_id=ann_a.id,
            major_category="A",
            sub_category="earnings_q1",
            confidence=0.92,
            model_version="v1.0",
            industry="银行",
            industry_group="金融",
        )
    )
    session.add(
        Classification(
            announcement_id=ann_b.id,
            major_category="F",
            sub_category="penalty",
            confidence=0.80,
            model_version="v1.0",
            industry="银行",
            industry_group="金融",
        )
    )

    session.add(
        Score(
            announcement_id=ann_a.id,
            direction=0.5,
            magnitude=0.6,
            surprise=0.4,
            credibility=0.85,
            composite_score=0.102,  # 利好
            score_version="v1.0",
        )
    )
    session.add(
        Score(
            announcement_id=ann_b.id,
            direction=-0.6,
            magnitude=0.5,
            surprise=0.3,
            credibility=0.80,
            composite_score=-0.200,  # 利空
            score_version="v1.0",
        )
    )

    session.add(
        ExtractedField(
            announcement_id=ann_a.id,
            field_name="revenue",
            field_value="10000000000",
            field_type="numeric",
            unit="CNY",
            confidence=0.95,
        )
    )

    session.add(
        DailyReport(
            report_date=date(2024, 4, 25),
            industry_filter=None,
            report_title="公告事件驱动分析日报 — 2024-04-25",
            report_content=(
                "# 公告事件驱动分析日报\n\n## 一、执行摘要\n\n"
                "- **公告总数**: 1\n\n## 二、高影响事件\n\n*今日无高影响事件*"
            ),
            summary_text="2024-04-25 共处理 1 条公告，无高影响事件。",
            high_impact_count=0,
            total_announcements=1,
        )
    )

    session.add(
        PipelineRun(
            run_id="run-test-1",
            run_type="full",
            status="completed",
            started_at=datetime(2024, 4, 25, 18, 0),
            completed_at=datetime(2024, 4, 25, 18, 1),
            records_processed=2,
            records_failed=0,
        )
    )

    session.commit()


@pytest.fixture
def client():
    engine = _make_engine()
    with Session(engine) as session:
        _seed(session)
    with _make_client(engine) as client:
        yield client
    engine.dispose()


@pytest.fixture
def empty_client():
    """空库客户端 — 验证仪表盘/列表在无数据时优雅降级。"""
    engine = _make_engine()
    with _make_client(engine) as client:
        yield client
    engine.dispose()


# ── 仪表盘 ────────────────────────────────────────────────────


def test_dashboard_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "数据概览" in html
    assert "公告总数" in html
    assert "高影响事件" in html
    assert "2024-04-25" in html            # 日报日期
    assert "run-test-1" in html            # 管线运行记录


def test_dashboard_empty_db(empty_client):
    resp = empty_client.get("/")
    assert resp.status_code == 200
    assert "暂无管线运行记录" in resp.text
    assert "暂无日报" in resp.text


# ── 公告列表 ───────────────────────────────────────────────────


def test_announcements_list(client):
    resp = client.get("/announcements")
    assert resp.status_code == 200
    html = resp.text
    assert "2024年第一季度报告" in html
    assert "关于收到行政处罚决定书的公告" in html


def test_announcements_filter_keyword(client):
    resp = client.get("/announcements", params={"keyword": "处罚"})
    assert resp.status_code == 200
    html = resp.text
    assert "关于收到行政处罚决定书的公告" in html
    assert "2024年第一季度报告" not in html


def test_announcements_filter_status(client):
    resp = client.get("/announcements", params={"processing_status": "scored"})
    assert resp.status_code == 200
    assert "关于收到行政处罚决定书的公告" in resp.text
    assert "2024年第一季度报告" not in resp.text


def test_announcements_filter_direction(client):
    resp = client.get("/announcements", params={"direction": "利好"})
    assert resp.status_code == 200
    assert "2024年第一季度报告" in resp.text
    assert "关于收到行政处罚决定书的公告" not in resp.text


def test_announcements_filter_industry_group(client):
    resp = client.get("/announcements", params={"industry_group": "金融"})
    assert resp.status_code == 200
    assert "2024年第一季度报告" in resp.text
    # 存在但未分类到该行业域的公告不应出现（join 筛选）
    assert "关于收到行政处罚决定书的公告" in resp.text  # 两者都映射到金融


def test_announcements_filter_major_category(client):
    resp = client.get("/announcements", params={"major_category": "A"})
    assert resp.status_code == 200
    assert "2024年第一季度报告" in resp.text
    assert "关于收到行政处罚决定书的公告" not in resp.text


def test_announcements_filter_date_range(client):
    resp = client.get(
        "/announcements",
        params={"start_date": "2024-04-25", "end_date": "2024-04-25"},
    )
    assert resp.status_code == 200
    assert "2024年第一季度报告" in resp.text
    assert "关于收到行政处罚决定书的公告" not in resp.text


def test_announcements_pagination(client):
    resp = client.get("/announcements", params={"page_size": 1, "page": 2})
    assert resp.status_code == 200
    html = resp.text
    # 共 2 条，每页 1 条 → 第 2 页应为最早那条（倒序：B 在前，A 在后）
    assert "关于收到行政处罚决定书的公告" not in html
    assert "2024年第一季度报告" in html
    assert "第 2 / 2 页" in html


def test_announcements_empty_result(client):
    resp = client.get("/announcements", params={"keyword": "不存在的关键词"})
    assert resp.status_code == 200
    assert "没有符合筛选条件的公告" in resp.text


# ── 公告详情 ───────────────────────────────────────────────────


def test_announcement_detail(client):
    import re

    # 筛选出有提取字段的季报公告，取它的详情链接
    resp = client.get("/announcements", params={"keyword": "第一季度"})
    assert resp.status_code == 200
    m = re.search(r"/announcements/(\d+)", resp.text)
    assert m, "列表页应包含公告详情链接"
    ann_id = int(m.group(1))

    resp = client.get(f"/announcements/{ann_id}")
    assert resp.status_code == 200
    html = resp.text
    assert "分类" in html
    assert "评分" in html
    assert "营业收入" in html          # 提取字段中文标签（revenue）
    assert "公告原文" in html


def test_announcement_detail_404(client):
    resp = client.get("/announcements/99999")
    assert resp.status_code == 404


# ── 日报 ───────────────────────────────────────────────────────


def test_reports_list(client):
    resp = client.get("/reports")
    assert resp.status_code == 200
    html = resp.text
    assert "2024-04-25" in html
    assert "公告事件驱动分析日报" in html


def test_reports_empty(empty_client):
    resp = empty_client.get("/reports")
    assert resp.status_code == 200
    assert "暂无日报" in resp.text


def test_report_detail_renders_markdown(client):
    resp = client.get("/reports/2024-04-25")
    assert resp.status_code == 200
    html = resp.text
    assert "一、执行摘要" in html         # markdown 渲染后的标题
    assert "今日无高影响事件" in html


def test_report_detail_404(client):
    resp = client.get("/reports/2020-01-01")
    assert resp.status_code == 404


def test_report_file(client, monkeypatch, tmp_path):
    # 路由用 config.resolve_path 定位文件，patch 类方法指到临时目录
    monkeypatch.setattr(Config, "resolve_path", _fake_resolve_path(tmp_path))
    (tmp_path / "report_2024-04-25.md").write_text("# test", encoding="utf-8")

    resp = client.get("/reports/2024-04-25/file")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    assert resp.text == "# test"


def test_report_file_404_when_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "resolve_path", _fake_resolve_path(tmp_path))
    # 数据库有 2024-04-25 记录，但文件不存在 → 404
    resp = client.get("/reports/2024-04-25/file")
    assert resp.status_code == 404

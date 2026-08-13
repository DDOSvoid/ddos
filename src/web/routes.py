"""只读路由 — 全部 GET，读取管线产出数据并渲染模板。"""

import math
from datetime import date, datetime
from typing import Optional
from urllib.parse import urlencode

import markdown as markdown_lib
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from src.config import config, event_registry, industry_registry
from src.database.repository import (
    AnnouncementRepository,
    DailyReportRepository,
    PipelineRunRepository,
    StatsRepository,
)
from src.web.deps import get_db
from src.web.labels import category_label

router = APIRouter()

# 处理状态全集（与 Announcement.processing_status 约束一致）
STATUS_OPTIONS = [
    "fetched",
    "preprocessed",
    "classified",
    "extracted",
    "scored",
    "reported",
    "failed",
]


def _parse_optional_date(value: Optional[str]) -> Optional[date]:
    """宽松解析 YYYY-MM-DD；空串 / 非法返回 None（避免 422）。"""
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _templates(request: Request):
    return request.app.state.templates


# ── 仪表盘 ─────────────────────────────────────────────────────


@router.get("/", name="dashboard")
def dashboard(request: Request, session: Session = Depends(get_db)):
    status_counts = AnnouncementRepository.count_group_by_status(session)
    category_dist = StatsRepository.category_distribution(session)
    industry_dist = StatsRepository.industry_group_distribution(session)
    direction_dist = StatsRepository.direction_distribution(session)

    charts = {
        "category": [
            {"name": category_label(code), "value": count}
            for code, count in category_dist
        ],
        "industry": [{"name": g, "value": c} for g, c in industry_dist],
        "direction": [
            {"name": k, "value": v} for k, v in direction_dist.items()
        ],
    }

    ctx = {
        "active": "dashboard",
        "total_announcements": AnnouncementRepository.count(session),
        "high_impact": StatsRepository.high_impact_count(session),
        "direction_dist": direction_dist,
        "status_counts": status_counts,
        "status_options": STATUS_OPTIONS,
        "charts": charts,
        "recent_runs": PipelineRunRepository.get_recent(session, limit=10),
        "reports": DailyReportRepository.get_all(session, limit=10),
    }
    return _templates(request).TemplateResponse(request, "dashboard.html", ctx)


# ── 公告列表 ───────────────────────────────────────────────────


@router.get("/announcements", name="announcements")
def announcements(
    request: Request,
    session: Session = Depends(get_db),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    industry_group: Optional[str] = None,
    major_category: Optional[str] = None,
    direction: Optional[str] = None,
    processing_status: Optional[str] = None,
    keyword: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    page = max(1, page)
    page_size = min(max(1, page_size), 100)

    filters = dict(
        start_date=_parse_optional_date(start_date),
        end_date=_parse_optional_date(end_date),
        industry_group=industry_group or None,
        major_category=major_category or None,
        direction=direction or None,
        status=processing_status or None,
        keyword=keyword or None,
    )

    items = AnnouncementRepository.search(session, page=page, page_size=page_size, **filters)
    total = AnnouncementRepository.count_search(session, **filters)
    total_pages = max(1, math.ceil(total / page_size))

    def page_url(n: int) -> str:
        params = {"page": n, "page_size": page_size}
        if filters["start_date"]:
            params["start_date"] = filters["start_date"].isoformat()
        if filters["end_date"]:
            params["end_date"] = filters["end_date"].isoformat()
        for key in ("industry_group", "major_category", "direction", "status", "keyword"):
            if filters.get(key):
                params[key] = filters[key]
        return str(request.url_for("announcements")) + "?" + urlencode(params)

    ctx = {
        "active": "announcements",
        "items": items,
        "pager": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
        },
        "page_url": page_url,
        # 筛选表单选项
        "filters": filters,
        "industry_groups": industry_registry.group_names(),
        "category_options": [
            (code, category_label(code)) for code in event_registry.categories
        ],
        "direction_options": ["利好", "利空", "中性"],
        "status_options": STATUS_OPTIONS,
    }
    return _templates(request).TemplateResponse(request, "announcements.html", ctx)


# ── 公告详情 ───────────────────────────────────────────────────


@router.get("/announcements/{announcement_id}", name="announcement_detail")
def announcement_detail(
    request: Request,
    announcement_id: int,
    session: Session = Depends(get_db),
):
    ann = AnnouncementRepository.get_by_id(session, announcement_id)
    if ann is None:
        raise HTTPException(status_code=404, detail="公告不存在")

    ctx = {
        "active": "announcements",
        "ann": ann,
    }
    return _templates(request).TemplateResponse(request, "announcement_detail.html", ctx)


# ── 日报 ───────────────────────────────────────────────────────


@router.get("/reports", name="reports")
def reports(request: Request, session: Session = Depends(get_db)):
    ctx = {
        "active": "reports",
        "reports": DailyReportRepository.get_all(session, limit=50),
    }
    return _templates(request).TemplateResponse(request, "reports.html", ctx)


@router.get("/reports/{report_date}", name="report_detail")
def report_detail(
    request: Request,
    report_date: date,
    session: Session = Depends(get_db),
):
    report = DailyReportRepository.get_primary(session, report_date)
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")

    content = report.report_content or ""
    rendered = markdown_lib.markdown(content, extensions=["tables", "fenced_code"])
    ctx = {
        "active": "reports",
        "report": report,
        "rendered_html": rendered,
    }
    return _templates(request).TemplateResponse(request, "report_detail.html", ctx)


@router.get("/reports/{report_date}/file", name="report_file")
def report_file(
    report_date: date,
    session: Session = Depends(get_db),
):
    report = DailyReportRepository.get_primary(session, report_date)
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")

    path = config.resolve_path(f"data/reports/report_{report_date.isoformat()}.md")
    if not path.exists():
        raise HTTPException(status_code=404, detail="报告文件不存在")
    return FileResponse(path, media_type="text/markdown", filename=path.name)

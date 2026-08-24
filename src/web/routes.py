"""公告展示与本地人工复核路由。"""

import math
from datetime import date, datetime
from typing import Optional
from urllib.parse import parse_qs, urlencode

import markdown as markdown_lib
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.orm import Session

from src.config import config, event_registry, industry_registry
from src.database.repository import (
    AnnouncementRepository,
    ClassificationRepository,
    DailyReportRepository,
    PipelineRunRepository,
    StatsRepository,
)
from src.pipeline.classification_rules import default_relevance
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

REVIEW_STATUS_OPTIONS = ["pending", "auto_accepted", "reviewed"]
CLASSIFICATION_SOURCE_OPTIONS = [
    "legacy", "rule", "rule+model", "model", "abstained", "manual"
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
    sub_category: Optional[str] = None,
    review_status: Optional[str] = None,
    classification_source: Optional[str] = None,
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
        sub_category=sub_category or None,
        review_status=review_status or None,
        classification_source=classification_source or None,
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
        for key in (
            "industry_group", "major_category", "sub_category", "review_status",
            "classification_source", "direction", "keyword",
        ):
            if filters.get(key):
                params[key] = filters[key]
        if filters.get("status"):
            params["processing_status"] = filters["status"]
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
        "subcategory_options": [
            (sub_code, sub_def.label)
            for category in event_registry.categories.values()
            for sub_code, sub_def in category.subcategories.items()
        ],
        "direction_options": ["利好", "利空", "中性"],
        "status_options": STATUS_OPTIONS,
        "review_status_options": REVIEW_STATUS_OPTIONS,
        "classification_source_options": CLASSIFICATION_SOURCE_OPTIONS,
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
        "classification_groups": [
            (code, category.label, list(category.subcategories.items()))
            for code, category in event_registry.categories.items()
        ],
        "revisions": ClassificationRepository.get_revisions(session, announcement_id),
        "saved": request.query_params.get("saved") == "1",
    }
    return _templates(request).TemplateResponse(request, "announcement_detail.html", ctx)


@router.post(
    "/announcements/{announcement_id}/classification",
    name="review_announcement_classification",
)
async def review_announcement_classification(
    request: Request,
    announcement_id: int,
    session: Session = Depends(get_db),
):
    """人工确认/修正分类；手工解析 urlencoded，避免引入额外表单依赖。"""
    ann = AnnouncementRepository.get_by_id(session, announcement_id)
    if ann is None or ann.classification is None:
        raise HTTPException(status_code=404, detail="公告或分类不存在")

    raw = (await request.body()).decode("utf-8", errors="replace")
    form = {key: values[-1] for key, values in parse_qs(raw).items() if values}
    choice = form.get("classification_choice", "").strip()
    try:
        major_category, sub_category = choice.split(":", 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="缺少有效的分类值")
    category = event_registry.categories.get(major_category)
    if category is None or sub_category not in category.subcategories:
        raise HTTPException(status_code=400, detail="分类值不在当前分类体系中")

    try:
        expected_id = int(form["classification_id"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="缺少有效的分类版本标识")

    try:
        ClassificationRepository.manual_review(
            session,
            announcement_id=announcement_id,
            major_category=major_category,
            sub_category=sub_category,
            reviewed_by=form.get("reviewed_by", "local-user").strip() or "local-user",
            note=form.get("review_note", "").strip() or None,
            relevance=default_relevance(major_category, sub_category),
            expected_classification_id=expected_id,
        )
        session.commit()
    except ValueError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    target = request.url_for("announcement_detail", announcement_id=announcement_id)
    return RedirectResponse(f"{target}?saved=1", status_code=303)


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

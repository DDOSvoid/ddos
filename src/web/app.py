"""FastAPI 应用工厂。

`create_app()` 无 import / 构造副作用：不调用 init_db / get_engine，
因此测试导入并 override get_db 依赖时不会触碰真实 data/ddos.db。
表结构由 scripts/run_web.py 在 serve 前通过 init_db() 确保。
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.web import routes
from src.web.labels import category_label, field_label, sub_label, unit_label

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
ECHARTS_AVAILABLE = (STATIC_DIR / "vendor" / "echarts.min.js").exists()


def _fmt_field_value(field) -> str:
    """格式化提取字段的展示值（numeric 加千分位，其余按原文）。"""
    if field.field_type == "numeric":
        v = field.value_as_float()
        if v is not None:
            if float(v).is_integer():
                return f"{int(v):,}"
            return f"{v:,.4f}".rstrip("0").rstrip(".")
        return field.field_value or ""
    return field.field_value or ""


def create_app() -> FastAPI:
    """构建应用实例。"""
    app = FastAPI(title="DDOS 公告事件分析", docs_url=None, redoc_url=None)

    templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
    # 注册模板全局：标签映射 + 数值格式化 + ECharts 可用性
    templates.env.globals.update(
        echarts_available=ECHARTS_AVAILABLE,
        category_label=category_label,
        sub_label=sub_label,
        field_label=field_label,
        unit_label=unit_label,
    )
    templates.env.filters["fmt_value"] = _fmt_field_value

    app.state.templates = templates
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(routes.router)
    return app

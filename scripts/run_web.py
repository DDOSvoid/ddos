#!/usr/bin/env python
"""本地 Web 界面启动入口。

启动只读展示前端（仪表盘 / 公告列表 / 日报查看器）:

    python scripts/run_web.py                  # http://127.0.0.1:8000
    python scripts/run_web.py --port 9000
    python scripts/run_web.py --db-url sqlite:///data/empty.db
"""

import argparse
import sys
from pathlib import Path

# 确保 src 在 Python path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from src.database.engine import init_db, set_database_url
from src.utils.logging_config import setup_logging
from src.web.app import create_app


def main():
    parser = argparse.ArgumentParser(description="DDOS 只读 Web 界面")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    parser.add_argument("--db-url", default=None, help="覆盖数据库 URL")
    args = parser.parse_args()

    setup_logging()

    # 覆盖数据库 URL（须在首次建连前）
    if args.db_url:
        set_database_url(args.db_url)

    # 与 run_pipeline.py 一致：serve 前确保表存在（已存在则幂等）
    init_db()

    print(f"启动只读 Web 界面: http://{args.host}:{args.port}")
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

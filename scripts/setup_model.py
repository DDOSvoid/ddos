#!/usr/bin/env python
"""生产分类模型构建脚本 — 只接受真实、人工验证的时间分区数据。

`models/` 在 .gitignore 中（大文件不入 git），clone 后模型缺失。
合成种子数据只允许用于研究或代码冒烟测试，不能由本脚本用于生产训练和校准。

用法:
    python scripts/setup_model.py --data data/labeled/real_verified.jsonl
    python scripts/setup_model.py --data data/labeled/real_verified.jsonl --force
    python scripts/setup_model.py --check    # 仅检查模型是否就绪
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "bert-classifier"

# 模型就绪判定：权重文件可能是 model.safetensors 或 pytorch_model.bin
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
REQUIRED_MODEL_FILES = (
    "config.json",
    "label_mapping.json",
    "temperature.json",  # 校准产物，缺失则推理置信度偏低被 0.6 门槛挡掉
    "training_manifest.json",
    "evaluation_manifest.json",
    "tokenizer_config.json",
    "tokenizer.json",
)


def _run(step: str, cmd: list[str]) -> None:
    print(f"\n[setup_model] ▶ {step}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[setup_model] ✗ {step} 失败 (exit={result.returncode})")
        sys.exit(result.returncode)
    print(f"[setup_model] ✓ {step} 完成")


def _model_ready() -> bool:
    if not MODEL_DIR.exists():
        return False
    if not any((MODEL_DIR / w).exists() for w in _WEIGHT_FILES):
        return False
    if not all((MODEL_DIR / f).exists() for f in REQUIRED_MODEL_FILES):
        return False
    try:
        import json

        with open(MODEL_DIR / "training_manifest.json", encoding="utf-8") as f:
            training_manifest = json.load(f)
        with open(MODEL_DIR / "temperature.json", encoding="utf-8") as f:
            calibration = json.load(f)
        with open(MODEL_DIR / "evaluation_manifest.json", encoding="utf-8") as f:
            evaluation = json.load(f)
    except (OSError, ValueError):
        return False
    return (
        training_manifest.get("data_contract") == "real-label-v1"
        and training_manifest.get("split_strategy") == "declared_strict_temporal"
        and calibration.get("calibration_contract") == "real-label-v1"
        and evaluation.get("evaluation_contract") == "real-label-v1"
        and evaluation.get("production_gate_pass") is True
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="复现 BERT 分类器（gitignore + 构建脚本方案）")
    parser.add_argument("--force", action="store_true",
                        help="即使模型已存在也强制重建")
    parser.add_argument("--check", action="store_true",
                        help="仅检查模型是否就绪")
    parser.add_argument(
        "--data",
        type=Path,
        help="real-label-v1 真实人工验证 JSONL（含严格 train/validation/test 分区）",
    )
    args = parser.parse_args()

    if args.check:
        ready = _model_ready()
        print("生产模型就绪" if ready else "生产模型缺失或仍为合成数据模型")
        sys.exit(0 if ready else 1)

    if _model_ready() and not args.force:
        print(f"[setup_model] 模型已存在: {MODEL_DIR}（跳过构建，--force 强制重建）")
        return

    if args.data is None:
        parser.error("生产构建必须提供 --data，禁止默认使用合成种子数据")
    data_path = args.data.resolve()
    if not data_path.exists():
        parser.error(f"真实标注文件不存在: {data_path}")

    # 依赖预检
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        print(f"[setup_model] ✗ 缺少训练依赖: {e}")
        print("   请先安装: pip install -r requirements.txt")
        sys.exit(1)

    # 1. 用真实人工标注数据微调 BERT
    _run("微调分类器",
         [sys.executable, "-m", "src.training.train_classifier",
          "--data", str(data_path), "--output", str(MODEL_DIR)])

    # 2. 只用时间外真实 validation 分区校准
    _run("温度校准",
         [sys.executable, str(PROJECT_ROOT / "scripts" / "calibrate_temperature.py"),
          "--data", str(data_path), "--model", str(MODEL_DIR)])

    # 3. 严格时间外 test 门禁；未达标时模型不具备生产资格。
    _run(
        "真实时间外评估",
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_classifier_gate.py"),
            "--data",
            str(data_path),
            "--model",
            str(MODEL_DIR),
        ],
    )

    if _model_ready():
        print(f"\n[setup_model] ✓ 模型复现完成: {MODEL_DIR}")
    else:
        print("\n[setup_model] ⚠ 模型不完整，请检查上方输出")
        sys.exit(1)


if __name__ == "__main__":
    main()

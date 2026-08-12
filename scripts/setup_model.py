#!/usr/bin/env python
"""模型复现脚本 — clone 后一键恢复 BERT 分类器。

`models/` 在 .gitignore 中（大文件不入 git），clone 后模型缺失。
本脚本依次执行 种子数据生成 → 微调 → 温度校准 三步，复现
`models/bert-classifier`（含 temperature.json 校准产物）。

用法:
    python scripts/setup_model.py            # 缺模型才构建，已存在则跳过
    python scripts/setup_model.py --force    # 强制重新构建
    python scripts/setup_model.py --check    # 仅检查模型是否就绪
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "bert-classifier"
SEED_DATA = PROJECT_ROOT / "data" / "labeled" / "seed.jsonl"

# 模型就绪判定：权重文件可能是 model.safetensors 或 pytorch_model.bin
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
REQUIRED_MODEL_FILES = (
    "config.json",
    "label_mapping.json",
    "temperature.json",  # 校准产物，缺失则推理置信度偏低被 0.6 门槛挡掉
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
    return all((MODEL_DIR / f).exists() for f in REQUIRED_MODEL_FILES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="复现 BERT 分类器（gitignore + 构建脚本方案）")
    parser.add_argument("--force", action="store_true",
                        help="即使模型已存在也强制重建")
    parser.add_argument("--check", action="store_true",
                        help="仅检查模型是否就绪")
    args = parser.parse_args()

    if args.check:
        ready = _model_ready()
        print("模型就绪" if ready else "模型缺失")
        sys.exit(0 if ready else 1)

    if _model_ready() and not args.force:
        print(f"[setup_model] 模型已存在: {MODEL_DIR}（跳过构建，--force 强制重建）")
        return

    # 依赖预检
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        print(f"[setup_model] ✗ 缺少训练依赖: {e}")
        print("   请先安装: pip install -r requirements.txt")
        sys.exit(1)

    # 1. 生成种子标注数据（从 event_types.yaml 模板合成，确定性可复现）
    _run("生成种子数据",
         [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_seed_data.py")])

    # 2. 微调 BERT（固定随机种子，可复现）
    _run("微调分类器",
         [sys.executable, "-m", "src.training.train_classifier",
          "--data", str(SEED_DATA), "--output", str(MODEL_DIR)])

    # 3. 温度校准（锐化 softmax 置信度，否则种子模型置信度 0.1~0.25 低于
    #    extraction_min_confidence=0.6，提取阶段会空转）
    _run("温度校准",
         [sys.executable, str(PROJECT_ROOT / "scripts" / "calibrate_temperature.py"),
          "--data", str(SEED_DATA), "--model", str(MODEL_DIR)])

    if _model_ready():
        print(f"\n[setup_model] ✓ 模型复现完成: {MODEL_DIR}")
    else:
        print("\n[setup_model] ⚠ 模型不完整，请检查上方输出")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""工具函数 — 中文文本清洗。"""

import re
from typing import Optional


def clean_html(text: str) -> str:
    """移除 HTML 标签，保留纯文本。"""
    if not text:
        return ""
    # 移除 script/style 标签及其内容
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 移除 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 解码常见 HTML 实体
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    return text


def normalize_whitespace(text: str) -> str:
    """规范化空白字符：合并连续空格/换行/制表符。"""
    if not text:
        return ""
    # 所有空白字符替换为单个空格
    text = re.sub(r"\s+", " ", text)
    # 移除首尾空格
    return text.strip()


def remove_control_chars(text: str) -> str:
    """移除控制字符（保留换行和制表符）。"""
    if not text:
        return ""
    # 移除除了 \n \t 之外的控制字符
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)


def clean_chinese_text(text: str) -> str:
    """完整的文本清洗流程：HTML → 空白规范化 → 控制字符清理。"""
    text = clean_html(text)
    text = normalize_whitespace(text)
    text = remove_control_chars(text)
    return text


def truncate_for_model(text: str, max_chars: int = 2000) -> str:
    """按字符数截断文本（适配模型输入窗口）。

    中文字符约占 1.5-2 tokens。512 tokens ≈ 256-340 中文字符。
    保守截断到 2000 字符用于 BERT（实际 tokenizer 会再截断）。
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# ── 单位换算 ───────────────────────────────────────────────────


def parse_chinese_number(text: str) -> Optional[float]:
    """解析中文数字表达为浮点数。

    支持格式:
      "1.5亿" → 150000000
      "2000万" → 20000000
      "5000" → 5000
      "3,000万" → 30000000
      "1.2%" → 0.012
    """
    if not text:
        return None

    text = text.strip().replace(",", "").replace("，", "").replace(" ", "")

    # 百分比
    if text.endswith("%"):
        try:
            return float(text[:-1]) / 100.0
        except ValueError:
            return None

    # 亿/万
    multiplier = 1.0
    if "亿" in text:
        multiplier = 100_000_000
        text = text.replace("亿", "")
    elif "万" in text:
        multiplier = 10_000
        text = text.replace("万", "")

    # "约"、"近" 等模糊前缀
    text = text.lstrip("约近超过超逾不多于至少不低于不超过不高于")

    try:
        return float(text) * multiplier
    except ValueError:
        return None


def normalize_amount(value: float, unit: str) -> float:
    """将金额统一化为万元。"""
    unit_lower = unit.lower().strip()
    if unit_lower in ("cny_100m", "亿", "亿元"):
        return value * 10_000  # 亿 → 万
    elif unit_lower in ("cny", "元"):
        return value / 10_000  # 元 → 万
    # 默认认为是万元
    return value


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法，b 为 0 时返回 default。"""
    if b == 0 or abs(b) < 1e-10:
        return default
    return a / b


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """将值钳制在 [low, high] 区间。"""
    return max(low, min(high, value))

"""统一 LLM 客户端 — 使用 DeepSeek 的 OpenAI-compatible API。

双层策略:
  - 提取模型（DeepSeek）: 批量提取字段
  - 分析模型（DeepSeek）: 深度分析高影响事件
"""

import json
import os
import time
from typing import Literal

from openai import OpenAI, OpenAIError

ThinkingMode = Literal["enabled", "disabled"]


class LlmClient:
    """LLM 客户端 — OpenAI SDK 封装。

    Usage:
        client = LlmClient()
        text = client.complete("你是AI助手", "请解析以下公告...", model="deepseek-v4-flash")
        data = client.complete_json("你是AI助手", "输出JSON...")
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        default_thinking: ThinkingMode = "disabled",
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """
        Args:
            api_key: DeepSeek API key（优先从 DEEPSEEK_API_KEY 读取）
            base_url: API 基础 URL（默认从 OPENAI_BASE_URL 环境变量读取）
            default_model: 默认模型名
            default_thinking: 默认思考模式
        """
        api_key = (
            api_key
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY", "")
        )
        base_url = (
            base_url
            or os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
        )

        try:
            client_options = {
                "api_key": api_key,
                "base_url": base_url,
                "timeout": timeout,
            }
            if max_retries is not None:
                client_options["max_retries"] = max_retries
            self.client = OpenAI(**client_options)
        except OpenAIError:
            # key 未配置：构造成功，首次实际调用时抛出清晰错误
            # （保持 .env 中 key 可暂留空、代码可导入/可测试）
            self.client = None
        self.default_model = default_model or "deepseek-v4-flash"
        self.default_thinking = default_thinking

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        response_format: str | None = None,  # "json_object"
        thinking: ThinkingMode | None = None,
    ) -> str:
        """发送请求，返回原始文本。

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词（包含公告文本）
            model: 模型名（默认使用构造时的 default_model）
            max_tokens: 最大输出 token 数
            temperature: 温度（0=确定性输出）
            response_format: 输出格式 "json_object" 等
            thinking: 思考模式；默认使用构造时配置

        Returns:
            模型返回的文本
        """
        if self.client is None:
            raise RuntimeError(
                "DeepSeek API key 未配置：请在 .env 中设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY"
            )

        model = model or self.default_model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        effective_thinking = thinking or self.default_thinking
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": {"thinking": {"type": effective_thinking}},
        }
        if response_format:
            kwargs["response_format"] = {"type": response_format}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        retries: int = 3,
        thinking: ThinkingMode | None = None,
    ) -> dict:
        """请求并解析 JSON 响应（自动重试格式错误）。

        Args:
            retries: JSON 解析失败时的重试次数

        Returns:
            解析后的 dict

        Raises:
            ValueError: 所有重试均失败
        """
        for attempt in range(retries):
            try:
                raw = self.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format="json_object",
                    thinking=thinking,
                )
                return json.loads(raw)
            except (json.JSONDecodeError, KeyError) as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
                else:
                    raise ValueError(f"Failed to parse JSON after {retries} attempts: {e}") from e
        return {}

    def complete_json_audited(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        thinking: ThinkingMode = "disabled",
    ) -> tuple[dict, dict]:
        """Make one stateless JSON request and return reproducibility metadata."""
        if self.client is None:
            raise RuntimeError(
                "DeepSeek API key 未配置：请在 .env 中设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY"
            )
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "extra_body": {"thinking": {"type": thinking}},
        }
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise ValueError(f"LLM response did not finish normally: {choice.finish_reason}")
        content = choice.message.content or ""
        parsed = json.loads(content)
        usage = response.usage.model_dump() if response.usage is not None else None
        metadata = {
            "response_id": response.id,
            "model_returned": response.model,
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "finish_reason": choice.finish_reason,
            "usage": usage,
            "raw_response_sha256": __import__("hashlib").sha256(
                content.encode()
            ).hexdigest(),
        }
        return parsed, metadata

    def complete_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        max_retries: int = 3,
        thinking: ThinkingMode | None = None,
    ) -> str:
        """带自动重试的 API 调用（处理速率限制和瞬时错误）。"""
        for attempt in range(max_retries):
            try:
                return self.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking=thinking,
                )
            except Exception:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
                else:
                    raise

    @property
    def extraction_model(self) -> str:
        """从配置获取提取模型名。"""
        from src.config import config
        return config.models.extraction.model

    @property
    def analysis_model(self) -> str:
        """从配置获取分析模型名。"""
        from src.config import config
        return config.models.analysis.model

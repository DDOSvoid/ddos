"""OpenAI client configuration and request-shape regression tests."""

from types import SimpleNamespace

from src.ml.llm_client import LlmClient


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="response-test",
            model=kwargs["model"],
            system_fingerprint="fingerprint-test",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='{"ok": true}'),
                )
            ],
            usage=SimpleNamespace(
                model_dump=lambda: {"prompt_tokens": 2, "completion_tokens": 3}
            ),
        )


def _client_with_fake_api() -> tuple[LlmClient, _FakeCompletions]:
    client = LlmClient(
        api_key="test-key",
        base_url="https://api.deepseek.com",
    )
    completions = _FakeCompletions()
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return client, completions


def test_deepseek_defaults(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client = LlmClient(api_key="test-key")

    assert client.default_model == "deepseek-v4-flash"
    assert client.default_thinking == "disabled"
    assert str(client.client.base_url).rstrip("/") == "https://api.deepseek.com"


def test_complete_uses_deepseek_request_parameters():
    client, completions = _client_with_fake_api()

    result = client.complete("system", "user", max_tokens=123)

    assert result == '{"ok": true}'
    assert completions.kwargs["model"] == "deepseek-v4-flash"
    assert completions.kwargs["max_tokens"] == 123
    assert completions.kwargs["temperature"] == 0.0
    assert completions.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    assert "max_completion_tokens" not in completions.kwargs
    assert "reasoning_effort" not in completions.kwargs


def test_audited_json_uses_disabled_deepseek_thinking():
    client, completions = _client_with_fake_api()

    parsed, metadata = client.complete_json_audited(
        system_prompt="system",
        user_prompt="user",
        model="deepseek-v4-flash",
        max_tokens=456,
        thinking="disabled",
    )

    assert parsed == {"ok": True}
    assert metadata["model_returned"] == "deepseek-v4-flash"
    assert completions.kwargs["max_tokens"] == 456
    assert completions.kwargs["temperature"] == 0.0
    assert completions.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    assert "reasoning_effort" not in completions.kwargs

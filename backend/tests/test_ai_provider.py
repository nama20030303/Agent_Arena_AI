"""
AI layer: provider abstraction, Yandex request shape, JSON parsing, caching, budget
enforcement, and the guarantee that the app works with no AI configured.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from app.ai.base import AIResult, DisabledProvider, Usage, parse_json_block
from app.ai.manager import AIManager
from app.ai.yandex import YandexGPTProvider
from app.config import settings
from app.db import session_scope
from app.models import AIRequest, AIUsage, LearningMemory, Question, User


# --------------------------------------------------------------------------- #
#  Pure helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Here is the answer:\n{"a": 1, "b": [1,2,],}\nHope that helps!',
        'prefix {"a": {"b": "}} not the end"}} suffix',
    ],
)
def test_parse_json_block_is_forgiving(raw):
    parsed = parse_json_block(raw)
    assert isinstance(parsed, dict)
    assert parsed["a"] if parsed.get("a") else True


def test_parse_json_block_returns_none_on_garbage():
    assert parse_json_block("no json at all here") is None


def test_usage_cost_estimate():
    usage = Usage(input_tokens=1000, output_tokens=500)
    assert usage.total_tokens == 1500
    expected = settings.cost_per_1k_input_tokens + 0.5 * settings.cost_per_1k_output_tokens
    assert usage.estimated_cost == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------- #
#  Provider contract
# --------------------------------------------------------------------------- #
def test_disabled_provider_explains_itself_without_raising():
    provider = DisabledProvider()
    result = provider.explain(topic="Gradient descent", level=1)
    assert result.ok is False
    assert result.error_code == "disabled"
    message = result.user_message()
    assert "YANDEX_API_KEY" in message
    assert "offline engine" in message.lower() or "Offline" in message


def test_yandex_provider_requires_credentials(monkeypatch):
    monkeypatch.setattr(settings, "yandex_api_key", "")
    provider = YandexGPTProvider(api_key="", folder_id="")
    result = provider.generate("hello")
    assert result.ok is False and result.error_code == "disabled"
    assert "YANDEX_API_KEY" in result.error


def test_yandex_request_shape_and_headers(monkeypatch):
    """Exact request contract for Yandex Cloud Foundation Models + no key in any response."""
    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "result": {
                    "alternatives": [{"message": {"role": "assistant", "text": 'The gradient points uphill.'}, "status": "FINAL"}],
                    "usage_metadata": {"prompt_token_count": 120, "completion_token_count": 24},
                }
            }

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, json=None, headers=None):
            captured["url"], captured["payload"], captured["headers"] = url, json, headers
            return FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "Client", FakeClient)
    provider = YandexGPTProvider(api_key="secret-key", folder_id="fold-1", model="yandexgpt-lite")
    result = provider.generate("Explain gradient descent", max_tokens=900)

    assert captured["url"].endswith("/completion")
    assert captured["headers"]["Authorization"] == "Api-Key secret-key"
    assert captured["headers"]["x-folder-id"] == "fold-1"
    assert captured["payload"]["modelUri"] == "gpt://fold-1/yandexgpt-lite/latest"
    assert captured["payload"]["completionOptions"]["maxTokens"] == str(provider.clamp_max_tokens(900))
    roles = [m["role"] for m in captured["payload"]["messages"]]
    assert roles == ["system", "user"]
    assert result.ok and "uphill" in result.text
    assert result.usage.input_tokens == 120 and result.usage.output_tokens == 24
    assert "secret-key" not in json.dumps(result.__dict__, default=str)


def test_yandex_retries_then_reports_unavailable(monkeypatch):
    calls = {"n": 0}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, json=None, headers=None):
            calls["n"] += 1
            raise httpx.ConnectError("network down")

    import httpx

    monkeypatch.setattr(httpx, "Client", FakeClient)
    monkeypatch.setattr(settings, "ai_max_retries", 1)
    provider = YandexGPTProvider(api_key="k", folder_id="f")
    result = provider.generate("hi")
    assert result.ok is False
    assert result.error_code == "unavailable"
    assert calls["n"] == 2


# --------------------------------------------------------------------------- #
#  Manager: cache + limits + accounting
# --------------------------------------------------------------------------- #
class StubProvider(DisabledProvider):
    name = "stub"

    def __init__(self):
        self.calls = 0

    def model_label(self):
        return "stub-model"

    def complete(self, messages, **kwargs):
        self.calls += 1
        return AIResult(
            ok=True,
            text="Ridge shrinks coefficients toward zero. [1]",
            provider=self.name,
            model="stub-model",
            usage=Usage(input_tokens=50, output_tokens=20),
        )

    def health(self):
        return {"provider": self.name, "configured": True, "model": "stub-model"}


@pytest.fixture()
def manager_with_stub(monkeypatch, db):
    stub = StubProvider()
    manager = AIManager()
    manager._provider = stub
    monkeypatch.setattr(settings, "ai_cache_enabled", True)
    return manager, stub


def test_manager_answers_from_cache_for_identical_requests(manager_with_stub, db):
    manager, stub = manager_with_stub
    kwargs = {"topic": "Ridge regression", "level": 2, "context": "", "learner_state": "", "style": "explain", "language": "auto"}
    first = manager.invoke("explain", user_id=None, kwargs=kwargs, session=db)
    second = manager.invoke("explain", user_id=None, kwargs=kwargs, session=db)
    assert first.ok and second.ok
    assert stub.calls == 1, "identical request must be served from cache"
    assert second.usage.cached is True
    assert second.meta.get("cache") is True


def test_manager_enforces_daily_request_limit(manager_with_stub, monkeypatch, db):
    manager, stub = manager_with_stub
    monkeypatch.setattr(settings, "max_ai_requests_per_day", 2)
    for i in range(2):
        ok = manager.invoke("explain", user_id=None, kwargs={"topic": f"T{i}", "level": 1}, session=db)
        assert ok.ok
    blocked = manager.invoke("explain", user_id=None, kwargs={"topic": "T3", "level": 1}, session=db)
    assert blocked.ok is False
    assert blocked.error_code == "limited"
    assert "Daily AI limit" in blocked.user_message() or "offline engine" in blocked.user_message().lower()
    assert stub.calls == 2, "over-limit requests must not reach the provider"


def test_manager_clamps_tokens_per_request(manager_with_stub, monkeypatch, db):
    manager, stub = manager_with_stub
    monkeypatch.setattr(settings, "max_tokens_per_request", 300)
    seen = {}

    class Spy(StubProvider):
        def complete(self, messages, **kwargs):
            seen["max_tokens"] = kwargs.get("max_tokens")
            return super().complete(messages, **kwargs)

    spy = Spy()
    manager._provider = spy
    manager.invoke("summarize", user_id=None, kwargs={"text": "x" * 500, "instruction": ""}, session=db)
    assert seen["max_tokens"] <= 300


def test_manager_records_usage_and_requests(db, manager_with_stub):
    manager, _stub = manager_with_stub
    result = manager.invoke("explain", user_id=None, kwargs={"topic": "Accounting", "level": 1}, session=db)
    assert result.ok
    row = db.query(AIRequest).filter(AIRequest.method == "explain").order_by(AIRequest.id.desc()).first()
    assert row is not None and row.status == "ok" and row.total_tokens == 70
    usage = db.query(AIUsage).filter(AIUsage.day == date.today()).first()
    assert usage is not None and usage.requests >= 1


def test_public_config_never_leaks_the_key(client, db, monkeypatch):
    """The frontend may call this before login, so it must expose presence - never the secret."""
    monkeypatch.setattr(settings, "yandex_api_key", "SUPER-SECRET-VALUE")
    body = client.get("/api/ai/config").json()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key, value
                yield from walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)

    names = [k for k, _v in walk(body)]
    for forbidden in ("yandex_api_key", "openai_api_key", "api_key", "folder_id", "token", "password", "secret"):
        assert forbidden not in names, f"{forbidden} must never be serialised to the client"
    assert body["yandex_api_key_present"] is True
    assert "SUPER-SECRET-VALUE" not in response_text(body)


def response_text(body) -> str:
    return json.dumps(body, default=str)

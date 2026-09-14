"""
OpenAI-compatible provider (works with OpenAI, OpenRouter, LiteLLM, vLLM, LocalAI,
Yandex AI Studio, Yandex GPT via its OpenAI-compatible gateway, and most Russian
gateways that expose /v1/chat/completions).
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import httpx

from app.ai.base import AIProvider, AIResult, Usage
from app.config import settings

logger = logging.getLogger(__name__)


class OpenAICompatProvider(AIProvider):
    name = "openai"
    supports_embeddings = True
    supports_json_mode = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        timeout: float | None = None,
        name: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.model = model or settings.openai_model
        self.embedding_model = embedding_model or settings.openai_embedding_model
        self.timeout = timeout or settings.ai_timeout_seconds
        if name:
            self.name = name

    def model_label(self) -> str:
        return self.model

    def health(self) -> dict[str, Any]:
        return {"provider": self.name, "configured": bool(self.api_key or "localhost" in self.base_url), "model": self.model, "endpoint": self.base_url}

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> AIResult:
        max_tokens = self.clamp_max_tokens(max_tokens)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": round(float(settings.yandex_temperature if temperature is None else temperature), 2),
            "stream": False,
        }
        # most gateways accept max_tokens; some newer ones want max_completion_tokens
        payload["max_tokens"] = max_tokens
        if json_mode and self.supports_json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/chat/completions"
        started = time.perf_counter()
        last_error = ""
        for attempt in range(max(1, settings.ai_max_retries + 1)):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload, headers=self._headers())
                if response.status_code in (429, 500, 502, 503, 504) and attempt < settings.ai_max_retries:
                    time.sleep(min(8.0, 0.6 * (2**attempt) + random.random() * 0.4))
                    last_error = f"HTTP {response.status_code}"
                    continue
                if response.status_code >= 400:
                    return AIResult(ok=False, error_code="http", error=f"HTTP {response.status_code}: {response.text[:400]}",
                                    provider=self.name, model=self.model)
                data = response.json()
                choice = (data.get("choices") or [{}])[0]
                text = ((choice.get("message") or {}).get("content") or choice.get("text") or "").strip()
                usage_raw = data.get("usage") or {}
                usage = Usage(
                    input_tokens=int(usage_raw.get("prompt_tokens") or 0),
                    output_tokens=int(usage_raw.get("completion_tokens") or 0),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
                if not usage.input_tokens and not usage.output_tokens:
                    usage = self.estimate_usage(messages, text)
                    usage.latency_ms = int((time.perf_counter() - started) * 1000)
                if not text:
                    return AIResult(ok=False, error_code="parse", error="empty completion", provider=self.name, model=self.model)
                return AIResult(ok=True, text=text, provider=self.name, model=self.model, usage=usage)
            except httpx.TimeoutException:
                last_error = f"timeout after {self.timeout}s"
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("OpenAI-compatible call failed: %s", last_error)
                break
        return AIResult(ok=False, error_code="unavailable", error=last_error or "request failed", provider=self.name, model=self.model)

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - network
        url = f"{self.base_url}/embeddings"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json={"model": self.embedding_model, "input": texts}, headers=self._headers())
        if response.status_code >= 400:
            raise RuntimeError(f"embeddings HTTP {response.status_code}: {response.text[:300]}")
        items = response.json().get("data") or []
        return [list(map(float, item.get("embedding") or [])) for item in items]


class OllamaProvider(OpenAICompatProvider):
    """Local model server (llama.cpp / ollama) - no API key, offline, zero marginal cost."""

    name = "ollama"

    def __init__(self) -> None:
        super().__init__(
            api_key="",
            base_url=f"{settings.ollama_base_url.rstrip('/')}/v1",
            model=settings.ollama_model,
            name="ollama",
        )
        self.supports_json_mode = True

    def health(self) -> dict[str, Any]:  # pragma: no cover - network
        try:
            with httpx.Client(timeout=4.0) as client:
                ok = client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags").status_code == 200
        except Exception:  # noqa: BLE001
            ok = False
        return {"provider": self.name, "configured": ok, "model": self.model, "endpoint": settings.ollama_base_url}

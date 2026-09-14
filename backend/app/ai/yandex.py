"""
YandexGPT provider.

Two flavours, selected by YANDEX_API_FLAVOUR:
  cloud     - Yandex Cloud Foundation Models gRPC-over-HTTP API
              (Authorization: Api-Key <oauth/iam api key> + x-folder-id)
  aistudio  - Yandex AI Studio, OpenAI-compatible /chat/completions

Keys come from the environment only. They are never returned by any endpoint and
never reach the browser (see /api/ai/config, which exposes presence booleans only).
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

import httpx

from app.ai.base import AIProvider, AIResult, Usage
from app.config import settings

logger = logging.getLogger(__name__)


class YandexGPTProvider(AIProvider):
    name = "yandex"
    supports_embeddings = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        folder_id: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        flavour: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.yandex_api_key
        self.folder_id = folder_id if folder_id is not None else settings.yandex_folder_id
        self.base_url = (base_url or settings.yandex_api_url).rstrip("/")
        self.flavour = (flavour or settings.yandex_api_flavour).lower()
        self.model = model or settings.yandex_model
        self.timeout = timeout or settings.ai_timeout_seconds
        if self.flavour == "cloud":
            self._model_uri = f"gpt://{self.folder_id}/{self.model}/{settings.yandex_model_version}"
            self._embedding_uri = f"embedding://{self.folder_id}/{settings.yandex_embedding_model}/latest"
        else:
            self._model_uri = self.model
            self._embedding_uri = ""

    def model_label(self) -> str:
        return self._model_uri

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "configured": bool(self.api_key and (self.folder_id or self.flavour != "cloud")),
            "model": self._model_uri,
            "flavour": self.flavour,
            "endpoint": self.base_url,
            "embeddings": self._embedding_uri if self.flavour == "cloud" else "n/a (AI Studio)",
        }

    # ------------------------------------------------------------------ core
    def _headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Api-Key {self.api_key}", "Content-Type": "application/json"}
        if self.flavour == "cloud" and self.folder_id:
            headers["x-folder-id"] = self.folder_id
        return headers

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> AIResult:
        if not self.api_key:
            return AIResult(ok=False, error_code="disabled", error="YANDEX_API_KEY is not set", provider=self.name, model=self._model_uri)
        if self.flavour == "cloud" and not self.folder_id:
            return AIResult(ok=False, error_code="disabled", error="YANDEX_FOLDER_ID is not set", provider=self.name, model=self._model_uri)

        max_tokens = self.clamp_max_tokens(max_tokens)
        temperature = settings.yandex_temperature if temperature is None else temperature
        if json_mode:
            # YandexGPT has no native JSON mode: reinforce it in the system message.
            reinforced = list(messages)
            if reinforced and reinforced[0].get("role") == "system":
                reinforced[0] = {**reinforced[0], "content": reinforced[0]["content"] + "\nReturn strictly valid JSON, no prose, no markdown fences."}
            else:
                reinforced = [{"role": "system", "content": "Return strictly valid JSON, no prose, no markdown fences."}, *messages]
            messages = reinforced

        url, payload = self._request_shape(messages, max_tokens, temperature)
        started = time.perf_counter()
        attempts = max(1, settings.ai_max_retries + 1)
        last_error = ""
        for attempt in range(attempts):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload, headers=self._headers())
                if response.status_code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                    delay = min(8.0, 0.6 * (2**attempt) + random.random() * 0.4)
                    logger.warning("YandexGPT HTTP %s; retrying in %.1fs", response.status_code, delay)
                    time.sleep(delay)
                    last_error = f"HTTP {response.status_code}"
                    continue
                if response.status_code >= 400:
                    detail = _error_detail(response)
                    return AIResult(
                        ok=False,
                        error_code="http",
                        error=f"YandexGPT HTTP {response.status_code}: {detail}",
                        provider=self.name,
                        model=self._model_uri,
                        usage=self.estimate_usage(messages, ""),
                    )
                data = response.json()
                text, usage = self._parse_completion(data, messages)
                latency = int((time.perf_counter() - started) * 1000)
                usage.latency_ms = latency
                if not usage.input_tokens and not usage.output_tokens:
                    usage = self.estimate_usage(messages, text)
                return AIResult(ok=True, text=text, provider=self.name, model=self._model_uri, usage=usage,
                                meta={"attempts": attempt + 1})
            except httpx.TimeoutException:
                last_error = f"timeout after {self.timeout}s"
                if attempt < attempts - 1:
                    continue
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < attempts - 1:
                    continue
            except Exception as exc:  # noqa: BLE001
                last_error = f"unexpected {type(exc).__name__}: {exc}"
                logger.exception("YandexGPT call failed")
                break
        return AIResult(
            ok=False,
            error_code="unavailable",
            error=last_error or "YandexGPT request failed",
            provider=self.name,
            model=self._model_uri,
            usage=self.estimate_usage(messages, ""),
        )

    def _request_shape(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> tuple[str, dict[str, Any]]:
        if self.flavour == "aistudio":
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages],
                "max_tokens": max_tokens,
                "temperature": round(float(temperature), 2),
                "stream": False,
            }
            return url, payload
        url = f"{self.base_url}/completion"
        payload = {
            "modelUri": self._model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": round(float(temperature), 2),
                "maxTokens": str(max_tokens),
            },
            "messages": [
                {"role": "system" if m.get("role") == "system" else "user", "text": m.get("content", "")}
                for m in messages
            ],
        }
        return url, payload

    @staticmethod
    def _parse_completion(data: dict[str, Any], messages: list[dict[str, str]]) -> tuple[str, Usage]:
        usage = Usage()
        if not isinstance(data, dict):
            return "", usage
        result = data.get("result") or data
        text = ""
        alternatives = result.get("alternatives")
        if alternatives:
            text = (alternatives[0].get("message") or {}).get("text", "") or ""
            if not text and len(alternatives) > 1:
                text = (alternatives[-1].get("message") or {}).get("text", "") or ""
        elif result.get("choices"):
            text = (result["choices"][0].get("message") or {}).get("content", "") or ""
        else:
            text = result.get("text") or result.get("content") or ""
        # Yandex has used several spellings across Cloud / AI Studio versions - accept them all.
        metric_keys = ("usageMetrics", "usage_metrics", "usage_metadata", "usage")
        input_keys = ("inputTextTokens", "input_text_tokens", "promptTokenCount", "prompt_token_count", "prompt_tokens")
        output_keys = (
            "completionTextTokens", "completion_text_tokens", "completionTokenCount",
            "completion_token_count", "completion_tokens", "total_tokens",
        )
        for key in metric_keys:
            metrics = result.get(key) or data.get(key)
            if isinstance(metrics, dict):
                for name in input_keys:
                    if metrics.get(name):
                        usage.input_tokens = int(metrics[name])
                        break
                for name in output_keys:
                    if metrics.get(name):
                        usage.output_tokens = int(metrics[name])
                        break
                break
        return str(text).strip(), usage

    # ------------------------------------------------------------ embeddings
    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - network
        if not self.api_key or self.flavour != "cloud":
            raise RuntimeError("Yandex embeddings need YANDEX_API_KEY + YANDEX_FOLDER_ID with flavour=cloud")
        url = f"{self.base_url}/embedding"
        payload = {"modelUri": self._embedding_uri, "texts": texts, "textTypes": ["query" if len(texts) == 1 else "passage"] * len(texts)}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload, headers=self._headers())
        if response.status_code >= 400:
            raise RuntimeError(f"Yandex embedding HTTP {response.status_code}: {_error_detail(response)}")
        data = response.json()
        embeddings = data.get("embeddings") or data.get("result", {}).get("embeddings") or []
        return [list(map(float, item.get("vector") or item.get("embedding") or [])) for item in embeddings]


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            err = payload.get("error") or payload
            if isinstance(err, dict):
                return str(err.get("message") or err.get("code") or err)[:400]
            return str(err)[:400]
    except Exception:  # noqa: BLE001
        pass
    return (response.text or "")[:400]

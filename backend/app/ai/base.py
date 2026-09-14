"""
AIProvider: the single seam between the product and any LLM backend.

Business logic (learning engine, teacher, evaluators) only ever talks to this
abstract interface, so YandexGPT can be replaced by a local model, an
OpenAI-compatible endpoint or another Russian provider through env vars only.

The class also implements every *semantic* method required by the spec on top of
one primitive (``complete``), so adding a provider = implementing 1-2 methods.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.ai import prompts
from app.config import settings
from app.knowledge.textutil import estimate_tokens

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Value objects
# --------------------------------------------------------------------------- #
@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost(self) -> float:
        return (
            self.input_tokens / 1000.0 * settings.cost_per_1k_input_tokens
            + self.output_tokens / 1000.0 * settings.cost_per_1k_output_tokens
        )

    @property
    def currency(self) -> str:
        return settings.cost_currency


@dataclass
class AIResult:
    ok: bool
    text: str = ""
    data: Any = None
    error: str = ""
    error_code: str = ""  # disabled | limited | unavailable | timeout | http | parse
    provider: str = "none"
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def unavailable(self) -> bool:
        return not self.ok and self.error_code in {"disabled", "unavailable", "limited", "timeout", "http"}

    def user_message(self) -> str:
        if self.ok:
            return ""
        return prompts.ai_unavailable_message(self.error_code, self.error)


# --------------------------------------------------------------------------- #
#  JSON handling (LLMs love to wrap JSON in prose)
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json_block(text: str) -> Any | None:
    if not text:
        return None
    candidates: list[str] = []
    for match in _FENCE_RE.finditer(text):
        candidates.append(match.group(1))
    candidates.append(text)
    start = text.find("{")
    arr_start = text.find("[")
    if start != -1 and (arr_start == -1 or start < arr_start):
        candidates.append(_balanced(text, start, "{", "}"))
    elif arr_start != -1:
        candidates.append(_balanced(text, arr_start, "[", "]"))
    for candidate in candidates:
        candidate = (candidate or "").strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    return None


def _balanced(text: str, start: int, open_ch: str, close_ch: str) -> str:
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


# --------------------------------------------------------------------------- #
#  Provider
# --------------------------------------------------------------------------- #
class AIProvider(ABC):
    name: str = "base"
    supports_embeddings: bool = False
    supports_json_mode: bool = False

    # ---------------------------------------------------------------- low level
    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> AIResult:
        """Send a chat completion request. Single point a provider must implement."""

    def embed(self, texts: list[str]) -> list[list[float]] | None:  # pragma: no cover - optional
        return None

    def health(self) -> dict[str, Any]:
        return {"provider": self.name, "configured": True, "model": self.model_label()}

    def model_label(self) -> str:
        return ""

    # -------------------------------------------------------------- semantics
    def _ask(
        self,
        *,
        system: str,
        user: str,
        method: str,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AIResult:
        result = self.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode or self.supports_json_mode,
        )
        result.meta.setdefault("method", method)
        if result.ok and json_mode:
            parsed = parse_json_block(result.text)
            if parsed is None:
                result.ok = False
                result.error_code = "parse"
                result.error = "Model did not return valid JSON"
                result.meta["raw"] = result.text[:800]
            else:
                result.data = parsed
        return result

    def generate(self, prompt: str, *, system: str | None = None, context: str = "", json_mode: bool = False,
                 max_tokens: int | None = None) -> AIResult:
        sys_prompt = system or prompts.GENERAL_SYSTEM
        user = prompts.with_context(prompt, context) if context else prompt
        return self._ask(system=sys_prompt, user=user, method="generate", json_mode=json_mode, max_tokens=max_tokens,
                        temperature=settings.yandex_temperature)

    def explain(self, *, topic: str, level: int, context: str = "", learner_state: str = "", style: str = "explain",
                language: str = "auto", extra: str = "", max_tokens: int | None = None) -> AIResult:
        return self._ask(
            system=prompts.TEACHER_SYSTEM,
            user=prompts.explain_prompt(topic=topic, level=level, context=context, learner_state=learner_state, style=style, language=language, extra=extra),
            method="explain",
            max_tokens=max_tokens or settings.yandex_max_tokens,
        )

    def evaluate_answer(self, *, question: dict[str, Any], answer: str, context: str = "", learner_state: str = "", max_tokens: int | None = None) -> AIResult:
        return self._ask(
            system=prompts.EVALUATOR_SYSTEM,
            user=prompts.evaluate_prompt(question=question, answer=answer, context=context, learner_state=learner_state),
            method="evaluate_answer",
            json_mode=True,
            max_tokens=max_tokens or 900,
            temperature=0.1,
        )

    def generate_question(self, *, spec: dict[str, Any], context: str = "", avoid: list[str] | None = None, max_tokens: int | None = None) -> AIResult:
        return self._ask(
            system=prompts.QUESTION_AUTHOR_SYSTEM,
            user=prompts.question_prompt(spec=spec, context=context, avoid=avoid or []),
            method="generate_question",
            json_mode=True,
            max_tokens=max_tokens or 1100,
            temperature=0.7,
        )

    def generate_practice(self, *, topic: str, level: str, context: str = "", learner_state: str = "", kind: str = "mixed", max_tokens: int | None = None) -> AIResult:
        return self._ask(
            system=prompts.PRACTICE_SYSTEM,
            user=prompts.practice_prompt(topic=topic, level=level, context=context, learner_state=learner_state, kind=kind),
            method="generate_practice",
            json_mode=True,
            max_tokens=max_tokens or 1400,
            temperature=0.6,
        )

    def generate_project(self, *, learner_state: str, context: str = "", level: str = "middle", brief: str = "", max_tokens: int | None = None) -> AIResult:
        return self._ask(
            system=prompts.PROJECT_SYSTEM,
            user=prompts.project_prompt(learner_state=learner_state, context=context, level=level, brief=brief),
            method="generate_project",
            json_mode=True,
            max_tokens=max_tokens or 1500,
            temperature=0.6,
        )

    def interview(self, *, transcript: list[dict[str, str]], stage: str, level: str, context: str = "", candidate_answer: str = "", max_tokens: int | None = None) -> AIResult:
        return self._ask(
            system=prompts.INTERVIEWER_SYSTEM if stage != "report" else prompts.INTERVIEW_REPORT_SYSTEM,
            user=prompts.interview_prompt(transcript=transcript, stage=stage, level=level, context=context, candidate_answer=candidate_answer),
            method="interview",
            json_mode=stage == "report",
            max_tokens=max_tokens or (1300 if stage == "report" else 700),
            temperature=0.5,
        )

    def summarize(self, text: str, *, instruction: str = "", max_tokens: int = 700) -> AIResult:
        return self._ask(
            system=prompts.SUMMARIZER_SYSTEM,
            user=prompts.summarize_prompt(text=text, instruction=instruction),
            method="summarize",
            max_tokens=max_tokens,
            temperature=0.2,
        )

    def plan_learning(self, *, learner_state: str, candidates: list[dict[str, Any]], due_reviews: list[dict[str, Any]], constraints: str = "", max_tokens: int | None = None) -> AIResult:
        return self._ask(
            system=prompts.PLANNER_SYSTEM,
            user=prompts.plan_prompt(learner_state=learner_state, candidates=candidates, due_reviews=due_reviews, constraints=constraints),
            method="plan_learning",
            json_mode=True,
            max_tokens=max_tokens or 1200,
            temperature=0.3,
        )

    def review_code(self, *, code: str, task: dict[str, Any], context: str = "", max_tokens: int | None = None) -> AIResult:
        return self._ask(
            system=prompts.CODE_REVIEW_SYSTEM,
            user=prompts.code_review_prompt(code=code, task=task, context=context),
            method="code_review",
            json_mode=True,
            max_tokens=max_tokens or 1000,
            temperature=0.2,
        )

    def mentor(self, *, question: str, project: dict[str, Any], progress: str, context: str = "", max_tokens: int | None = None) -> AIResult:
        return self._ask(
            system=prompts.MENTOR_SYSTEM,
            user=prompts.mentor_prompt(question=question, project=project, progress=progress, context=context),
            method="mentor",
            max_tokens=max_tokens or 1100,
        )

    # ------------------------------------------------------------- utilities
    @staticmethod
    def clamp_max_tokens(max_tokens: int | None) -> int:  # noqa: D401 - provider-level cap
        limit = settings.max_tokens_per_request
        if not max_tokens or max_tokens <= 0:
            return min(900, limit)
        return min(int(max_tokens), limit)

    @staticmethod
    def estimate_usage(messages: list[dict[str, str]], text: str) -> Usage:
        prompt_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        return Usage(input_tokens=prompt_tokens, output_tokens=estimate_tokens(text))


class DisabledProvider(AIProvider):
    """No credentials configured: AI features short-circuit with a clear message."""

    name = "none"

    def complete(self, messages: list[dict[str, str]], **_: Any) -> AIResult:
        return AIResult(
            ok=False,
            error_code="disabled",
            error="No AI provider configured",
            provider=self.name,
            model="none",
        )

    def health(self) -> dict[str, Any]:
        return {"provider": "none", "configured": False, "model": "none"}


def health_of(provider: AIProvider | None) -> dict[str, Any]:
    if provider is None:
        return {"provider": "none", "configured": False, "model": "none"}
    try:
        return provider.health()
    except Exception as exc:  # noqa: BLE001
        return {"provider": getattr(provider, "name", "?"), "configured": False, "model": "", "error": str(exc)[:200]}

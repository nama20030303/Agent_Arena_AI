"""
AI orchestration: provider selection, hard budget limits, caching and accounting.

Every AI call in the product goes through :class:`AIManager`, which means
  * cost control (§35) is enforced in exactly one place,
  * identical requests are answered from cache (§36) - explanations, questions,
    summaries and embeddings are expensive and highly repetitive,
  * we always know *which* provider answered, with how many tokens, so the UI can
    show real usage instead of a made-up number.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai.base import AIProvider, AIResult, DisabledProvider, Usage, health_of
from app.ai.openai_compat import OllamaProvider, OpenAICompatProvider
from app.ai.yandex import YandexGPTProvider
from app.config import settings
from app.db import get_sessionmaker, session_scope
from app.models import AICacheEntry, AIRequest, AIUsage

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v3"
CACHEABLE = {
    "explain",
    "generate",
    "summarize",
    "generate_question",
    "generate_practice",
    "generate_project",
    "plan_learning",
    "review_code",
}


def build_provider() -> AIProvider:
    """Pick a provider from configuration only - business logic must not know this."""
    choice = settings.ai_provider
    if choice in {"", "none", "off", "disabled"}:
        return DisabledProvider()
    if choice in {"auto", "yandex"}:
        if settings.has_yandex_credentials():
            return YandexGPTProvider()
        if choice == "yandex":
            return DisabledProvider()
    if choice in {"auto", "openai"} and settings.openai_api_key:
        return OpenAICompatProvider()
    if choice in {"auto", "ollama"} and choice == "ollama":
        return OllamaProvider()
    if choice == "openai-compatible":
        return OpenAICompatProvider()
    if choice in {"auto"} and settings.ollama_base_url:
        probe = OllamaProvider()
        if probe.health().get("configured"):
            return probe
    return DisabledProvider()


class AIManager:
    def __init__(self) -> None:
        self._provider: AIProvider | None = None

    # ------------------------------------------------------------- provider io
    @property
    def provider(self) -> AIProvider:
        if self._provider is None:
            self._provider = build_provider()
        return self._provider

    def reset(self) -> None:
        self._provider = None

    def health(self) -> dict[str, Any]:
        return health_of(self._provider if self._provider is not None else self.provider)

    def is_configured(self) -> bool:
        """Duck-typed: a provider says whether it is usable (keys present / server reachable)."""
        try:
            return bool(self.provider.health().get("configured"))
        except Exception:  # noqa: BLE001
            return False

    # ----------------------------------------------------------------- budget
    def usage_for(self, session: Session, user_id: int | None, *, day: date | None = None) -> dict[str, Any]:
        day = day or date.today()
        query = select(func.coalesce(func.sum(AIUsage.requests), 0), func.coalesce(func.sum(AIUsage.prompt_tokens + AIUsage.completion_tokens), 0),
                       func.coalesce(func.sum(AIUsage.estimated_cost), 0.0), func.coalesce(func.sum(AIUsage.cached_hits), 0),
                       func.coalesce(func.sum(AIUsage.errors), 0), func.coalesce(func.sum(AIUsage.limited), 0)).where(
            AIUsage.day == day
        )
        if user_id is not None:
            query = query.where(AIUsage.user_id == user_id)
        row = session.execute(query).one()
        return {
            "day": day.isoformat(),
            "requests": int(row[0]),
            "tokens": int(row[1]),
            "cost": float(row[2]),
            "cached_hits": int(row[3]),
            "errors": int(row[4]),
            "limited": int(row[5]),
        }

    def month_usage(self, session: Session, user_id: int | None) -> dict[str, Any]:
        start = date.today().replace(day=1)
        query = select(
            func.coalesce(func.sum(AIUsage.requests), 0),
            func.coalesce(func.sum(AIUsage.prompt_tokens + AIUsage.completion_tokens), 0),
            func.coalesce(func.sum(AIUsage.estimated_cost), 0.0),
            func.coalesce(func.sum(AIUsage.cached_hits), 0),
        ).where(AIUsage.day >= start)
        if user_id is not None:
            query = query.where(AIUsage.user_id == user_id)
        row = session.execute(query).one()
        return {"requests": int(row[0]), "tokens": int(row[1]), "cost": float(row[2]), "cached_hits": int(row[3]),
                "since": start.isoformat(), "budget": settings.monthly_budget, "currency": settings.cost_currency}

    def limits_for(self, user_id: int | None, session: Session | None = None) -> dict[str, int]:
        """Env limits, optionally overridden per profile (Settings page -> user.settings_json)."""
        limits = {
            "max_requests_per_day": settings.max_ai_requests_per_day,
            "max_tokens_per_day": settings.max_ai_tokens_per_day,
            "max_tokens_per_request": settings.max_tokens_per_request,
        }
        if not user_id:
            return limits
        from app.models import User

        owns = session is None
        db = session or get_sessionmaker()()
        try:
            blob = db.execute(select(User.settings_json).where(User.id == user_id)).scalar_one_or_none() or {}
        finally:
            if owns:
                db.close()
        try:
            if blob.get("max_ai_requests_per_day") is not None:
                limits["max_requests_per_day"] = max(0, min(2000, int(blob["max_ai_requests_per_day"])))
            if blob.get("max_ai_tokens_per_day") is not None:
                limits["max_tokens_per_day"] = max(0, min(5_000_000, int(blob["max_ai_tokens_per_day"])))
            if blob.get("max_tokens_per_request") is not None:
                limits["max_tokens_per_request"] = max(64, min(8000, int(blob["max_tokens_per_request"])))
        except (TypeError, ValueError):
            pass
        return limits

    def budget_state(self, user_id: int | None, session: Session | None = None) -> dict[str, Any]:
        owns = session is None
        db = session or get_sessionmaker()()
        try:
            today = self.usage_for(db, user_id)
            month = self.month_usage(db, user_id)
            limits = self.limits_for(user_id, db)
        finally:
            if owns:
                db.close()
        requests_left = max(0, limits["max_requests_per_day"] - today["requests"])
        tokens_left = max(0, limits["max_tokens_per_day"] - today["tokens"])
        return {
            "requests_today": today["requests"],
            "tokens_today": today["tokens"],
            "cost_today": round(today["cost"], 6),
            "requests_this_month": month["requests"],
            "tokens_this_month": month["tokens"],
            "cost_this_month": round(month["cost"], 6),
            "max_requests_per_day": limits["max_requests_per_day"],
            "max_tokens_per_day": limits["max_tokens_per_day"],
            "max_tokens_per_request": limits["max_tokens_per_request"],
            "requests_remaining_today": requests_left,
            "tokens_remaining_today": tokens_left,
            "currency": settings.cost_currency,
            "monthly_budget": settings.monthly_budget,
            "budget_exhausted": bool(settings.monthly_budget and month["cost"] >= settings.monthly_budget),
            "limit_reached": requests_left <= 0 or tokens_left <= 0,
            "cache_saved_requests": today["cached_hits"],
        }


    # --------------------------------------------------------------- main call
    def invoke(
        self,
        method: str,
        *,
        user_id: int | None = None,
        kwargs: dict[str, Any] | None = None,
        allow_ai: bool = True,
        use_cache: bool | None = None,
        session: Session | None = None,
    ) -> AIResult:
        kwargs = kwargs or {}
        provider = self.provider
        owned = session is None
        db = session if session is not None else get_sessionmaker()()
        try:
            if not allow_ai or not self.is_configured():
                return AIResult(ok=False, error_code="disabled", error="AI provider not configured", provider=provider.name)

            budget = self.budget_state(user_id, db)
            if budget["limit_reached"] or budget["budget_exhausted"]:
                reason = "monthly budget reached" if budget["budget_exhausted"] else "daily AI limit reached"
                self._log(db, user_id=user_id, provider=provider.name, method=method, model=provider.model_label(),
                          status="limited", error=reason, kwargs=kwargs, result=AIResult(ok=False, error_code="limited", error=reason))
                return AIResult(ok=False, error_code="limited", error=reason, provider=provider.name, model=provider.model_label())

            cacheable = settings.ai_cache_enabled and (use_cache if use_cache is not None else method in CACHEABLE)
            key = self.cache_key(method, provider.model_label(), kwargs)
            if cacheable:
                hit = db.execute(select(AICacheEntry).where(AICacheEntry.cache_key == key)).scalar_one_or_none()
                if hit and (not hit.expires_at or hit.expires_at > datetime.utcnow()):
                    hit.hits += 1
                    hit.last_hit_at = datetime.utcnow()
                    self._log(
                        db,
                        user_id=user_id,
                        provider=provider.name,
                        method=method,
                        model=hit.model or provider.model_label(),
                        status="cached",
                        kwargs=kwargs,
                        result=AIResult(ok=True, text=hit.response_text, data=hit.response_json),
                        tokens=hit.tokens,
                    )
                    db.commit()
                    return AIResult(
                        ok=True,
                        text=hit.response_text,
                        data=hit.response_json or None,
                        provider=provider.name,
                        model=hit.model or provider.model_label(),
                        usage=Usage(input_tokens=hit.tokens // 2, output_tokens=hit.tokens // 2, cached=True),
                        meta={"cache": True, "hits": hit.hits},
                    )

            if not hasattr(provider, method):
                return AIResult(ok=False, error_code="parse", error=f"provider has no method {method}", provider=provider.name)
            # hard per-request token cap (cost control), applied to every provider
            kwargs = {**kwargs, "max_tokens": min(int(kwargs.get("max_tokens") or 0) or provider.clamp_max_tokens(None), budget["max_tokens_per_request"])}
            result: AIResult = getattr(provider, method)(**kwargs)
            result.provider = provider.name

            if result.ok and cacheable:
                db.add(
                    AICacheEntry(
                        cache_key=key,
                        method=method,
                        model=provider.model_label(),
                        request_json=json.dumps(_safe(kwargs), ensure_ascii=False)[:20000],
                        response_text=result.text[:60000],
                        response_json=result.data if isinstance(result.data, dict) else {},
                        tokens=result.usage.total_tokens,
                        expires_at=datetime.utcnow() + timedelta(days=settings.ai_cache_ttl_days),
                    )
                )
            self._log(db, user_id=user_id, provider=provider.name, method=method, model=provider.model_label(),
                      status="ok" if result.ok else (result.error_code or "error"), error=result.error,
                      kwargs=kwargs, result=result)
            db.commit()
            return result
        finally:
            if owned:
                db.close()

    # ---------------------------------------------------------------- logging
    def _log(
        self,
        session: Session,
        *,
        user_id: int | None,
        provider: str,
        method: str,
        model: str,
        status: str,
        kwargs: dict[str, Any],
        result: AIResult,
        error: str = "",
        tokens: int | None = None,
    ) -> None:
        prompt_chars = sum(len(json.dumps(_safe(v), ensure_ascii=False)) for v in kwargs.values()) if kwargs else 0
        in_tokens = result.usage.input_tokens or (prompt_chars // 4)
        out_tokens = result.usage.output_tokens
        usage_row = session.execute(
            select(AIUsage).where(AIUsage.user_id == (user_id or 0), AIUsage.day == date.today())
        ).scalar_one_or_none()
        if usage_row is None:
            usage_row = AIUsage(
                user_id=user_id or 0,
                day=date.today(),
                requests=0,
                cached_hits=0,
                errors=0,
                limited=0,
                prompt_tokens=0,
                completion_tokens=0,
                estimated_cost=0.0,
            )
            session.add(usage_row)
            session.flush()
        if status == "cached":
            usage_row.cached_hits += 1
        elif status == "limited":
            usage_row.limited += 1
        else:
            usage_row.requests += 1
            usage_row.prompt_tokens += in_tokens
            usage_row.completion_tokens += out_tokens
            usage_row.estimated_cost += result.usage.estimated_cost
        if status not in {"ok", "cached"}:
            usage_row.errors += 1
        session.add(
            AIRequest(
                user_id=user_id,
                provider=provider,
                method=method,
                model=model,
                prompt_hash=hashlib.sha256(json.dumps(_safe(kwargs), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:32],
                prompt_chars=prompt_chars,
                completion_chars=len(result.text or ""),
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                total_tokens=(tokens if tokens is not None else in_tokens + out_tokens),
                estimated_cost=result.usage.estimated_cost,
                latency_ms=result.usage.latency_ms,
                status=status,
                error=(error or result.error)[:1000],
                cached=status == "cached",
                meta={"user_id": user_id},
            )
        )
        session.flush()

    @staticmethod
    def cache_key(method: str, model: str, kwargs: dict[str, Any]) -> str:
        blob = json.dumps({"m": method, "model": model, "kw": _safe(kwargs), "v": PROMPT_VERSION}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------- statistics
    def stats(self, user_id: int | None, *, window_days: int = 30) -> dict[str, Any]:
        with session_scope() as session:  # noqa: SIM117
            since = datetime.utcnow() - timedelta(days=window_days)
            per_method = session.execute(
                select(AIRequest.method, func.count(), func.avg(AIRequest.total_tokens), func.sum(AIRequest.estimated_cost))
                .where(AIRequest.created_at >= since)
                .group_by(AIRequest.method)
            ).all()
            by_day = session.execute(
                select(AIUsage.day, AIUsage.requests, AIUsage.cached_hits, AIUsage.estimated_cost)
                .where(AIUsage.day >= since.date())
                .order_by(AIUsage.day.asc())
            ).all()
            errors = session.execute(
                select(AIRequest.created_at, AIRequest.method, AIRequest.status, AIRequest.error)
                .where(AIRequest.status.notin_(["ok", "cached"]))
                .order_by(AIRequest.id.desc())
                .limit(8)
            ).all()
            cache_rows = session.execute(select(func.coalesce(func.sum(AICacheEntry.hits), 0), func.count(AICacheEntry.id))).one()
            cached, entries = int(cache_rows[0] or 0), int(cache_rows[1] or 0)
        return {
            "budget": self.budget_state(user_id, session),
            "by_method": [
                {"method": m, "calls": int(n), "avg_tokens": round(float(avg or 0), 1), "cost": round(float(cost or 0), 6)}
                for m, n, avg, cost in per_method
            ],
            "by_day": [
                {"date": d.isoformat(), "requests": int(r), "cache_hits": int(c), "cost": round(float(cost), 6)}
                for d, r, c, cost in by_day
            ],
            "recent_problems": [
                {"at": ts.isoformat(), "method": m, "status": s, "error": (err or "")[:220]} for ts, m, s, err in errors
            ],
            "cache": {"entries": entries, "total_hits": cached},
            "window_days": window_days,
        }


def _safe(value: Any) -> Any:
    """Make anything JSON-dumpable (used for cache keys and logging)."""
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)[:2000]


manager = AIManager()


def get_ai_manager() -> AIManager:
    return manager


def as_public_dict(result: AIResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": result.ok,
        "text": result.text,
        "data": result.data,
        "provider": result.provider,
        "model": result.model,
        "cached": result.usage.cached,
        "error": result.error,
        "error_code": result.error_code,
        "message": result.user_message(),
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
            "estimated_cost": round(result.usage.estimated_cost, 6),
            "currency": settings.cost_currency,
            "latency_ms": result.usage.latency_ms,
        },
    }
    return payload

"""Dashboard, progress, notes, journal, memory and AI cost control (§26, §27, §32-§37)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.api.deps import db_dep, get_current_user, get_optional_user, user_payload
from app.config import settings
from app.db import session_scope
from app.knowledge.vectorstore import get_vector_store
from app.models import AICacheEntry, Document, StudySession, Topic, User
from app.seed import seed_summary
from app.services import learning_engine, notes as notes_service, progress
from app.services.projects import active_enrollment
from app.services.srs import due_summary

router = APIRouter(tags=["progress & system"])


@router.get("/health")
def health(session: Session = Depends(db_dep)) -> dict[str, Any]:
    """Liveness + what is available. Deliberately requires no auth."""
    from app.ai.manager import manager as ai_manager

    with session_scope() as scoped:
        seed = seed_summary(scoped)
    return {
        "status": "ok",
        "version": settings.app_version,
        "ai": {
            "provider": settings.selected_provider_name(),
            "configured": ai_manager.is_configured(),
            "model": ai_manager.health().get("model", ""),
            "limits": {
                "max_requests_per_day": settings.max_ai_requests_per_day,
                "max_tokens_per_request": settings.max_tokens_per_request,
                "cost_currency": settings.cost_currency,
            },
        },
        "vector_store": get_vector_store().health(),
        "embedding_provider": settings.embedding_provider,
        "seed": seed,
        "offline_first": True,
    }


@router.get("/dashboard")
def dashboard(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    summary = progress.progress_summary(session, user)
    plan = learning_engine.get_or_build_plan(session, user)
    plan_payload = {
        "date": plan.plan_date.isoformat(),
        "items": plan.items or [],
        "completed": sum(1 for i in (plan.items or []) if i.get("done")),
        "total": len(plan.items or []),
        "minutes_budget": plan.minutes_budget,
        "rationale": plan.rationale,
    }
    session.commit()
    state = learning_engine.learner_state(session, user)
    rec = state.get("current_topic")
    topic_name = rec
    topic_obj = session.execute(select(Topic).where(Topic.code == (rec or ""))).scalar_one_or_none()
    if topic_obj is not None:
        topic_name = topic_obj.name
    enrollment = active_enrollment(session, user)
    due = due_summary(session, user.id)
    streak = progress.streak(session, user.id)
    docs = int(session.execute(select(func.count(Document.id)).where(Document.status == Document.STATUS_INDEXED)).scalar_one() or 0)
    from app.ai.manager import manager as ai_manager

    return {
        "ml_engineer_score": summary["ml_engineer_score"],
        "dimensions": summary["dimensions"],
        "current_level": {"index": user.level_index, "label": user.level_label},
        "today": plan_payload,
        "weak_skills": summary["weak_skills"][:6],
        "strong_skills": summary["strong_skills"][:6],
        "due_for_review": {"total": due["due_total"], "items": due["urgent"][:6]},
        "current_course": {"topic": rec, "name": topic_name, "readiness": learning_engine.recommend_next(session, user).get("topic", {}).get("readiness")},
        "current_project": enrollment,
        "streak": streak,
        "velocity": progress.velocity(session, user, days=14),
        "library": {"indexed_documents": docs},
        "ai": {
            "configured": ai_manager.is_configured(),
            "provider": settings.selected_provider_name(),
            **ai_manager.budget_state(user.id),
        },
        "recommendation": learning_engine.recommend_next(session, user),
        "diagnostic_done": user.diagnostic_done,
    }


@router.get("/progress")
def get_progress(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    return progress.progress_summary(session, user)


@router.get("/progress/roadmap")
def get_roadmap(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> list[dict[str, Any]]:
    return progress.roadmap(session, user)


@router.get("/velocity")
def get_velocity(days: int = 28, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    return progress.velocity(session, user, days=max(7, min(180, days)))


# --------------------------------------------------------------------------- #
#  Notes / journal / memory
# --------------------------------------------------------------------------- #
class NoteIn(BaseModel):
    kind: str = "note"
    title: str = ""
    body: str = Field(default="", max_length=20000)
    topic_code: str = ""
    chunk_id: int | None = None
    document_id: int | None = None
    tags: list[str] = []
    excerpt: str = ""


class NotePatch(BaseModel):
    title: str | None = None
    body: str | None = None
    tags: list[str] | None = None
    pinned: bool | None = None
    kind: str | None = None


@router.get("/notes")
def list_notes(kind: str = "", topic: str = "", q: str = "", limit: int = 100, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> list[dict[str, Any]]:
    return notes_service.list_notes(session, user, kind=kind, topic=topic, q=q, limit=limit)


@router.post("/notes", status_code=201)
def create_note(payload: NoteIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    if not payload.body.strip() and not payload.excerpt.strip():
        raise HTTPException(status_code=400, detail="note body is empty")
    return notes_service.create_note(
        session,
        user,
        kind=payload.kind,
        title=payload.title,
        body=payload.body,
        topic_code=payload.topic_code,
        chunk_id=payload.chunk_id,
        document_id=payload.document_id,
        tags=payload.tags,
        excerpt=payload.excerpt,
    )


@router.post("/notes/from-mistake/{attempt_id}")
def note_from_mistake(attempt_id: int, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    payload = notes_service.note_from_mistake(session, user, attempt_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="attempt not found")
    return payload


@router.patch("/notes/{note_id}")
def patch_note(note_id: int, payload: NotePatch, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    result = notes_service.update_note(session, user, note_id, **payload.model_dump(exclude_none=True))
    if result is None:
        raise HTTPException(status_code=404, detail="note not found")
    return result


@router.delete("/notes/{note_id}")
def delete_note(note_id: int, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    if not notes_service.delete_note(session, user, note_id):
        raise HTTPException(status_code=404, detail="note not found")
    return {"deleted": note_id}


class JournalIn(BaseModel):
    content: str = Field(min_length=4, max_length=20000)
    mood: int = 3
    minutes: int = 0
    date: str = ""


@router.get("/journal")
def list_journal(limit: int = 30, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> list[dict[str, Any]]:
    return notes_service.list_journal(session, user, limit=limit)


@router.post("/journal", status_code=201)
def write_journal(payload: JournalIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    row = notes_service.write_journal(
        session, user, content=payload.content, mood=payload.mood, minutes=payload.minutes, entry_date=payload.date or None
    )
    if payload.minutes:
        today = session.execute(
            select(StudySession).where(StudySession.user_id == user.id, StudySession.activity == "learn").order_by(StudySession.id.desc()).limit(1)
        ).scalars().first()
        if today is not None and today.ended_at is None:
            today.meta = {**(today.meta or {}), "journal_minutes": payload.minutes}
    session.commit()
    return row


@router.delete("/journal/{entry_id}")
def delete_journal(entry_id: int, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    if not notes_service.delete_journal(session, user, entry_id):
        raise HTTPException(status_code=404, detail="entry not found")
    return {"deleted": entry_id}


@router.get("/memory")
def memory(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    return progress.memory_bundle(session, user.id)


@router.post("/memory/consolidate")
def consolidate(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    return notes_service.consolidate_memory(session, user)


# --------------------------------------------------------------------------- #
#  AI usage / cost control / cache
# --------------------------------------------------------------------------- #
@router.get("/ai/usage")
def ai_usage(days: int = 30, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    from app.ai.manager import manager as ai_manager

    return ai_manager.stats(user.id, window_days=max(1, min(180, days)))


@router.get("/ai/config")
def ai_config(session: Session = Depends(db_dep), user: User | None = Depends(get_optional_user)) -> dict[str, Any]:
    """Public AI status. Never contains API keys - only presence flags, model names and counts."""
    from app.ai.manager import manager as ai_manager

    health = ai_manager.health()
    return {
        **settings.to_public_dict(),
        "health": health,
        "budget": ai_manager.budget_state(user.id if user else None),
        "vector_store": get_vector_store().health(),
        "seed": seed_summary(session),
        "sandbox": {
            "enabled": settings.code_execution_enabled,
            "timeout_seconds": settings.code_execution_timeout_seconds,
            "memory_mb": settings.code_execution_memory_mb,
        },
    }


@router.post("/ai/purge-cache")
def purge_cache(max_age_days: int | None = None, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    before = datetime.utcnow() - timedelta(days=max_age_days) if max_age_days else None
    stmt = select(AICacheEntry)
    if before:
        stmt = stmt.where(AICacheEntry.created_at < before)
    rows = list(session.execute(stmt).scalars())
    for row in rows:
        session.delete(row)
    session.commit()
    return {"removed": len(rows), "mode": "aged" if before else "all"}


@router.post("/ai/limits")
def set_limits(max_requests_per_day: int | None = None, max_tokens_per_request: int | None = None, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    """Per-profile budget overrides (persisted in settings; enforced by the AI manager)."""
    blob = dict(user.settings_json or {})
    if max_requests_per_day is not None:
        blob["max_ai_requests_per_day"] = max(0, min(2000, int(max_requests_per_day)))
    if max_tokens_per_request is not None:
        blob["max_tokens_per_request"] = max(64, min(8000, int(max_tokens_per_request)))
    user.settings_json = blob
    session.commit()
    return {"settings": blob, "effective": settings.to_public_dict()}


@router.get("/system/seed")
def seed_status(session: Session = Depends(db_dep)) -> dict[str, Any]:
    return seed_summary(session)

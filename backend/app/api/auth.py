"""Profile, session and diagnostic routes (§13, §49)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import db_dep, get_current_user, issue_token, user_payload
from app.models import DiagnosticSession, LearningMemory, User
from app.services import learning_engine
from app.services.graph import SkillGraph
from app.services.user_knowledge import ensure_skill_rows

router = APIRouter(tags=["auth"])


class RegisterIn(BaseModel):
    username: str = Field(min_length=2, max_length=40)
    display_name: str = ""
    background: str = ""
    target_role: str = "Senior ML Engineer"
    goal: str = ""
    daily_minutes: int = 60
    reply_language: str = "auto"


class ProfilePatch(BaseModel):
    display_name: str | None = None
    background: str | None = None
    target_role: str | None = None
    goal: str | None = None
    daily_minutes: int | None = None
    level_index: int | None = None
    reply_language: str | None = None
    web_search_allowed: bool | None = None
    ai_enabled: bool | None = None
    settings: dict[str, Any] | None = None


@router.post("/auth/register")
def register(payload: RegisterIn, session: Session = Depends(db_dep)) -> dict[str, Any]:
    username = payload.username.strip().lower()
    if session.execute(select(User).where(User.username == username)).scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail="That username already exists. Sign in instead.")
    user = User(
        username=username,
        display_name=payload.display_name.strip()[:120] or username,
        background=payload.background.strip()[:200],
        target_role=payload.target_role.strip()[:60] or "Senior ML Engineer",
        goal=payload.goal.strip()[:2000],
        daily_minutes=max(10, min(600, int(payload.daily_minutes or 60))),
        reply_language=payload.reply_language if payload.reply_language in {"auto", "en", "ru"} else "auto",
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    ensure_skill_rows(session, user.id)
    session.commit()
    return {"token": issue_token(session, user), "user": user_payload(session, user)}


@router.post("/auth/login")
def login(payload: dict[str, Any], session: Session = Depends(db_dep)) -> dict[str, Any]:
    username = str(payload.get("username") or "").strip().lower()
    user = session.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="No such profile. Create one first.")
    return {"token": issue_token(session, user), "user": user_payload(session, user)}


@router.get("/auth/me")
def me(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    state = learning_engine.learner_state(session, user)
    return {
        "user": user_payload(session, user),
        "level_index": user.level_index,
        "level_label": user.level_label,
        "stats": {
            "weak_skills": state["weaknesses"],
            "strong_skills": state["strengths"],
            "goal": state["goal"],
            "velocity": state["velocity"],
        },
    }


@router.patch("/auth/me")
def patch_me(payload: ProfilePatch, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    data = payload.model_dump(exclude_none=True)
    settings_blob = data.pop("settings", None)
    for field, value in data.items():
        if field == "daily_minutes" and value is not None:
            value = max(10, min(600, int(value)))
        if field == "level_index" and value is not None:
            from app.seed.curriculum import LEVELS

            value = max(0, min(10, int(value)))
            user.level_label = LEVELS[value]["title"]
        setattr(user, field, value)
    if settings_blob:
        user.settings_json = {**(user.settings_json or {}), **settings_blob}
    session.commit()
    return user_payload(session, user)


@router.get("/diagnostic")
def diagnostic_state(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    record = session.execute(
        select(DiagnosticSession).where(DiagnosticSession.user_id == user.id).order_by(DiagnosticSession.id.desc()).limit(1)
    ).scalar_one_or_none()
    pool = learning_engine.diagnostic_question_pool(session, count=28)
    return {
        "done": bool(user.diagnostic_done),
        "available_questions": len(pool),
        "state": learning_engine.diagnostic_payload(record) if record else None,
    }


@router.post("/diagnostic/start")
def diagnostic_start(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    try:
        payload = learning_engine.start_diagnostic(session, user, count=28)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session.commit()
    return payload


@router.post("/diagnostic/submit")
def diagnostic_submit(
    answers: dict[str, Any],
    user: User = Depends(get_current_user),
    session: Session = Depends(db_dep),
) -> dict[str, Any]:
    try:
        result = learning_engine.submit_diagnostic(session, user, answers.get("answers") or answers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    graph = SkillGraph(session)
    return {**result, "skill_map": graph.graph_payload(user.id)}


@router.post("/diagnostic/skip")
def diagnostic_skip(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    """A learner can skip the test - we then treat everything as unknown instead of pretending."""
    user.diagnostic_done = True
    user.onboarded = True
    ensure_skill_rows(session, user.id)
    session.add(
        LearningMemory(
            user_id=user.id,
            category="preference",
            key="diagnostic",
            value="skipped the diagnostic; starting from zero evidence",
            payload={"skipped": True},
        )
    )
    learning_engine.build_daily_plan(session, user, reason="engine")
    session.commit()
    return {"ok": True, "recommendation": learning_engine.recommend_next(session, user)}


@router.get("/diagnostic/result")
def diagnostic_result(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    record = session.execute(
        select(DiagnosticSession).where(DiagnosticSession.user_id == user.id, DiagnosticSession.status == "completed")
        .order_by(DiagnosticSession.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    if record is None:
        return {"result": None}
    return {"result": record.result or {}, "completed_at": record.completed_at.isoformat() if record.completed_at else None}

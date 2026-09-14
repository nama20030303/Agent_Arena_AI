"""
Auth + shared dependencies.

MVP auth: register a profile, receive a bearer token, send it on every request.
A single-user local install can also use the auto-created default profile, so opening
the app in a browser works immediately (configurable via ALLOW_ANONYMOUS_DEFAULT_USER).
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import User, UserSession


def db_dep(session: Session = Depends(get_db)) -> Session:
    return session


def _resolve_token(session: Session, token: str) -> User | None:
    row = session.execute(select(UserSession).where(UserSession.token == token)).scalar_one_or_none()
    if row is None:
        return None
    user = session.get(User, row.user_id)
    if user is None:
        return None
    row.last_used_at = datetime.utcnow()
    session.commit()
    return user


def get_current_user(
    request: Request,
    authorization: str | None = Header(default=None),
    session: Session = Depends(get_db),
) -> User:
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not token:
        token = (request.query_params.get("token") or "").strip()
    if token:
        user = _resolve_token(session, token)
        if user is not None:
            return user
    user = default_user(session)
    if user is not None:
        return user
    raise HTTPException(status_code=401, detail="Not authenticated. Create a profile at POST /api/auth/register.")


def get_optional_user(request: Request, authorization: str | None = Header(default=None), session: Session = Depends(get_db)) -> User | None:
    try:
        return get_current_user(request, authorization, session)
    except HTTPException:
        return None


def default_user(session: Session) -> User | None:
    if not getattr(settings, "allow_anonymous_default_user", True):
        return None
    return session.execute(select(User).order_by(User.id.asc()).limit(1)).scalar_one_or_none()


def issue_token(session: Session, user: User) -> str:
    token = hashlib.sha256((secrets.token_hex(16) + str(user.id) + datetime.utcnow().isoformat()).encode()).hexdigest()[:48]
    session.add(UserSession(user_id=user.id, token=token))
    session.commit()
    return token


def user_payload(session: Session, user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "target_role": user.target_role,
        "background": user.background,
        "goal": user.goal,
        "daily_minutes": user.daily_minutes,
        "level_index": user.level_index,
        "level_label": user.level_label,
        "onboarded": user.onboarded,
        "diagnostic_done": user.diagnostic_done,
        "web_search_allowed": user.web_search_allowed,
        "ai_enabled": user.ai_enabled,
        "reply_language": user.reply_language,
        "settings": user.settings_json or {},
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }

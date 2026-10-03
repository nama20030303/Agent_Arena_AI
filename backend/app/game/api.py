# -*- coding: utf-8 -*-
"""Game REST API: /api/game/* — the mobile/web client of the Knowledge Universe."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import db_dep, get_current_user
from app.game import engine, teacher as teacher_svc
from app.game.models import (
    GameAchievement,
    GameChatMessage,
    GameEdge,
    GameEvent,
    GameItem,
    GameNode,
    GameQuestion,
    GameWorld,
    PlayerAchievement,
    PlayerChest,
    PlayerItem,
    PlayerNode,
    PlayerQuestionState,
    PlayerState,
)
from app.models import User

router = APIRouter(prefix="/game", tags=["game"])


def _player(session: Session, user: User) -> PlayerState:
    return engine.ensure_player(session, user.id)


# --------------------------------------------------------------------------- #
#  Onboarding & state
# --------------------------------------------------------------------------- #
class OnboardIn(BaseModel):
    interests: list[str] = Field(default_factory=list)
    start_level: str = "beginner"  # beginner|middle|advanced
    goal: str = ""
    intensity: str = "normal"  # light|normal|deep


@router.post("/onboard")
def onboard(payload: OnboardIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    ps = _player(session, user)
    valid_worlds = {w.code for w in session.execute(select(GameWorld)).scalars()}
    ps.interests = [w for w in payload.interests if w in valid_worlds]
    ps.start_level = payload.start_level if payload.start_level in {"beginner", "middle", "advanced"} else "beginner"
    ps.goal_ru = payload.goal.strip()[:300]
    ps.intensity = payload.intensity if payload.intensity in {"light", "normal", "deep"} else "normal"

    fx: dict[str, Any] = {}
    if not ps.onboarded:
        ps.onboarded = True
        engine.touch_streak(session, ps)
        engine._grant_xp(session, ps, 20, "onboarding", fx)
        engine._grant_chest(session, ps, source="onboarding", world_code=(ps.interests[0] if ps.interests else ""), fx=fx)
        engine._event(session, ps.user_id, "onboarding.completed", {"interests": ps.interests})
    engine.ensure_daily_quests(session, ps)
    session.commit()
    return {"state": engine.public_state(session, ps), "fx": fx}


@router.get("/state")
def state(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    ps = _player(session, user)
    quests = engine.ensure_daily_quests(session, ps)
    objective = engine.current_objective(session, ps)
    chests = list(
        session.execute(
            select(PlayerChest).where(PlayerChest.user_id == ps.user_id, PlayerChest.opened.is_(False)).order_by(PlayerChest.id)
        ).scalars()
    )
    session.commit()
    return {
        "state": engine.public_state(session, ps),
        "objective": objective,
        "quests": [_quest_payload(q) for q in quests],
        "chests": [{"id": c.id, "source": c.source, "world": c.world_code} for c in chests],
        "worlds": engine.world_powers(session, ps.user_id),
        "display_name": user.display_name or user.username,
        "username": user.username,
    }


def _quest_payload(q) -> dict[str, Any]:
    return {
        "code": q.code,
        "title": q.title_ru,
        "target": q.target,
        "progress": q.progress,
        "completed": q.completed,
        "xp": q.xp_reward,
        "chest": q.chest_on_complete,
    }


# --------------------------------------------------------------------------- #
#  Map & nodes
# --------------------------------------------------------------------------- #
@router.get("/map")
def knowledge_map(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    ps = _player(session, user)
    worlds = list(session.execute(select(GameWorld).order_by(GameWorld.order_index)).scalars())
    nodes = list(session.execute(select(GameNode).order_by(GameNode.order_index)).scalars())
    edges = list(session.execute(select(GameEdge)).scalars())
    pns = {pn.node_code: pn for pn in session.execute(select(PlayerNode).where(PlayerNode.user_id == ps.user_id)).scalars()}
    session.commit()
    return {
        "worlds": [
            {"code": w.code, "name": w.name_ru, "tagline": w.tagline_ru, "color": w.color, "glyph": w.glyph}
            for w in worlds
        ],
        "nodes": [
            {
                "code": n.code,
                "world": n.world_code,
                "name": n.name_ru,
                "short": n.short_ru,
                "difficulty": n.difficulty,
                "is_boss": n.is_boss,
                "x": n.pos_x,
                "y": n.pos_y,
                "status": (pns.get(n.code).status if pns.get(n.code) else "locked"),
                "progress": (pns.get(n.code).progress if pns.get(n.code) else 0.0),
            }
            for n in nodes
        ],
        "edges": [{"from": e.from_code, "to": e.to_code} for e in edges],
    }


@router.get("/node/{code}")
def node_detail(code: str, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    ps = _player(session, user)
    node = session.execute(select(GameNode).where(GameNode.code == code)).scalar_one_or_none()
    if node is None:
        raise HTTPException(404, "Узел знаний не найден.")
    pn = session.execute(
        select(PlayerNode).where(PlayerNode.user_id == ps.user_id, PlayerNode.node_code == code)
    ).scalar_one_or_none()
    status = pn.status if pn else "locked"
    world = session.execute(select(GameWorld).where(GameWorld.code == node.world_code)).scalar_one()

    questions: list[dict[str, Any]] = []
    if status != PlayerNode.ST_LOCKED:
        qstates = {
            qs.question_id: qs
            for qs in session.execute(
                select(PlayerQuestionState).where(PlayerQuestionState.user_id == ps.user_id)
            ).scalars()
        }
        for q in session.execute(select(GameQuestion).where(GameQuestion.node_code == code).order_by(GameQuestion.id)).scalars():
            qs = qstates.get(q.id)
            questions.append(
                {
                    "id": q.id,
                    "type": q.qtype,
                    "prompt": q.prompt_ru,
                    "options": q.options if q.qtype == "mcq" else [],
                    "difficulty": q.difficulty,
                    "hints_total": len(q.hints),
                    "hints_used": qs.hints_used if qs else 0,
                    "solved": bool(qs.solved) if qs else False,
                    "attempts": qs.attempts if qs else 0,
                }
            )
    session.commit()
    prereq_codes = [e.from_code for e in session.execute(select(GameEdge).where(GameEdge.to_code == code)).scalars()]
    prereqs = []
    for pc in prereq_codes:
        pnode = session.execute(select(GameNode).where(GameNode.code == pc)).scalar_one_or_none()
        if pnode:
            prereqs.append({"code": pc, "name": pnode.name_ru})
    return {
        "code": node.code,
        "name": node.name_ru,
        "short": node.short_ru,
        "theory": node.theory_ru if status != PlayerNode.ST_LOCKED else "",
        "difficulty": node.difficulty,
        "xp_reward": node.xp_reward,
        "is_boss": node.is_boss,
        "world": {"code": world.code, "name": world.name_ru, "color": world.color, "glyph": world.glyph},
        "status": status,
        "progress": pn.progress if pn else 0.0,
        "questions": questions,
        "prereqs": prereqs,
    }


# --------------------------------------------------------------------------- #
#  Practice
# --------------------------------------------------------------------------- #
class AnswerIn(BaseModel):
    question_id: int
    answer: str = ""
    option: int | None = None


@router.post("/answer")
def answer(payload: AnswerIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    ps = _player(session, user)
    qs = session.execute(
        select(PlayerQuestionState).where(
            PlayerQuestionState.user_id == ps.user_id, PlayerQuestionState.question_id == payload.question_id
        )
    ).scalar_one_or_none()
    hints_used = qs.hints_used if qs else 0
    try:
        result = engine.submit_answer(
            session, ps, payload.question_id, answer=payload.answer, option=payload.option, hints_used=hints_used
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    session.commit()
    return result


class HintIn(BaseModel):
    question_id: int


@router.post("/hint")
def hint(payload: HintIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    ps = _player(session, user)
    q = session.get(GameQuestion, payload.question_id)
    if q is None:
        raise HTTPException(404, "Задание не найдено.")
    qs = session.execute(
        select(PlayerQuestionState).where(
            PlayerQuestionState.user_id == ps.user_id, PlayerQuestionState.question_id == q.id
        )
    ).scalar_one_or_none()
    if qs is None:
        qs = PlayerQuestionState(user_id=ps.user_id, question_id=q.id)
        session.add(qs)
        session.flush()
    if not q.hints:
        raise HTTPException(400, "Для этого задания нет подсказок.")
    idx = min(qs.hints_used, len(q.hints) - 1)
    if qs.hints_used < len(q.hints):
        qs.hints_used += 1
    session.commit()
    return {"hint": q.hints[idx], "hints_used": qs.hints_used, "hints_total": len(q.hints)}


# --------------------------------------------------------------------------- #
#  Chests, inventory
# --------------------------------------------------------------------------- #
@router.post("/chests/{chest_id}/open")
def chest_open(chest_id: int, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    ps = _player(session, user)
    try:
        result = engine.open_chest(session, ps, chest_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    session.commit()
    return {**result, "state": engine.public_state(session, ps)}


@router.get("/inventory")
def inventory(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    ps = _player(session, user)
    items = {i.code: i for i in session.execute(select(GameItem)).scalars()}
    rows = list(
        session.execute(
            select(PlayerItem).where(PlayerItem.user_id == ps.user_id).order_by(PlayerItem.acquired_at.desc())
        ).scalars()
    )
    out = []
    for row in rows:
        item = items.get(row.item_code)
        if item:
            out.append({**engine._item_payload(item), "count": row.count, "acquired_at": row.acquired_at.isoformat()})
    return {"items": out, "selected_title": ps.selected_title, "selected_frame": ps.selected_frame}


class EquipIn(BaseModel):
    title: str | None = None
    frame: str | None = None


@router.post("/equip")
def equip(payload: EquipIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    ps = _player(session, user)
    owned = {pi.item_code for pi in session.execute(select(PlayerItem).where(PlayerItem.user_id == ps.user_id)).scalars()}
    if payload.title is not None:
        if payload.title and payload.title not in owned:
            raise HTTPException(400, "Этот титул ещё не получен.")
        ps.selected_title = payload.title
    if payload.frame is not None:
        if payload.frame and payload.frame not in owned:
            raise HTTPException(400, "Эта рамка ещё не получена.")
        ps.selected_frame = payload.frame
    session.commit()
    return {"selected_title": ps.selected_title, "selected_frame": ps.selected_frame}


# --------------------------------------------------------------------------- #
#  Profile, achievements, leaderboard
# --------------------------------------------------------------------------- #
_EVENT_LABELS = {
    "level.up": "Новый уровень",
    "concept.mastered": "Концепт освоен",
    "boss.defeated": "Босс повержен",
    "achievement.unlocked": "Достижение",
    "chest.opened": "Сундук открыт",
    "onboarding.completed": "Путь начат",
    "node.unlocked": "Открыт новый узел",
}


@router.get("/profile")
def profile(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    ps = _player(session, user)
    achievements = list(session.execute(select(GameAchievement)).scalars())
    unlocked = {
        pa.achievement_code: pa.unlocked_at
        for pa in session.execute(select(PlayerAchievement).where(PlayerAchievement.user_id == ps.user_id)).scalars()
    }
    nodes = {n.code: n for n in session.execute(select(GameNode)).scalars()}
    items_map = {i.code: i for i in session.execute(select(GameItem)).scalars()}
    achs_map = {a.code: a for a in achievements}

    events = list(
        session.execute(
            select(GameEvent)
            .where(GameEvent.user_id == ps.user_id, GameEvent.etype.in_(list(_EVENT_LABELS)))
            .order_by(GameEvent.id.desc())
            .limit(30)
        ).scalars()
    )
    timeline = []
    for e in events:
        detail = ""
        p = e.payload or {}
        if e.etype in {"concept.mastered", "boss.defeated", "node.unlocked"}:
            n = nodes.get(p.get("node", ""))
            detail = n.name_ru if n else ""
        elif e.etype == "level.up":
            detail = f"Уровень {p.get('to')}"
        elif e.etype == "achievement.unlocked":
            a = achs_map.get(p.get("code", ""))
            detail = a.name_ru if a else ""
        elif e.etype == "chest.opened":
            it = items_map.get(p.get("item", ""))
            detail = it.name_ru if it else ""
        timeline.append(
            {"type": e.etype, "label": _EVENT_LABELS.get(e.etype, e.etype), "detail": detail, "at": e.created_at.isoformat()}
        )

    session.commit()
    return {
        "state": engine.public_state(session, ps),
        "display_name": user.display_name or user.username,
        "username": user.username,
        "worlds": engine.world_powers(session, ps.user_id),
        "achievements": [
            {
                "code": a.code,
                "name": a.name_ru,
                "description": a.description_ru if (not a.secret or a.code in unlocked) else "Секретное достижение",
                "glyph": a.glyph,
                "xp": a.xp_reward,
                "unlocked": a.code in unlocked,
                "unlocked_at": unlocked[a.code].isoformat() if a.code in unlocked else None,
                "secret": a.secret,
            }
            for a in achievements
        ],
        "timeline": timeline,
    }


@router.get("/leaderboard")
def leaderboard(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    rows = list(
        session.execute(
            select(PlayerState, User)
            .join(User, User.id == PlayerState.user_id)
            .order_by(PlayerState.xp.desc())
            .limit(20)
        ).all()
    )
    return {
        "entries": [
            {
                "username": u.username,
                "display_name": u.display_name or u.username,
                "xp": ps.xp,
                "level": engine.level_from_xp(ps.xp),
                "rank": engine.rank_for_level(engine.level_from_xp(ps.xp)),
                "knowledge_power": ps.knowledge_power,
                "me": u.id == user.id,
            }
            for ps, u in rows
        ]
    }


# --------------------------------------------------------------------------- #
#  Teacher
# --------------------------------------------------------------------------- #
class TeacherIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    node_code: str = ""
    mode: str = "explain"


@router.post("/teacher")
def teacher(payload: TeacherIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    ps = _player(session, user)
    lp = engine.level_payload(ps.xp)
    fx: dict[str, Any] = {}
    streak_info = engine.touch_streak(session, ps)
    if streak_info:
        fx["streak"] = streak_info
    engine._event(session, ps.user_id, "teacher.message", {"mode": payload.mode})
    engine._bump_quest(session, ps, "teacher", fx)
    result = teacher_svc.teacher_reply(
        session, ps, message=payload.message.strip(), node_code=payload.node_code, mode=payload.mode,
        rank=lp["rank"], level=lp["level"],
    )
    engine._check_achievements(session, ps, fx)
    session.commit()
    return {**result, "fx": fx, "state": engine.public_state(session, ps)}


@router.get("/teacher/history")
def teacher_history(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    msgs = list(
        session.execute(
            select(GameChatMessage).where(GameChatMessage.user_id == user.id).order_by(GameChatMessage.id.desc()).limit(60)
        ).scalars()
    )[::-1]
    return {
        "messages": [
            {"role": m.role, "text": m.text, "mode": m.mode, "node": m.node_code, "via_ai": m.via_ai, "at": m.created_at.isoformat()}
            for m in msgs
        ]
    }

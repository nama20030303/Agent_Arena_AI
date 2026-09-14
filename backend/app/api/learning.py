"""
Learning routes: roadmap, topics, skill graph, daily plan, review queue, study sessions (§10-§18, §25).
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.api.deps import db_dep, get_current_user
from app.knowledge.retrieval import Retriever
from app.models import (
    CodingTask,
    Document,
    DocumentChunk,
    PracticeTask,
    Question,
    QuestionAttempt,
    Skill,
    Topic,
    User,
    UserSkill,
)
from app.services import learning_engine, progress
from app.services.graph import SkillGraph
from app.services.practice import task_payload as practice_payload
from app.services.questions import question_payload
from app.services.srs import due_summary, schedule_payload, stats as srs_stats

router = APIRouter(tags=["learning"])


@router.get("/roadmap")
def roadmap(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    levels = progress.roadmap(session, user)
    state = learning_engine.learner_state(session, user)
    return {
        "levels": levels,
        "current_level": {"index": user.level_index, "label": user.level_label},
        "goal": user.goal,
        "target_role": user.target_role,
        "weak_skills": state["weaknesses"],
        "recommendation": learning_engine.recommend_next(session, user),
    }


@router.get("/topics")
def topics(
    level: int | None = None,
    q: str = "",
    user: User = Depends(get_current_user),
    session: Session = Depends(db_dep),
) -> list[dict[str, Any]]:
    graph = SkillGraph(session)
    scores = graph.user_scores(user.id)
    rows = session.execute(select(Topic).order_by(Topic.level.asc(), Topic.order_index.asc())).scalars()
    out = []
    for topic in rows:
        if level is not None and topic.level != level:
            continue
        if q:
            haystack = " ".join([topic.name, topic.code, topic.summary, " ".join(topic.keywords or [])]).lower()
            if q.lower() not in haystack:
                continue
        skills = graph.topic_skills.get(topic.code, [])
        values = [scores.get(c, (0.0, 0.0))[0] for c in skills]
        avg = sum(values) / len(values) if values else 0.0
        out.append(
            {
                "code": topic.code,
                "name": topic.name,
                "level": topic.level,
                "domain": topic.domain,
                "summary": topic.summary,
                "keywords": (topic.keywords or [])[:10],
                "skills": skills,
                "avg_score": round(avg, 1),
                "prerequisites": topic.prerequisites or [],
                "est_hours": topic.est_hours,
            }
        )
    return out


@router.get("/topics/{code}")
def topic_detail(code: str, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    topic = session.execute(select(Topic).where(Topic.code == code)).scalar_one_or_none()
    if topic is None:
        raise HTTPException(status_code=404, detail=f"unknown topic {code}")
    graph = SkillGraph(session)
    readiness = graph.readiness(user.id, code)
    skills = []
    for skill_code in graph.topic_skills.get(code, []):
        detail = None
        from app.services.user_knowledge import skill_detail

        detail = skill_detail(session, user.id, skill_code)
        node = graph.nodes.get(skill_code)
        skills.append(
            {
                "code": skill_code,
                "name": node.name if node else skill_code,
                "category": node.category if node else "theory",
                "difficulty": node.difficulty if node else 2,
                "state": detail or {"knowledge_score": 0, "confidence": 0, "attempts": 0},
                "prerequisites": graph.prerequisites(skill_code, depth=1),
            }
        )
    question_count = int(session.execute(select(func.count(Question.id)).where(Question.topic_code == code)).scalar_one() or 0)
    practice_count = int(session.execute(select(func.count(PracticeTask.id)).where(PracticeTask.topic_code == code)).scalar_one() or 0)
    coding = list(session.execute(select(CodingTask).where(CodingTask.topic_code == code).limit(8)).scalars())
    documents = list(session.execute(select(Document).where(Document.status == Document.STATUS_INDEXED).limit(60)).scalars())
    relevant_docs = []
    for d in documents:
        tagged = code in (d.topics or [])
        if tagged or _mentions(session, code, topic):
            relevant_docs.append({"id": d.id, "title": d.title, "author": d.author, "chunks": d.chunk_count, "matched": tagged})
        if len(relevant_docs) >= 8:
            break
    retrieval = Retriever(session)
    hits = retrieval.search(f"{topic.name} {code.replace('_', ' ')}", top_k=6, topic_codes=[code])
    return {
        "code": topic.code,
        "name": topic.name,
        "level": topic.level,
        "domain": topic.domain,
        "summary": topic.summary,
        "keywords": topic.keywords or [],
        "est_hours": topic.est_hours,
        "prerequisites": topic.prerequisites or [],
        "skills": skills,
        "readiness": readiness,
        "counts": {"questions": question_count, "practice": practice_count, "coding": len(coding), "documents": len(relevant_docs)},
        "coding_tasks": [
            {"id": t.id, "slug": t.slug, "title": t.title, "difficulty": t.difficulty, "from_scratch": bool(t.from_scratch), "libraries": t.libraries or []}
            for t in coding
        ],
        "documents": relevant_docs,
        "library_hits": [
            {"document_id": h.document_id, "document": h.document_title, "chapter": h.chapter, "page": h.page_start, "score": round(h.score, 4), "snippet": h.text[:400], "citation": h.citation}
            for h in hits
        ],
        "suggested_modes": _modes_for(readiness, question_count, len(hits)),
    }


def _mentions(session: Session, code: str, topic: Topic) -> bool:
    terms = [t for t in (topic.keywords or []) if len(t) > 4][:4]
    if not terms:
        return False
    like = f"%{terms[0]}%"
    found = session.execute(select(DocumentChunk.id).where(DocumentChunk.text.ilike(like)).limit(1)).scalar()
    return bool(found)


def _modes_for(readiness: dict[str, Any], question_count: int, hits: int) -> list[str]:
    modes = ["explain"]
    if readiness["unmet"]:
        modes.insert(0, "practice")
    if question_count or hits:
        modes += ["quiz", "practice"]
    if hits >= 3:
        modes.append("deep_dive")
    modes += ["socratic", "code_review" if question_count else "review_answer"]
    return list(dict.fromkeys(modes))[:6]


@router.get("/skills/graph")
def skills_graph(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    return SkillGraph(session).graph_payload(user.id)


@router.get("/skills/{code}")
def skill_view(code: str, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    graph = SkillGraph(session)
    node = graph.nodes.get(code)
    if node is None:
        raise HTTPException(status_code=404, detail=f"unknown skill {code}")
    from app.services.user_knowledge import skill_detail

    raw_attempts = session.execute(
        select(QuestionAttempt, Question)
        .join(Question, Question.id == QuestionAttempt.question_id)
        .where(QuestionAttempt.user_id == user.id)
        .order_by(QuestionAttempt.id.desc())
        .limit(300)
    ).all()
    attempts = [(a, q) for a, q in raw_attempts if code in (q.skill_codes or [])][:8]
    return {
        "code": node.code,
        "name": node.name,
        "category": node.category,
        "level": node.level,
        "difficulty": node.difficulty,
        "topic": node.topic_code,
        "state": skill_detail(session, user.id, code),
        "prerequisites": [{"code": c, "name": graph.nodes[c].name, "score": round(graph.user_scores(user.id).get(c, (0, 0))[0], 1)} for c in graph.prerequisites(code, depth=1)],
        "unlocks": [c for c in graph.unlocks.get(code, [])],
        "path_from_python": graph.shortest_path("py.idioms", code),
        "recent_attempts": [
            {
                "question": (q.stem or "")[:200],
                "correct": a.is_correct,
                "score": a.score,
                "error_type": a.error_type,
                "at": a.created_at.isoformat() if a.created_at else None,
            }
            for a, q in attempts
        ],
    }


# --------------------------------------------------------------------------- #
#  Daily plan
# --------------------------------------------------------------------------- #
@router.get("/plan/today")
def plan_today(day: str = "", user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    plan_date = datetime.fromisoformat(day).date() if day else date.today()
    plan = learning_engine.get_or_build_plan(session, user, plan_date=plan_date)
    session.commit()
    from app.services.learning_engine import learner_state

    return _plan_payload(plan, learner_state(session, user))


def _plan_payload(plan: Any, state: dict[str, Any]) -> dict[str, Any]:
    items = plan.items or []
    done = sum(1 for i in items if i.get("done"))
    return {
        "date": plan.plan_date.isoformat(),
        "items": items,
        "rationale": plan.rationale,
        "minutes_budget": plan.minutes_budget,
        "used_minutes": sum(int(i.get("minutes") or 0) for i in items if i.get("done")),
        "completed": done,
        "total": len(items),
        "progress": round(done / len(items), 3) if items else 0.0,
        "generated_by": plan.generated_by,
        "focus_topic": next((i.get("code") for i in items if i.get("kind") == "learn"), ""),
        "weak_skills": state.get("weaknesses", [])[:6],
    }


class PlanRefreshIn(BaseModel):
    use_ai: bool = True
    day: str = ""


@router.post("/plan/today/refresh")
def plan_refresh(payload: PlanRefreshIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    plan_date = datetime.fromisoformat(payload.day).date() if payload.day else date.today()
    plan = learning_engine.build_daily_plan(session, user, plan_date=plan_date, reason="engine")
    directive_note = ""
    if payload.use_ai:
        from app.ai.manager import manager as ai_manager
        from app.services.learning_engine import candidate_topics, learner_state

        state = learner_state(session, user)
        if ai_manager.is_configured() and user.ai_enabled:
            result = ai_manager.invoke(
                "plan_learning",
                user_id=user.id,
                kwargs={
                    "learner_state": json.dumps(state, ensure_ascii=False)[:2600],
                    "candidates": [c.to_dict() for c in candidate_topics(session, user, limit=8)],
                    "due_reviews": (due_summary(session, user.id).get("urgent") or [])[:8],
                    "constraints": f"budget={plan.minutes_budget} minutes",
                },
                session=session,
            )
            if result.ok and isinstance(result.data, dict):
                plan = learning_engine.build_daily_plan(
                    session,
                    user,
                    plan_date=plan_date,
                    reason="engine+ai",
                    extra_directives=result.data.get("items") or [],
                )
                directive_note = "AI planner refined the order: " + str(result.data.get("rationale") or "")[:300]
            else:
                directive_note = "AI planner unavailable - using the local planner. " + (result.user_message() if not result.ok else "")
        else:
            directive_note = "Local planner (no AI provider configured)."
    session.commit()
    payload_out = _plan_payload(plan, learning_engine.learner_state(session, user))
    payload_out["note"] = directive_note
    return payload_out


class PlanItemIn(BaseModel):
    item_id: str
    done: bool = True


@router.post("/plan/item")
def plan_item(payload: PlanItemIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    plan = learning_engine.get_or_build_plan(session, user)
    if payload.done:
        learning_engine.mark_item_done(session, user, payload.item_id)
    else:
        ids = [i for i in (plan.completed_ids or []) if i != payload.item_id]
        plan.completed_ids = ids
        for item in plan.items or []:
            if item.get("id") == payload.item_id:
                item["done"] = False
        session.commit()
    fresh = learning_engine.get_or_build_plan(session, user, force=False)
    return _plan_payload(fresh, learning_engine.learner_state(session, user))


# --------------------------------------------------------------------------- #
#  Spaced-repetition review queue
# --------------------------------------------------------------------------- #
@router.get("/review/due")
def review_due(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    summary = due_summary(session, user.id)
    queue: list[dict[str, Any]] = []
    for entry in summary.get("urgent") or []:
        if entry["type"] != "skill":
            queue.append({**entry, "question": None})
            continue
        code = entry["code"]
        pool = list(session.execute(select(Question).where(Question.approved.is_(True)).order_by(Question.id.desc()).limit(500)).scalars())
        question = next((q for q in pool if code in (q.skill_codes or []) or q.topic_code == code), None)
        item = {**entry}
        if question is not None:
            item["question"] = question_payload(question)
        else:
            from app.services.questions import generate_questions

            generated = generate_questions(session, user, topic_code=code, skill_codes=[code], count=1, difficulty=2, types=["conceptual", "open"])
            if generated:
                item["question"] = question_payload(generated[0])
        queue.append(item)
    return {
        "summary": summary,
        "queue": queue,
        "stats": srs_stats(session, user.id),
    }


class ReviewAnswerIn(BaseModel):
    item_code: str = ""
    question_id: int
    answer: str = ""
    selected_option: int | None = None
    seconds: float = 0.0


@router.post("/review/answer")
def review_answer(payload: ReviewAnswerIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    from app.services.questions import submit_answer

    result = submit_answer(
        session,
        user,
        question_id=payload.question_id,
        answer=payload.answer,
        selected_option=payload.selected_option,
        time_seconds=payload.seconds,
        source="review",
    )
    return {
        "verdict": result["verdict"],
        "skill_updates": result["skill_updates"],
        "learning_engine": result["learning_engine"],
        "retry_question": result["retry_question"],
        "engine": result["engine"],
        "due": due_summary(session, user.id)["due_total"],
    }


# --------------------------------------------------------------------------- #
#  Study sessions
# --------------------------------------------------------------------------- #
class SessionIn(BaseModel):
    activity: str = "learn"
    topic_code: str = ""
    item_ref: str = ""


@router.post("/sessions/start")
def session_start(payload: SessionIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    row = progress.start_session(session, user, activity=payload.activity, topic_code=payload.topic_code, item_ref=payload.item_ref)
    session.commit()
    return {"id": row.id, "started_at": row.started_at.isoformat(), "activity": row.activity}


class SessionEndIn(BaseModel):
    id: int
    xp: int = 0
    items_done: int = 0


@router.post("/sessions/stop")
def session_stop(payload: SessionEndIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    result = progress.end_session(session, user, payload.id, xp=payload.xp, items_done=payload.items_done)
    session.commit()
    if result is None:
        raise HTTPException(status_code=404, detail="session not found")
    return result


# --------------------------------------------------------------------------- #
#  Self-improvement hooks
# --------------------------------------------------------------------------- #
@router.get("/learning/state")
def learning_state(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    state = learning_engine.learner_state(session, user)
    return {**state, "recommendation": learning_engine.recommend_next(session, user), "due": due_summary(session, user.id)}


@router.get("/learning/weak")
def learning_weak(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    graph = SkillGraph(session)
    weak = learning_engine.weak_skills(session, user.id, limit=10)
    for entry in weak:
        prereqs = graph.prerequisites(entry["code"], depth=1)
        scores = graph.user_scores(user.id)
        entry["blocking_prerequisites"] = [
            {"code": p, "name": graph.nodes[p].name, "score": round(scores.get(p, (0, 0))[0], 1)} for p in prereqs if scores.get(p, (0, 0))[0] < 55
        ]
    return {"weak_skills": weak, "strong_skills": learning_engine.strong_skills(session, user.id, limit=10), "recently_learned": learning_engine.recently_learned(session, user.id)}

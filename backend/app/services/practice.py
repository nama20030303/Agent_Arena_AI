"""
Practice Engine (§19, §21).

Every topic gets practice at four depths - beginner (compute it), intermediate
(implement it with the library), advanced (implement it from scratch), production
(serve and operate it). Tasks are generated on demand, persisted, and reused, so the
same task can be re-tested by the spaced-repetition scheduler.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.knowledge.retrieval import Retriever, build_context
from app.knowledge.textutil import content_hash
from app.models import PracticeAttempt, PracticeTask, StudySession, User
from app.services.graph import SkillGraph

LEVEL_ORDER = ["beginner", "intermediate", "advanced", "production"]


def task_payload(task: PracticeTask, *, with_solution: bool = False) -> dict[str, Any]:
    data = {
        "id": task.id,
        "title": task.title,
        "topic": task.topic_code,
        "level": task.level,
        "kind": task.kind,
        "difficulty": task.difficulty,
        "statement": task.statement,
        "given_data": task.given_data,
        "steps_required": task.steps_required or [],
        "starter_code": task.starter_code or "",
        "est_minutes": task.est_minutes,
        "skills": task.skill_codes or [],
        "citation": task.citation or {},
        "generated_by": task.generated_by,
    }
    if with_solution:
        data.update({"solution": task.solution, "explanation": task.explanation, "expected_answer": task.expected_answer, "expected_value": task.expected_value})
    return data


def list_tasks(
    session: Session,
    *,
    topic_code: str = "",
    level: str = "",
    kind: str = "",
    limit: int = 40,
    offset: int = 0,
) -> dict[str, Any]:
    stmt = select(PracticeTask)
    if topic_code:
        stmt = stmt.where(PracticeTask.topic_code == topic_code)
    if level:
        stmt = stmt.where(PracticeTask.level == level)
    if kind:
        stmt = stmt.where(PracticeTask.kind == kind)
    total = int(session.execute(select(func.count()).select_from(stmt.subquery())).scalar_one() or 0)
    rows = session.execute(stmt.order_by(PracticeTask.id.asc()).limit(limit).offset(offset)).scalars()
    return {"total": total, "items": [task_payload(t) for t in rows]}


def bank_tasks(session: Session, *, topic_code: str, level: str, count: int = 2) -> list[PracticeTask]:
    stmt = select(PracticeTask).where(or_(PracticeTask.topic_code == topic_code, PracticeTask.topic_code == ""))
    if level:
        stmt = stmt.where(PracticeTask.level == level)
    return list(session.execute(stmt.order_by(PracticeTask.id.asc()).limit(count)).scalars())


def generate_for_topic(
    session: Session,
    user: User,
    *,
    topic_code: str,
    level: str = "intermediate",
    count: int = 2,
    allow_ai: bool = True,
) -> list[PracticeTask]:
    """Bank first, then AI over the library, then the offline generator. Never fabricate a task with no content."""
    from app.services.learning_engine import difficulty_for

    graph = SkillGraph(session)
    topic_label = topic_code
    if topic_code in graph.topics:
        topic_label = graph.topics[topic_code].name
    elif topic_code in graph.nodes:
        topic_label = graph.nodes[topic_code].name

    existing = bank_tasks(session, topic_code=topic_code, level=level, count=count)
    if len(existing) >= count:
        return existing[:count]

    skills = graph.topic_skills.get(topic_code, [])
    difficulty = difficulty_for(session, user.id, skills or [topic_code])
    retriever = Retriever(session)
    results = retriever.search(f"{topic_label} {topic_code.replace('_', ' ')}", top_k=5)
    context, citations = build_context(results)

    created: list[PracticeTask] = []
    if allow_ai:
        from app.ai.manager import manager as ai_manager

        if ai_manager.is_configured() and user.ai_enabled:
            result = ai_manager.invoke(
                "generate_practice",
                user_id=user.id,
                kwargs={
                    "topic": topic_label or topic_code,
                    "level": level,
                    "context": context,
                    "learner_state": json.dumps({"level_index": user.level_index, "difficulty": difficulty, "weak": skills[:4]}, ensure_ascii=False),
                },
                session=session,
            )
            if result.ok and isinstance(result.data, dict):
                for raw in (result.data or {}).get("tasks", [])[:count]:
                    task = _from_ai(session, raw, topic_code=topic_code, level=level, skills=skills, user=user, citations=citations, source="ai")
                    if task is not None:
                        created.append(task)

    if len(created) < count:
        from app.ai.fallback import local_practice

        for i in range(count - len(created)):
            spec = local_practice(f"{topic_label or topic_code}", level, results[i % max(1, len(results)):]) if results else None
            if spec is None:
                spec = _fallback_from_bank_topic(session, topic_code, level, difficulty)
            if spec is None:
                continue
            task = _from_ai(session, spec, topic_code=topic_code, level=level, skills=skills, user=user, citations=citations, source="template")
            if task is not None:
                created.append(task)

    if created:
        session.add_all(created)
        session.flush()
    return created


def _fallback_from_bank_topic(session: Session, topic_code: str, level: str, difficulty: int) -> dict[str, Any] | None:
    from app.models import Topic

    topic = session.execute(select(Topic).where(Topic.code == topic_code)).scalar_one_or_none()
    if topic is None:
        return None
    return {
        "title": f"{topic.name}: reconstruct it from memory",
        "statement": (
            f"Without opening your notes, write:\n1. the definition of {topic.name};\n"
            "2. the formula or update rule (or pseudocode) if one exists;\n"
            "3. the failure mode you would check first in production;\n"
            "4. one numeric example with 3 data points.\n\n"
            "Then read the topic summary and repair only what was wrong."
        ),
        "given_data": f"Topic summary: {topic.summary[:600]}",
        "expected_answer": "A definition, a mechanism, a failure mode and a worked mini-example.",
        "steps_required": ["definition", "mechanism/formula", "failure mode", "numeric example"],
        "starter_code": "",
        "hints": ["Start from the smallest example you can compute by hand.", "If the formula is fuzzy, derive it for 2 points."],
        "solution": "Model answer follows the four parts above; grading is on whether the mechanism is stated correctly, not on length.",
        "explanation": "Reconstruction-from-memory is the fastest way to convert recognition into recall.",
        "difficulty": max(1, min(5, difficulty)),
        "est_minutes": 15,
        "kind": "concept",
    }


def _from_ai(
    session: Session,
    raw: dict[str, Any],
    *,
    topic_code: str,
    level: str,
    skills: list[str],
    user: User,
    citations: list[dict[str, Any]],
    source: str,
) -> PracticeTask | None:
    statement = str(raw.get("statement") or raw.get("title") or "").strip()
    title = str(raw.get("title") or "").strip() or statement[:80]
    if len(statement) < 30:
        return None
    digest = content_hash(title + statement)
    existing = session.execute(select(PracticeTask).where(PracticeTask.content_hash == digest)).scalar_one_or_none()
    if existing is not None:
        return None
    refs = [i for i in (raw.get("source_refs") or []) if isinstance(i, int) and 1 <= i <= len(citations)]
    return PracticeTask(
        topic_code=topic_code,
        skill_codes=list(raw.get("skills") or skills)[:6],
        level=level if level in LEVEL_ORDER else "intermediate",
        kind=str(raw.get("kind") or "concept"),
        title=title[:290],
        statement=statement[:8000],
        given_data=str(raw.get("given_data") or "")[:3000],
        expected_answer=str(raw.get("expected_answer") or "")[:3000],
        expected_value=str(raw.get("expected_value") or "")[:180],
        tolerance=float(raw.get("tolerance") or 0.0),
        steps_required=[str(s)[:300] for s in (raw.get("steps_required") or [])][:8],
        starter_code=str(raw.get("starter_code") or ""),
        hints=[str(h)[:400] for h in (raw.get("hints") or [])][:5],
        solution=str(raw.get("solution") or "")[:4000],
        explanation=str(raw.get("explanation") or "")[:4000],
        difficulty=max(1, min(5, int(raw.get("difficulty") or 2))),
        est_minutes=max(5, int(raw.get("est_minutes") or 15)),
        citation=citations[refs[0] - 1] if refs else {},
        source_chunk_id=citations[refs[0] - 1]["chunk_id"] if refs else None,
        generated_by=source,
        content_hash=digest,
    )


# --------------------------------------------------------------------------- #
#  Grading
# --------------------------------------------------------------------------- #
def grade_practice(task: PracticeTask, answer: str, code: str = "") -> dict[str, Any]:
    from app.ai.fallback import evaluate_locally

    pseudo = {
        "stem": task.statement[:2000],
        "type": "math" if task.kind == "math" else ("code" if code else "open"),
        "difficulty": task.difficulty,
        "expected_points": task.steps_required or [],
        "expected_answer": task.expected_answer or task.solution or "",
        "expected_value": task.expected_value or "",
        "tolerance": task.tolerance or 0.0,
        "options": [],
        "correct_option": None,
        "explanation": task.explanation or "",
        "skills": task.skill_codes or [],
        "citation": task.citation or {},
    }
    verdict = evaluate_locally(question=pseudo, answer=(answer or "") + (("\n\nCODE:\n" + code) if code else ""))
    verdict["steps_required"] = task.steps_required or []
    return verdict


def submit(
    session: Session,
    user: User,
    *,
    task_id: int,
    answer: str,
    code: str = "",
    hints_used: int = 0,
    seconds: float = 0.0,
    session_id: int | None = None,
    allow_ai: bool = True,
) -> dict[str, Any]:
    task = session.get(PracticeTask, int(task_id))
    if task is None:
        raise ValueError(f"Practice task {task_id} not found")

    verdict = grade_practice(task, answer, code)
    engine = "local"

    # A code task is graded by execution first: tests are ground truth over vibes.
    if code and task.starter_code:
        exec_result = None
        from app.models import CodingTask

        sibling = session.execute(
            select(CodingTask).where(CodingTask.topic_code == task.topic_code, CodingTask.test_code != "")
        ).scalars().first()
        if sibling is not None and sibling.test_code:
            from app.services.sandbox import run_python

            exec_result = run_python(code, sibling.test_code, banned_imports=sibling.banned_imports or [])
            if exec_result is not None:
                engine = "tests"
                passed = exec_result.passed_tests
                total = max(1, exec_result.total_tests)
                verdict = {
                    **verdict,
                    "correct": bool(exec_result.ok),
                    "score": round(100.0 * passed / total, 1),
                    "feedback": (
                        f"Tests: {passed}/{total} passed. " + (exec_result.error[:400] if exec_result.error else "")
                    ),
                    "what_is_wrong": "" if exec_result.ok else (exec_result.error[:400] or "some tests failed"),
                    "test_results": [asdict(t) if hasattr(t, "__dict__") else t for t in exec_result.tests],
                }

    if allow_ai and verdict.get("score", 0) and engine == "local":
        from app.ai.manager import manager as ai_manager

        if ai_manager.is_configured() and user.ai_enabled:
            result = ai_manager.invoke(
                "evaluate_answer",
                user_id=user.id,
                kwargs={
                    "question": {
                        "stem": task.statement[:1500],
                        "type": task.kind,
                        "difficulty": task.difficulty,
                        "expected_points": task.steps_required or [],
                        "expected_answer": (task.expected_answer or task.solution or "")[:1500],
                    },
                    "answer": (answer or "") + (("\n\nCODE:\n" + code) if code else ""),
                },
                session=session,
            )
            if result.ok and isinstance(result.data, dict):
                payload = result.data
                verdict = {
                    **verdict,
                    "correct": bool(payload.get("correct", verdict["correct"])),
                    "score": float(payload.get("score", verdict["score"])),
                    "dimensions": {**verdict["dimensions"], **(payload.get("dimensions") or {})},
                    "error_type": payload.get("error_type") or verdict["error_type"],
                    "what_is_wrong": payload.get("what_is_wrong") or verdict.get("what_is_wrong", ""),
                    "correction": payload.get("correction") or verdict.get("correction", ""),
                    "feedback": payload.get("feedback") or verdict["feedback"],
                    "missing_points": payload.get("missing_points") or verdict["missing_points"],
                    "check_question": payload.get("check_question") or verdict.get("check_question"),
                    "usage": asdict(result.usage),
                }
                engine = "ai"

    attempt = PracticeAttempt(
        user_id=user.id,
        task_id=task.id,
        answer=(answer or "")[:8000],
        code=(code or "")[:8000],
        is_correct=bool(verdict["correct"]),
        score=float(verdict["score"]),
        feedback=str(verdict.get("feedback") or "")[:4000],
        missing_steps=verdict.get("missing_points") or [],
        hints_used=int(hints_used or 0),
        evaluated_by=engine,
        time_seconds=float(seconds or 0),
        meta={"what_is_wrong": verdict.get("what_is_wrong", ""), "dimensions": verdict.get("dimensions", {})},
    )
    session.add(attempt)
    session.flush()

    from app.services.user_knowledge import update_from_activity

    activity = "code" if code else ("math" if task.kind == "math" else "practice")
    updates = update_from_activity(
        session,
        user_id=user.id,
        skill_codes=task.skill_codes or [],
        activity=activity,
        score=float(verdict["score"]),
        correct=bool(verdict["correct"]),
        dimensions={
            "math_score": float(verdict["dimensions"].get("precision", 0)) if task.kind == "math" else 0.0,
            "coding_score": min(100.0, float(verdict["score"])) if code else 0.0,
            "theory_score": float(verdict["dimensions"].get("correctness", 0)),
            "problem_solving_score": float(verdict["dimensions"].get("reasoning", 0)),
            "engineering_score": float(verdict["dimensions"].get("depth", 0)) if task.level in {"production", "advanced"} else 0.0,
        },
        difficulty=task.difficulty,
        xp=10 * int(task.difficulty or 2) if verdict["correct"] else 4,
        minutes=(seconds or 0) / 60.0,
        error_type=str(verdict.get("error_type") or ""),
    )

    from app.services.learning_engine import on_attempt_recorded

    loop = on_attempt_recorded(
        session,
        user,
        skill_codes=task.skill_codes or [],
        correct=bool(verdict["correct"]),
        score=float(verdict["score"]),
        error_type=str(verdict.get("error_type") or ""),
    )

    if session_id:
        study = session.get(StudySession, session_id)
        if study is not None:
            study.items_done = int(study.items_done or 0) + 1
            study.ai_used = study.ai_used or engine == "ai"

    session.commit()
    return {
        "attempt_id": attempt.id,
        "task_id": task.id,
        "verdict": verdict,
        "engine": engine,
        "skill_updates": [asdict(u) for u in updates],
        "learning_engine": loop,
        "hint_available": bool(task.hints),
        "solution": task.solution if (verdict["correct"] or hints_used >= 2) else "",
    }


def next_for_today(session: Session, user: User, *, count: int = 2, topic_code: str = "") -> list[PracticeTask]:
    """Tasks the learner has not solved yet, focused on weak skills, plus generation when the bank is empty."""
    graph = SkillGraph(session)
    from app.services.learning_engine import weak_skills

    solved = set(
        session.execute(
            select(PracticeAttempt.task_id).where(PracticeAttempt.user_id == user.id, PracticeAttempt.is_correct.is_(True))
        ).scalars()
    )
    weak = [w["code"] for w in weak_skills(session, user.id, limit=6)]
    candidates = list(session.execute(select(PracticeTask).limit(400)).scalars())
    wanted = set(weak) | set(graph.topic_skills.get(topic_code, []))
    level = LEVEL_ORDER[min(3, max(0, (user.level_index or 0) // 3))]

    def key(task: PracticeTask) -> tuple[Any, ...]:
        overlap = len(set(task.skill_codes or []) & wanted)
        return (0 if task.id in solved else 1, -(overlap + (1 if task.level == level else 0)), task.difficulty, task.id)

    candidates.sort(key=key)
    picked = [t for t in candidates if t.id not in solved][:count]
    if len(picked) < count and topic_code:
        picked += [t for t in generate_for_topic(session, user, topic_code=topic_code, level=level, count=count - len(picked)) if t not in picked]
    return picked[:count]


def attempts_for(session: Session, user: User, task_id: int, limit: int = 10) -> list[dict[str, Any]]:
    rows = session.execute(
        select(PracticeAttempt).where(PracticeAttempt.user_id == user.id, PracticeAttempt.task_id == task_id).order_by(PracticeAttempt.id.desc()).limit(limit)
    ).scalars()
    return [
        {
            "id": a.id,
            "answer": a.answer[:2000],
            "correct": a.is_correct,
            "score": a.score,
            "feedback": a.feedback,
            "missing_steps": a.missing_steps or [],
            "engine": a.evaluated_by,
            "at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in rows
    ]

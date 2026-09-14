"""Question engine + practice endpoints (§16, §17, §19, §21)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import db_dep, get_current_user
from app.models import DocumentChunk, PracticeAttempt, PracticeTask, Question, QuestionAttempt, User
from app.services import learning_engine, practice as practice_service, questions as question_service

router = APIRouter(tags=["questions & practice"])


class NextQuestionIn(BaseModel):
    topic_code: str = ""
    skill_codes: list[str] = []
    types: list[str] = []
    difficulty: int | None = None
    include_answered: bool = False


@router.get("/questions")
def list_questions(topic_code: str = "", qtype: str = "", limit: int = 50, offset: int = 0, session: Session = Depends(db_dep)) -> dict[str, Any]:
    return question_service.list_questions(session, topic_code=topic_code, qtype=qtype, limit=limit, offset=offset)


@router.get("/questions/stats")
def question_stats(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    by_type = session.execute(
        select(Question.question_type, func.count(Question.id)).group_by(Question.question_type)
    ).all()
    by_source = session.execute(select(Question.generated_by, func.count(Question.id)).group_by(Question.generated_by)).all()
    total_attempts = int(session.execute(select(func.count(QuestionAttempt.id)).where(QuestionAttempt.user_id == user.id)).scalar_one() or 0)
    correct = int(
        session.execute(
            select(func.count(QuestionAttempt.id)).where(QuestionAttempt.user_id == user.id, QuestionAttempt.is_correct.is_(True))
        ).scalar_one()
        or 0
    )
    return {
        "bank_size": int(session.execute(select(func.count(Question.id))).scalar_one() or 0),
        "by_type": {t: int(n) for t, n in by_type},
        "by_source": {s: int(n) for s, n in by_source},
        "your_attempts": total_attempts,
        "your_correct": correct,
        "your_accuracy": round(100.0 * correct / total_attempts, 1) if total_attempts else None,
        "grounded_in_library": int(session.execute(select(func.count(Question.id)).where(Question.source_chunk_id.isnot(None))).scalar_one() or 0),
    }


@router.get("/questions/{question_id}/history")
def question_history(question_id: int, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> list[dict[str, Any]]:
    rows = session.execute(
        select(QuestionAttempt).where(QuestionAttempt.user_id == user.id, QuestionAttempt.question_id == question_id).order_by(QuestionAttempt.id.desc()).limit(20)
    ).scalars()
    return [
        {
            "id": a.id,
            "answer": a.answer[:1500],
            "selected_option": a.selected_option,
            "correct": a.is_correct,
            "score": a.score,
            "dimensions": {
                "correctness": a.correctness,
                "depth": a.depth,
                "reasoning": a.reasoning,
                "precision": a.precision,
                "confidence": a.confidence,
            },
            "error_type": a.error_type,
            "missing_points": a.missing_points or [],
            "feedback": a.feedback,
            "correction": a.correction,
            "engine": a.evaluated_by,
            "seconds": a.time_seconds,
            "at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in rows
    ]


@router.get("/questions/{question_id}")
def get_question(question_id: int, show_answer: bool = False, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    question = session.get(Question, question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="question not found")
    payload = question_service.question_payload(question, with_answer=show_answer)
    if question.source_chunk_id:
        chunk = session.get(DocumentChunk, question.source_chunk_id)
        if chunk is not None:
            payload["source_excerpt"] = chunk.text[:900]
    attempts = session.execute(
        select(QuestionAttempt).where(QuestionAttempt.user_id == user.id, QuestionAttempt.question_id == question.id).order_by(QuestionAttempt.id.desc()).limit(5)
    ).scalars()
    payload["your_attempts"] = [
        {"id": a.id, "correct": a.is_correct, "score": a.score, "feedback": a.feedback, "at": a.created_at.isoformat() if a.created_at else None} for a in attempts
    ]
    return payload


@router.post("/questions/next")
def next_question(payload: NextQuestionIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    result = question_service.select_next(
        session,
        user,
        topic_code=payload.topic_code,
        skill_codes=payload.skill_codes or None,
        types=payload.types or None,
        difficulty=payload.difficulty,
        include_answered=payload.include_answered,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="no question available for that topic - seed the bank or index material")
    session.commit()
    return result


class GenerateIn(BaseModel):
    topic_code: str = Field(min_length=1)
    count: int = 3
    types: list[str] = []
    difficulty: int | None = None


@router.post("/questions/generate")
def generate(payload: GenerateIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    from app.ai.manager import manager as ai_manager

    created = question_service.generate_questions(
        session,
        user,
        topic_code=payload.topic_code,
        count=max(1, min(6, payload.count)),
        difficulty=payload.difficulty or learning_engine.difficulty_for(session, user.id, []),
        types=payload.types or None,
    )
    session.commit()
    return {
        "created": [question_service.question_payload(q) for q in created],
        "count": len(created),
        "engine": "ai" if ai_manager.is_configured() else "template-over-knowledge-base",
        "message": (
            f"{len(created)} question(s) generated and stored (deduplicated against the bank)."
            if created
            else "Nothing generated: the library has no material for this topic yet, so a grounded question cannot be authored."
        ),
    }


class AnswerIn(BaseModel):
    question_id: int
    answer: str = ""
    selected_option: int | None = None
    code: str = ""
    seconds: float = 0.0
    session_id: int | None = None
    allow_ai: bool = True
    source: str = "learn"


@router.post("/questions/answer")
def answer(payload: AnswerIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    try:
        result = question_service.submit_answer(
            session,
            user,
            question_id=payload.question_id,
            answer=payload.answer,
            selected_option=payload.selected_option,
            code=payload.code,
            time_seconds=payload.seconds,
            session_id=payload.session_id,
            allow_ai=payload.allow_ai,
            source=payload.source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result


class ReviewIn(BaseModel):
    prompt: str = Field(min_length=8)
    answer: str = Field(min_length=1)
    topic_code: str = ""
    context: str = ""


@router.post("/questions/review-answer")
def review_answer(payload: ReviewIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    result = question_service.review_free_text(
        session, user, prompt=payload.prompt, answer=payload.answer, topic_code=payload.topic_code, context=payload.context
    )
    return {"verdict": result, "engine": result.get("engine", "local")}


# --------------------------------------------------------------------------- #
#  Practice
# --------------------------------------------------------------------------- #
@router.get("/practice")
def list_practice(topic_code: str = "", level: str = "", kind: str = "", limit: int = 30, offset: int = 0, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    payload = practice_service.list_tasks(session, topic_code=topic_code, level=level, kind=kind, limit=limit, offset=offset)
    solved = set(session.execute(select(PracticeAttempt.task_id).where(PracticeAttempt.user_id == user.id, PracticeAttempt.is_correct.is_(True))).scalars())
    for item in payload["items"]:
        item["solved"] = item["id"] in solved
    return payload


@router.get("/practice/next")
def practice_next(topic_code: str = "", count: int = 2, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    tasks = practice_service.next_for_today(session, user, count=max(1, min(4, count)), topic_code=topic_code)
    return {"items": [practice_service.task_payload(t) for t in tasks], "count": len(tasks)}


class PracticeGenerateIn(BaseModel):
    topic_code: str = Field(min_length=1)
    level: str = "intermediate"
    count: int = 2


@router.post("/practice/generate")
def practice_generate(payload: PracticeGenerateIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    from app.ai.manager import manager as ai_manager

    tasks = practice_service.generate_for_topic(
        session, user, topic_code=payload.topic_code, level=payload.level, count=max(1, min(4, payload.count))
    )
    session.commit()
    return {
        "items": [practice_service.task_payload(t) for t in tasks],
        "engine": "ai" if any(t.generated_by == "ai" for t in tasks) else "offline",
        "message": "" if tasks else "No practice could be derived: the topic has neither bank entries nor library material.",
        "ai_note": "" if ai_manager.is_configured() else "Offline generator used (no AI provider configured).",
    }


@router.get("/practice/{task_id}")
def get_practice(task_id: int, show_solution: bool = False, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    task = session.get(PracticeTask, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="practice task not found")
    solved = bool(
        session.execute(
            select(PracticeAttempt.id).where(PracticeAttempt.user_id == user.id, PracticeAttempt.task_id == task.id, PracticeAttempt.is_correct.is_(True)).limit(1)
        ).scalar()
    )
    return {
        **practice_service.task_payload(task, with_solution=show_solution and (solved or True)),
        "solved": solved,
        "attempts": practice_service.attempts_for(session, user, task.id),
    }


class PracticeSubmitIn(BaseModel):
    task_id: int
    answer: str = ""
    code: str = ""
    hints_used: int = 0
    seconds: float = 0.0
    session_id: int | None = None


@router.post("/practice/submit")
def practice_submit(payload: PracticeSubmitIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    try:
        return practice_service.submit(
            session,
            user,
            task_id=payload.task_id,
            answer=payload.answer,
            code=payload.code,
            hints_used=payload.hints_used,
            seconds=payload.seconds,
            session_id=payload.session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

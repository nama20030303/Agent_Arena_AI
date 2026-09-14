"""Coding Lab endpoints (§20, §21) - execution happens in the isolated sandbox."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import db_dep, get_current_user
from app.models import CodingAttempt, CodingTask, User
from app.services import coding as coding_service
from app.services.sandbox import sandbox_capabilities

router = APIRouter(tags=["coding lab"])


@router.get("/coding/capabilities")
def capabilities(user: User = Depends(get_current_user)) -> dict[str, Any]:
    return sandbox_capabilities()


@router.get("/coding/tasks")
def list_tasks(
    topic_code: str = "",
    skill: str = "",
    level: str = "",
    only_unsolved: bool = False,
    limit: int = 60,
    offset: int = 0,
    user: User = Depends(get_current_user),
    session: Session = Depends(db_dep),
) -> dict[str, Any]:
    return coding_service.list_tasks(
        session, user, topic_code=topic_code, skill=skill, level=level, only_unsolved=only_unsolved, limit=limit, offset=offset
    )


@router.get("/coding/tasks/{ident}")
def get_task(ident: str, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    task = coding_service.get_task(session, user, ident)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return coding_service.task_payload(task, user_id=user.id, session=session)


@router.get("/coding/tasks/{ident}/solution")
def get_solution(ident: str, force: bool = False, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    task = coding_service.get_task(session, user, ident)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return coding_service.reveal_solution(session, user, task, force=force)


class RunIn(BaseModel):
    code: str = Field(default="", max_length=80000)
    persist: bool = False
    hints_used: int = 0


@router.post("/coding/tasks/{ident}/run")
def run_task(ident: str, payload: RunIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    """Execute against the hidden tests. Nothing is recorded unless the run passes."""
    task = coding_service.get_task(session, user, ident)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    result = coding_service.run(session, user, task, payload.code, persist=payload.persist, hints_used=payload.hints_used)
    if payload.persist:
        session.commit()
    return result


class SubmitIn(BaseModel):
    code: str = Field(min_length=1, max_length=80000)
    hints_used: int = 0
    seconds: float = 0.0
    session_id: int | None = None


@router.post("/coding/tasks/{ident}/submit")
def submit_task(ident: str, payload: SubmitIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    task = coding_service.get_task(session, user, ident)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return coding_service.submit(session, user, task_id=task.id, code=payload.code, hints_used=payload.hints_used, seconds=payload.seconds, session_id=payload.session_id)


@router.get("/coding/tasks/{ident}/hints")
def hints(ident: str, index: int = 0, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    task = coding_service.get_task(session, user, ident)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    pool = list(task.hints or [])
    if not pool:
        return {"available": 0, "hint": "", "message": "This task has no hints."}
    i = max(0, min(index, len(pool) - 1))
    return {"available": len(pool), "index": i, "hint": pool[i], "all_after_solve": _solution_unlocked(session, user, task)}


def _solution_unlocked(session: Session, user: User, task: CodingTask) -> bool:
    return bool(
        session.execute(
            select(CodingAttempt.id).where(CodingAttempt.user_id == user.id, CodingAttempt.task_id == task.id, CodingAttempt.passed.is_(True)).limit(1)
        ).scalar()
    )


@router.post("/coding/check")
def static_check(payload: dict[str, Any]) -> dict[str, Any]:
    return coding_service.static_check(str(payload.get("code") or ""))


class GenerateTasksIn(BaseModel):
    topic_code: str = Field(min_length=1)
    count: int = 2


@router.post("/coding/generate")
def generate_tasks(payload: GenerateTasksIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    result = coding_service.generate_tasks(session, user, topic_code=payload.topic_code, count=max(1, min(3, payload.count)))
    return result


@router.get("/coding/attempts")
def attempts(limit: int = 30, task_id: int | None = None, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> list[dict[str, Any]]:
    stmt = (
        select(CodingAttempt, CodingTask)
        .join(CodingTask, CodingTask.id == CodingAttempt.task_id)
        .where(CodingAttempt.user_id == user.id)
    )
    if task_id:
        stmt = stmt.where(CodingAttempt.task_id == task_id)
    rows = session.execute(stmt.order_by(CodingAttempt.id.desc()).limit(min(limit, 100))).all()
    return [
        {
            "id": a.id,
            "task": t.title,
            "task_id": t.id,
            "slug": t.slug,
            "passed": a.passed,
            "passed_tests": a.passed_tests,
            "total_tests": a.total_tests,
            "runtime_ms": a.runtime_ms,
            "hints_used": a.hints_used,
            "error": (a.error or "")[:400],
            "at": a.created_at.isoformat() if a.created_at else None,
        }
        for a, t in rows
    ]


@router.get("/coding/stats")
def stats(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    total_tasks = int(session.execute(select(func.count(CodingTask.id)).where(CodingTask.test_code != "")).scalar_one() or 0)
    from_scratch = int(session.execute(select(func.count(CodingTask.id)).where(CodingTask.from_scratch.is_(True))).scalar_one() or 0)
    solved = int(
        session.execute(
            select(func.count(func.distinct(CodingAttempt.task_id))).where(CodingAttempt.user_id == user.id, CodingAttempt.passed.is_(True))
        ).scalar_one()
        or 0
    )
    runs = int(session.execute(select(func.count(CodingAttempt.id)).where(CodingAttempt.user_id == user.id)).scalar_one() or 0)
    first_pass = int(
        session.execute(
            select(func.count(func.distinct(CodingAttempt.task_id)))
            .select_from(CodingAttempt)
            .where(CodingAttempt.user_id == user.id, CodingAttempt.passed.is_(True), CodingAttempt.hints_used == 0)
        ).scalar_one()
        or 0
    )
    return {
        "tasks": total_tasks,
        "from_scratch_tasks": from_scratch,
        "solved": solved,
        "runs": runs,
        "solved_without_hints": first_pass,
        "accuracy": round(100.0 * solved / total_tasks, 1) if total_tasks else None,
    }

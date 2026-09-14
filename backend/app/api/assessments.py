"""Exams (§28) and mock interviews (§29)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import db_dep, get_current_user
from app.models import User
from app.services import exams as exam_service
from app.services import interview as interview_service

router = APIRouter(tags=["exams & interview"])


# ---------------------------------------------------------------- exams
@router.get("/exams")
def list_exams(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    return {"items": exam_service.list_exams(session), "readiness": exam_service.exam_readiness(session, user)}


@router.get("/exams/history")
def exam_history(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> list[dict[str, Any]]:
    return exam_service.history(session, user)


@router.post("/exams/{code}/start")
def start_exam(code: str, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    try:
        payload = exam_service.start(session, user, code)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not payload["items"]:
        raise HTTPException(status_code=409, detail="No questions available for this exam yet - seed content or generate questions first.")
    return payload


@router.get("/exams/attempts/{attempt_id}")
def exam_attempt(attempt_id: int, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    from app.models import Exam, ExamAttempt

    attempt = session.get(ExamAttempt, int(attempt_id))
    if attempt is None or attempt.user_id != user.id:
        raise HTTPException(status_code=404, detail="attempt not found")
    return exam_service.payload_for(session, session.get(Exam, attempt.exam_id), attempt)


class ExamAnswersIn(BaseModel):
    answers: dict[str, Any]


@router.post("/exams/attempts/{attempt_id}/answers")
def save_exam_answers(attempt_id: int, payload: ExamAnswersIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    try:
        return exam_service.save_answers(session, user, attempt_id, payload.answers)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/exams/attempts/{attempt_id}/submit")
def submit_exam(attempt_id: int, allow_ai: bool = True, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    try:
        return exam_service.submit(session, user, attempt_id, allow_ai=allow_ai)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------- interview
class InterviewStartIn(BaseModel):
    level: str = Field(default="middle", pattern="^(junior|middle|senior)$")
    focus: str = ""


@router.post("/interview/start")
def interview_start(payload: InterviewStartIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    try:
        return interview_service.start(session, user, level=payload.level, focus=payload.focus)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


class InterviewAnswerIn(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    seconds: float = 0.0


@router.post("/interview/{interview_id}/answer")
def interview_answer(interview_id: int, payload: InterviewAnswerIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    try:
        return interview_service.answer(session, user, interview_id, payload.text, seconds=payload.seconds)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/interview/{interview_id}/finish")
def interview_finish(interview_id: int, allow_ai: bool = True, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    try:
        return interview_service.finish(session, user, interview_id, allow_ai=allow_ai)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/interview/history")
def interview_history(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> list[dict[str, Any]]:
    return interview_service.history(session, user)


@router.get("/interview/{interview_id}")
def interview_get(interview_id: int, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    try:
        return interview_service.snapshot(session, user, interview_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

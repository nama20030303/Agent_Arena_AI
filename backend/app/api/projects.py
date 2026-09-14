"""Project engine endpoints (§22, §23, §24)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import db_dep, get_current_user
from app.models import Project, ProjectEnrollment, User
from app.services import projects as project_service

router = APIRouter(tags=["projects"])


@router.get("/projects")
def list_projects(level: str = "", include_all: bool = False, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    return project_service.list_projects(session, user, level=level, include_all=include_all)


@router.get("/projects/active")
def active(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    payload = project_service.active_enrollment(session, user)
    return {"enrollment": payload, "none": payload is None}


@router.post("/projects/enroll")
def enroll(payload: EnrollIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    try:
        return project_service.enroll(session, user, payload.project_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/projects/generate")
def generate(payload: GenerateIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    return project_service.generate_project(session, user, brief=payload.brief, level=payload.level)


class EnrollIn(BaseModel):
    project_id: int


@router.get("/projects/enrollments/{enrollment_id}")
def enrollment(enrollment_id: int, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    row = session.get(ProjectEnrollment, int(enrollment_id))
    if row is None or row.user_id != user.id:
        raise HTTPException(status_code=404, detail="enrollment not found")
    payload = project_service.enrollment_payload(session, row, include_log=True)
    return {**(payload or {}), "project": project_service.project_payload(session, row.project, user=user)}


class MilestoneStatusIn(BaseModel):
    status: str = Field(pattern="^(todo|doing|review|done)$")
    notes: str = ""


@router.post("/projects/milestones/{task_id}/status")
def milestone_status(task_id: int, payload: MilestoneStatusIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    result = project_service.set_milestone_status(session, user, task_id, payload.status, notes=payload.notes)
    if result is None:
        raise HTTPException(status_code=404, detail="milestone not found")
    return result


class MentorIn(BaseModel):
    message: str = Field(min_length=2, max_length=12000)
    enrollment_id: int | None = None


@router.post("/projects/mentor")
def mentor(payload: MentorIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    return project_service.mentor_reply(session, user, payload.message, enrollment_id=payload.enrollment_id)


class SubmissionIn(BaseModel):
    enrollment_id: int
    task_id: int | None = None
    content: str = Field(min_length=1, max_length=40000)
    user_implemented: str = ""
    files: list[dict[str, Any]] = []


@router.post("/projects/submission")
def submission(payload: SubmissionIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    try:
        return project_service.record_submission(
            session,
            user,
            enrollment_id=payload.enrollment_id,
            task_id=payload.task_id,
            content=payload.content,
            user_implemented=payload.user_implemented,
            files=payload.files,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/projects/evaluate")
def evaluate(enrollment_id: int, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    try:
        return project_service.evaluate(session, user, enrollment_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/projects/minutes")
def minutes(enrollment_id: int, minutes: float = 10.0, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    project_service.project_minutes(session, enrollment_id, minutes)
    return {"ok": True}


@router.get("/projects/{project_id}")
def get_project(project_id: int, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    project = session.get(Project, int(project_id))
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return project_service.project_payload(session, project, user=user)


class GenerateIn(BaseModel):
    brief: str = ""
    level: str = "auto"

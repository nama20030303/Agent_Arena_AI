"""
Project Engine + Mentor + Evaluation (§22, §23, §24).

Projects are curated for the learner's level or generated from the current skill model.
The mentor deliberately refuses to write the whole project: it returns architecture,
interfaces, risks and the next concrete step, and it records *who* produced what
(`user_implemented` vs `ai_suggested`) so the portfolio stays honest.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.knowledge.retrieval import Retriever, build_context
from app.knowledge.textutil import content_hash
from app.models import (
    CodingAttempt,
    Project,
    ProjectAttempt,
    ProjectEnrollment,
    ProjectTask,
    StudySession,
    User,
    UserSkill,
    Skill,
)
from app.services.graph import SkillGraph

RUBRIC_DIMENSIONS = (
    "Architecture",
    "Code Quality",
    "ML correctness",
    "Data handling",
    "Testing",
    "Deployment",
    "Monitoring",
    "Documentation",
    "Scalability",
)

DO_IT_FOR_ME = re.compile(
    r"(?:\b(write|give|produce|generate|code|do|make|build)\b[^.?\n]{0,40}?"
    r"(?:\b(entire|whole|full|all|complete)\b[^.?\n]{0,20}?)?"
    r"\b(code|implementation|project|solution|app|service|everything)\b)"
    r"|(?:\b(напиши|сделай|сгенерируй|давай)\b[^.?\n]{0,40}?\b(весь|всё|весь\s+код|код|проект)\b)",
    re.IGNORECASE,
)


def project_payload(session: Session, project: Project, *, user: User | None = None) -> dict[str, Any]:
    enrollment = None
    if user is not None:
        enrollment = session.execute(
            select(ProjectEnrollment).where(ProjectEnrollment.user_id == user.id, ProjectEnrollment.project_id == project.id)
            .order_by(ProjectEnrollment.id.desc())
        ).scalars().first()
    return {
        "id": project.id,
        "slug": project.slug,
        "title": project.title,
        "description": project.description,
        "target_level": project.target_level,
        "domain": project.domain,
        "difficulty": project.difficulty,
        "est_hours": project.est_hours,
        "stack": project.stack or [],
        "skills": project.skill_codes or [],
        "deliverables": project.deliverables or [],
        "milestones": project.milestones or [],
        "rubric": project.rubric or [],
        "guidance": project.guidance,
        "dataset_hint": project.dataset_hint,
        "generated_by": project.generated_by,
        "enrollment": enrollment_payload(session, enrollment) if enrollment else None,
    }


def enrollment_payload(session: Session, enrollment: ProjectEnrollment | None, *, include_log: bool = False) -> dict[str, Any] | None:
    if enrollment is None:
        return None
    project = enrollment.project
    tasks = list(enrollment.tasks or [])
    done = sum(1 for t in tasks if t.status == "done")
    log = []
    if include_log:
        for attempt in enrollment.attempts or []:
            log.append(
                {
                    "id": attempt.id,
                    "kind": attempt.kind,
                    "author": attempt.author,
                    "content": (attempt.content or "")[:4000],
                    "user_implemented": (attempt.user_implemented or "")[:2000],
                    "ai_suggested": (attempt.ai_suggested or "")[:2000],
                    "evaluation": attempt.evaluation or {},
                    "task_id": attempt.project_task_id,
                    "at": attempt.created_at.isoformat() if attempt.created_at else None,
                }
            )
    return {
        "id": enrollment.id,
        "project_id": enrollment.project_id,
        "project_title": project.title if project else "",
        "status": enrollment.status,
        "started_at": enrollment.started_at.isoformat() if enrollment.started_at else None,
        "submitted_at": enrollment.submitted_at.isoformat() if enrollment.submitted_at else None,
        "minutes_spent": round(float(enrollment.minutes_spent or 0.0), 1),
        "milestones": [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "acceptance_criteria": t.acceptance_criteria or [],
                "order": t.order_index,
                "status": t.status,
                "review_notes": t.review_notes,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            }
            for t in tasks
        ],
        "progress": round(done / len(tasks), 3) if tasks else 0.0,
        "milestones_done": done,
        "milestones_total": len(tasks),
        "log": log,
    }


def list_projects(session: Session, user: User, *, level: str = "", include_all: bool = False) -> dict[str, Any]:
    graph = SkillGraph(session)
    rows = list(session.execute(select(Project).order_by(Project.difficulty.asc(), Project.id.asc())).scalars())
    state = SkillGraph(session).user_scores(user.id)
    out = []
    for project in rows:
        if level and project.target_level != level:
            continue
        required = [c for c in (project.skill_codes or []) if c in graph.nodes]
        scores = [state.get(c, (0.0, 0.0))[0] for c in required]
        readiness = round(sum(scores) / len(scores), 1) if scores else 0.0
        recommended = readiness >= 55 or (user.level_index or 0) >= int(project.difficulty or 2) * 2
        payload = project_payload(session, project, user=user)
        payload["skill_readiness"] = readiness
        payload["recommended"] = bool(recommended)
        out.append(payload)
    if not include_all:
        recommended_rows = [p for p in out if p["recommended"] or p["enrollment"]]
        out = recommended_rows or out[:3]
    return {"items": out, "total": len(rows)}


def generate_project(session: Session, user: User, *, brief: str = "", level: str = "auto") -> dict[str, Any]:
    from app.ai.manager import manager as ai_manager

    graph = SkillGraph(session)
    state_scores = graph.user_scores(user.id)
    weak = [c for c, (s, _c) in state_scores.items() if s < 55][:8]
    resolved_level = level if level != "auto" else {0: "beginner", 1: "beginner", 2: "junior", 3: "junior", 4: "middle", 5: "middle", 6: "strong_middle", 7: "strong_middle", 8: "senior", 9: "senior", 10: "senior"}.get(int(user.level_index or 0), "middle")

    retriever = Retriever(session)
    results = retriever.search(" ".join(weak[:4]) or f"{resolved_level} ml project", top_k=4)
    context, _ = build_context(results, max_chars=4000)

    learner_state = {
        "level_index": user.level_index,
        "level_label": user.level_label,
        "weak_skills": weak,
        "strong_skills": [c for c, (s, _f) in state_scores.items() if s >= 75][:8],
        "goal": user.goal,
    }

    if not ai_manager.is_configured():
        from app.services.learning_engine import recommend_next

        rec = recommend_next(session, user)
        topic = (rec.get("topic") or {}).get("code", "")
        label = (rec.get("topic") or {}).get("name", "your current topic")
        existing = session.execute(
            select(Project).where(Project.target_level == resolved_level).order_by(Project.id.asc())
        ).scalars().first()
        return {
            "created": [],
            "suggested": [project_payload(session, existing, user=user)] if existing else [],
            "message": (
                "Project generation needs an AI provider to produce a *new* brief. Offline, I recommend an existing "
                f"curated project at your level{f' (current topic: {label})' if label else ''}. "
                "Configure AI in .env to generate a custom one."
            ),
            "ai_error_code": "disabled",
        }

    result = ai_manager.invoke(
        "generate_project",
        user_id=user.id,
        kwargs={
            "learner_state": json.dumps(learner_state, ensure_ascii=False),
            "level": resolved_level,
            "brief": brief[:1000],
            "context": context,
        },
        session=session,
    )
    if not result.ok or not isinstance(result.data, dict):
        return {"created": [], "suggested": [], "message": f"Generation failed: {result.error or result.user_message()}", "ai_error_code": result.error_code}

    raw = result.data
    title = str(raw.get("title") or "").strip()[:200]
    description = str(raw.get("description") or "").strip()[:6000]
    if not title or len(description) < 40:
        return {"created": [], "suggested": [], "message": "The model returned an incomplete project; nothing was stored.", "ai_error_code": "parse"}
    digest = content_hash(title + description[:400])
    if session.execute(select(Project).where(Project.content_hash == digest)).scalar_one_or_none():
        return {"created": [], "suggested": [], "message": "An identical project already exists in your list.", "ai_error_code": "duplicate"}

    milestones = []
    for item in (raw.get("milestones") or [])[:8]:
        if not isinstance(item, dict):
            continue
        milestones.append(
            {
                "title": str(item.get("title") or "Milestone")[:200],
                "description": str(item.get("description") or "")[:3000],
                "acceptance_criteria": [str(c)[:400] for c in (item.get("acceptance_criteria") or [])][:6],
            }
        )
    rubric = []
    for item in (raw.get("rubric") or [])[:12]:
        if isinstance(item, dict) and item.get("dimension"):
            rubric.append(
                {
                    "dimension": str(item["dimension"])[:60],
                    "weight": float(item.get("weight") or 1.0),
                    "levels": {str(k): str(v)[:400] for k, v in (item.get("levels") or {}).items()},
                }
            )
    if not rubric:
        rubric = default_rubric()

    project = Project(
        slug=f"gen-{digest[:10]}",
        title=title,
        description=description,
        target_level=str(raw.get("target_level") or resolved_level)[:40],
        domain=str(raw.get("domain") or "general")[:60],
        skill_codes=[s for s in (raw.get("skills") or []) if s in graph.nodes][:8] or weak[:6],
        stack=[str(s)[:40] for s in (raw.get("stack") or [])][:12],
        deliverables=[str(d)[:400] for d in (raw.get("deliverables") or [])][:10],
        milestones=milestones,
        rubric=rubric,
        guidance=str(raw.get("guidance") or "")[:3000],
        dataset_hint=str(raw.get("dataset_hint") or "")[:1500],
        difficulty=min(5, 1 + max(0, int(user.level_index or 0)) // 2),
        est_hours=float(raw.get("est_hours") or 20.0),
        generated_for_user_id=user.id,
        generated_by="ai",
        content_hash=digest,
    )
    session.add(project)
    session.flush()
    session.add(
        ProjectAttempt(
            enrollment_id=_ensure_note_enrollment(session, user, project),
            kind=ProjectAttempt.KIND_AI_SUGGESTION,
            author="ai",
            content=f"Generated project brief for level {resolved_level}: {title}",
            ai_suggested=description[:4000],
        )
    )
    session.commit()
    return {"created": [project_payload(session, project, user=user)], "milestones": milestones, "message": "Project generated and stored."}


def _ensure_note_enrollment(session: Session, user: User, project: Project) -> int:
    """Generated projects are recorded against a (possibly not yet started) enrollment."""
    enrollment = session.execute(
        select(ProjectEnrollment).where(ProjectEnrollment.user_id == user.id, ProjectEnrollment.project_id == project.id)
    ).scalars().first()
    if enrollment is None:
        enrollment = ProjectEnrollment(user_id=user.id, project_id=project.id, status="paused")
        session.add(enrollment)
        session.flush()
    return int(enrollment.id)


def enroll(session: Session, user: User, project_id: int, *, milestones: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    project = session.get(Project, int(project_id))
    if project is None:
        raise ValueError("project not found")
    enrollment = session.execute(
        select(ProjectEnrollment).where(ProjectEnrollment.user_id == user.id, ProjectEnrollment.project_id == project.id)
    ).scalars().first()
    if enrollment is None:
        enrollment = ProjectEnrollment(user_id=user.id, project_id=project.id, status="active")
        session.add(enrollment)
        session.flush()
        seeds = milestones or _milestones_from_project(project)
        for i, item in enumerate(seeds):
            session.add(
                ProjectTask(
                    enrollment_id=enrollment.id,
                    title=str(item.get("title") or f"Milestone {i + 1}")[:300],
                    description=str(item.get("description") or "")[:4000],
                    acceptance_criteria=[str(c)[:400] for c in (item.get("acceptance_criteria") or [])][:8],
                    order_index=i,
                )
            )
        session.flush()
    enrollment.status = "active"
    session.commit()
    return enrollment_payload(session, enrollment) or {}


def _milestones_from_project(project: Project) -> list[dict[str, Any]]:
    if project.milestones:
        return [dict(m) for m in project.milestones if isinstance(m, dict)]
    if project.deliverables:
        return [
            {
                "title": f"Deliverable {i + 1}: {str(d)[:120]}",
                "description": str(d),
                "acceptance_criteria": ["exists in the repository", "documented in README", "testable or demonstrable"],
            }
            for i, d in enumerate(project.deliverables)
        ]
    return [{"title": "Plan & data", "description": "Scope, data source, protocol", "acceptance_criteria": ["written plan"]}]


def set_milestone_status(session: Session, user: User, task_id: int, status: str, *, notes: str = "") -> dict[str, Any] | None:
    task = session.get(ProjectTask, int(task_id))
    if task is None:
        return None
    enrollment = session.get(ProjectEnrollment, task.enrollment_id)
    if enrollment is None or enrollment.user_id != user.id:
        return None
    if status not in {"todo", "doing", "review", "done"}:
        raise ValueError("invalid status")
    task.status = status
    if notes:
        task.review_notes = notes[:4000]
    task.completed_at = datetime.utcnow() if status == "done" else None
    session.flush()
    return enrollment_payload(session, enrollment)


def mentor_reply(session: Session, user: User, message: str, *, enrollment_id: int | None = None) -> dict[str, Any]:
    """Mentor turn with the 'never do the project for them' policy, applied to AI *and* offline paths."""
    from app.ai.manager import manager as ai_manager

    enrollment = None
    if enrollment_id:
        enrollment = session.get(ProjectEnrollment, int(enrollment_id))
    if enrollment is None:
        enrollment = session.execute(
            select(ProjectEnrollment).where(ProjectEnrollment.user_id == user.id).order_by(ProjectEnrollment.id.desc()).limit(1)
        ).scalars().first()
    if enrollment is None:
        return {"text": "Start a project first (Projects → Start). Then I can mentor you against its milestones.", "engine": "local"}

    project = enrollment.project
    progress = _progress_text(session, enrollment)
    graph = SkillGraph(session)
    retrieval = Retriever(session)
    results = retrieval.search(f"{project.title} {message}", top_k=4)
    context, citations = build_context(results, max_chars=5000)

    demands_code = bool(DO_IT_FOR_ME.search(message))
    engine = "local"
    body = ""
    usage: dict[str, Any] = {}

    if ai_manager.is_configured() and user.ai_enabled:
        result = ai_manager.invoke(
            "mentor",
            user_id=user.id,
            kwargs={
                "question": (("[POLICY: learner asked for a full implementation - refuse politely, give architecture/plan/snippets only]\n" if demands_code else "") + message)[:4000],
                "project": {"title": project.title, "target_level": project.target_level, "rubric": project.rubric or [], "deliverables": project.deliverables or []},
                "progress": progress,
                "context": context,
            },
            session=session,
        )
        if result.ok:
            body, engine, usage = result.text.strip(), "ai", result.usage.__dict__
        else:
            body = _local_mentor(session, user, enrollment, message, demands_code=demands_code, citations=citations, graph=graph, note=result.user_message())
    else:
        body = _local_mentor(session, user, enrollment, message, demands_code=demands_code, citations=citations, graph=graph, note="Offline mentor: structure and acceptance criteria from your project + library (no AI provider configured).")

    session.add(
        ProjectAttempt(
            enrollment_id=enrollment.id,
            kind=ProjectAttempt.KIND_MESSAGE,
            author="user",
            content=message[:8000],
        )
    )
    session.add(
        ProjectAttempt(
            enrollment_id=enrollment.id,
            kind=ProjectAttempt.KIND_AI_SUGGESTION if engine == "ai" else ProjectAttempt.KIND_MESSAGE,
            author="ai",
            content=body[:12000],
            ai_suggested=(body if demands_code else "")[:6000],
            evaluation={"engine": engine, "refused_full_code": demands_code},
        )
    )
    session.commit()
    return {
        "text": body,
        "engine": engine,
        "enrollment_id": enrollment.id,
        "refused_full_code": demands_code,
        "sources": [c for c in citations[:4]],
        "usage": usage,
    }


def _local_mentor(
    session: Session,
    user: User,
    enrollment: ProjectEnrollment,
    message: str,
    *,
    demands_code: bool,
    citations: list[dict[str, Any]],
    graph: SkillGraph,
    note: str = "",
) -> str:
    project = enrollment.project
    tasks = [t for t in (enrollment.tasks or []) if t.status != "done"]
    nxt = tasks[0] if tasks else None
    lines: list[str] = []
    if demands_code:
        lines.append(
            f"I will not write `{project.title}` for you — the portfolio value and the skill gain would both be zero. "
            "What I can do: give you the structure, the interfaces, and the risk list, and review what you produce."
        )
        lines.append("")
    lines.append(f"### Suggested structure for “{project.title}”")
    lines.append("```text")
    lines.extend(_skeleton(project))
    lines.append("```")
    if nxt is not None:
        lines.append("")
        lines.append(f"### Next milestone: {nxt.title}")
        lines.append(nxt.description[:600])
        if nxt.acceptance_criteria:
            lines.append("\nDone when:")
            lines.extend(f"- [ ] {c}" for c in nxt.acceptance_criteria[:6])
    if project.rubric:
        lines.append("")
        lines.append("### How this will be graded")
        for item in project.rubric[:9]:
            lines.append(f"- **{item.get('dimension')}** (weight {item.get('weight', 1)})")
    if citations:
        lines.append("")
        lines.append("### From your library")
        for i, cite in enumerate(citations[:3], start=1):
            lines.append(f"- [{i}] {cite.get('label', cite.get('document', ''))}")
    lines.append("")
    lines.append("What you should implement now: " + (nxt.title if nxt else "the first deliverable in the list") + ".")
    if note:
        lines.append("")
        lines.append(f"> {note}")
    return "\n".join(lines)


def _skeleton(project: Project) -> list[str]:
    stack = ", ".join((project.stack or [])[:4]) or "python, fastapi"
    if "llm" in (project.domain or "").lower() or "rag" in project.slug:
        return [
            "project/",
            "  ingest/      # parse -> chunk -> metadata -> index",
            "  retrieve/    # lexical + vector search, fusion, rerank",
            "  generate/    # prompt assembly, citation validation, refusal",
            "  eval/        # golden set, metrics, regression runner",
            "  serve/       # FastAPI app, schemas, caching, metrics",
            "  tests/       # unit + contract + eval regression",
            f"stack: {stack}",
        ]
    if (project.target_level or "") in {"senior", "strong_middle"}:
        return [
            "repo/",
            "  config/          # pydantic settings, one source of truth",
            "  data/            # loading, contracts, point-in-time splits",
            "  features/        # deterministic transforms, versioned",
            "  models/          # train + score, no I/O in the model layer",
            "  service/         # API, batching, timeouts, fallback",
            "  observability/   # metrics, drift, run manifests",
            "  pipelines/       # idempotent training/scoring DAGs",
            "  tests/           # unit, contract, golden-metric, load",
            f"stack: {stack}",
        ]
    return [
        "repo/",
        "  src/data.py        # load + validate + split (no leakage)",
        "  src/features.py    # transforms inside a pipeline",
        "  src/train.py       # CV, config, logged metrics",
        "  src/evaluate.py    # metrics + threshold under constraints",
        "  tests/             # shapes, dtypes, golden metric",
        f"stack: {stack}",
    ]


def _progress_text(session: Session, enrollment: ProjectEnrollment) -> str:
    lines = []
    for task in enrollment.tasks or []:
        lines.append(f"- [{task.status}] {task.title}: {(task.description or '')[:180]}")
    log = []
    for attempt in list(enrollment.attempts or [])[-6:]:
        who = "learner" if attempt.author == "user" else "ai"
        kind = "submission" if attempt.kind == ProjectAttempt.KIND_SUBMISSION else "chat"
        log.append(f"  ({who}/{kind}) {(attempt.content or '')[:220]}")
    return "\n".join(lines + log)[:4000] or "no milestones yet"


def record_submission(
    session: Session,
    user: User,
    *,
    enrollment_id: int,
    task_id: int | None,
    content: str,
    user_implemented: str = "",
    files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    enrollment = session.get(ProjectEnrollment, int(enrollment_id))
    if enrollment is None or enrollment.user_id != user.id:
        raise ValueError("enrollment not found")
    attempt = ProjectAttempt(
        enrollment_id=enrollment.id,
        project_task_id=int(task_id) if task_id else None,
        kind=ProjectAttempt.KIND_SUBMISSION,
        author="user",
        content=content[:20000],
        user_implemented=(user_implemented or content)[:12000],
        ai_suggested="",
        files=files or [],
        meta={"submitted_at": datetime.utcnow().isoformat()},
    )
    session.add(attempt)
    if task_id:
        task = session.get(ProjectTask, int(task_id))
        if task is not None and task.status != "done":
            task.status = "review"
    session.commit()
    return {"submission_id": attempt.id, "enrollment": enrollment_payload(session, enrollment) or {}}


def evaluate(session: Session, user: User, enrollment_id: int, *, allow_ai: bool = True) -> dict[str, Any]:
    """
    Rubric evaluation. With AI: model reads the submission and scores each dimension with
    evidence. Without AI: transparent checklist scoring (milestones, acceptance criteria,
    linked coding tests) - explicitly labelled as such, never presented as a code review.
    """
    enrollment = session.get(ProjectEnrollment, int(enrollment_id))
    if enrollment is None or enrollment.user_id != user.id:
        raise ValueError("enrollment not found")
    project = enrollment.project
    rubric = project.rubric or default_rubric()
    submissions = [a for a in (enrollment.attempts or []) if a.kind == ProjectAttempt.KIND_SUBMISSION]
    transcript = "\n\n".join(
        f"[{a.kind}/{a.author}] {(a.content or '')[:1500]}" for a in (enrollment.attempts or [])[-14:]
    )

    from app.ai.manager import manager as ai_manager

    if allow_ai and ai_manager.is_configured() and user.ai_enabled and submissions:
        result = ai_manager.invoke(
            "generate",
            user_id=user.id,
            kwargs={
                "system": "You grade ML project submissions against a rubric. Quote evidence for every score. Respond with JSON only.",
                "prompt": f"""Grade this submission.

PROJECT: {project.title} ({project.target_level})
DELIVERABLES: {json.dumps(project.deliverables or [], ensure_ascii=False)}
RUBRIC: {json.dumps(rubric, ensure_ascii=False)[:3000]}
ACCEPTANCE CRITERIA STATE: {_progress_text(session, enrollment)[:2500]}
SUBMISSIONS/LOG:
{transcript[:9000]}

Output JSON:
{{"dimensions": [{{"dimension": "", "score": 0-100, "evidence": "", "next_action": ""}}],
 "summary": "3-6 sentences", "strengths": [""], "gaps": [""], "grade": "pass|borderline|redo",
 "interview_questions": ["3 questions a reviewer would ask about this"]}}""",
                "json_mode": True,
                "max_tokens": 1400,
            },
            session=session,
        )
        if result.ok and isinstance(result.data, dict):
            data = result.data
            dims = data.get("dimensions") or []
            scores = {str(d.get("dimension")): float(d.get("score") or 0) for d in dims if isinstance(d, dict)}
            weights = {str(d.get("dimension")): float(d.get("weight") or 1.0) for d in rubric}
            total = _weighted(scores, weights)
            payload = {
                "engine": "ai",
                "total": total,
                "grade": data.get("grade") or ("pass" if total >= 70 else "borderline"),
                "dimensions": dims,
                "summary": data.get("summary") or "",
                "strengths": data.get("strengths") or [],
                "gaps": data.get("gaps") or [],
                "interview_questions": data.get("interview_questions") or [],
                "submissions": len(submissions),
                "rubric": rubric,
            }
            _after_evaluation(session, user, enrollment, payload)
            return payload

    # deterministic checklist scoring
    tasks = list(enrollment.tasks or [])
    done = sum(1 for t in tasks if t.status == "done")
    review = sum(1 for t in tasks if t.status == "review")
    total_tasks = max(1, len(tasks))
    completion = done / total_tasks
    words = sum(len((a.content or "").split()) for a in submissions)
    dims: list[dict[str, Any]] = []
    for item in rubric:
        name = str(item.get("dimension"))
        base = 100.0 * completion
        if name in {"Testing", "Documentation"}:
            base = 100.0 * (completion * 0.7 + min(1.0, len(submissions) / 3) * 0.3)
        if name in {"Deployment", "Monitoring", "Scalability"} and not submissions:
            base *= 0.4
        score = round(min(100.0, max(0.0, base + (5.0 if review else 0.0))), 1)
        dims.append(
            {
                "dimension": name,
                "score": score,
                "evidence": f"checklist: {done}/{total_tasks} milestones done, {len(submissions)} submission(s), {words} words recorded",
                "next_action": "mark the milestone done after your acceptance criteria are met, or submit evidence"
                if score < 70
                else "keep the evidence trail (tests, dashboards, docs) linked in the submission",
            }
        )
    payload = {
        "engine": "local",
        "note": "Checklist-based evaluation (no AI used): derived from milestone states and submitted evidence. Configure AI for a qualitative code review.",
        "total": _weighted({d["dimension"]: d["score"] for d in dims}, {str(d.get("dimension")): float(d.get("weight") or 1.0) for d in rubric}),
        "grade": "in-progress" if completion < 1 else "pass",
        "dimensions": dims,
        "summary": f"{done}/{total_tasks} milestones complete with {len(submissions)} submission entries.",
        "strengths": [t.title for t in tasks if t.status == "done"][:4],
        "gaps": [t.title for t in tasks if t.status != "done"][:4],
        "interview_questions": [
            "Why this architecture and not the alternative you rejected?",
            "What breaks first at 10x traffic, and how do you know?",
            "How would you detect that this model quietly got worse?",
        ],
        "submissions": len(submissions),
        "rubric": rubric,
    }
    _after_evaluation(session, user, enrollment, payload)
    return payload


def _after_evaluation(session: Session, user: User, enrollment: ProjectEnrollment, payload: dict[str, Any]) -> None:
    project = enrollment.project
    session.add(
        ProjectAttempt(
            enrollment_id=enrollment.id,
            kind=ProjectAttempt.KIND_EVALUATION,
            author="ai" if payload.get("engine") == "ai" else "system",
            content=payload.get("summary", "")[:4000],
            evaluation={k: v for k, v in payload.items() if k in {"total", "grade", "dimensions", "strengths", "gaps", "engine"}},
        )
    )
    if float(payload.get("total") or 0) >= 60:
        enrollment.status = "evaluated"
        enrollment.submitted_at = enrollment.submitted_at or datetime.utcnow()
    from app.services.user_knowledge import update_from_activity

    update_from_activity(
        session,
        user_id=user.id,
        skill_codes=(project.skill_codes or [])[:6],
        activity="project",
        score=float(payload.get("total") or 0),
        correct=float(payload.get("total") or 0) >= 70,
        dimensions={
            "engineering_score": float(payload.get("total") or 0),
            "problem_solving_score": float(payload.get("total") or 0) * 0.9,
            "coding_score": float(payload.get("total") or 0) * 0.8,
        },
        difficulty=int(project.difficulty or 3),
        xp=15 * max(1, int(project.difficulty or 2)),
    )
    from app.models import LearningMemory

    key = f"project:{project.slug}"
    value = f"{project.title}: {payload.get('grade')} ({round(float(payload.get('total') or 0))}/100)"
    row = session.execute(
        select(LearningMemory).where(LearningMemory.user_id == user.id, LearningMemory.category == "project", LearningMemory.key == key)
    ).scalar_one_or_none()
    if row is None:
        session.add(
            LearningMemory(
                user_id=user.id,
                category="project",
                key=key,
                value=value,
                payload={"dimensions": payload.get("dimensions", []), "engine": payload.get("engine"), "at": datetime.utcnow().isoformat()},
            )
        )
    else:
        row.value = value
        row.payload = {"dimensions": payload.get("dimensions", []), "engine": payload.get("engine"), "at": datetime.utcnow().isoformat()}
        row.weight = round(float(row.weight or 1.0) + 0.1, 2)
        row.updated_at = datetime.utcnow()
    session.commit()


def _weighted(scores: dict[str, float], weights: dict[str, float]) -> float:
    if not scores:
        return 0.0
    num = sum(float(scores.get(k, 0.0)) * float(weights.get(k, 1.0)) for k in scores)
    den = sum(float(weights.get(k, 1.0)) for k in scores) or 1.0
    return round(num / den, 1)


def default_rubric() -> list[dict[str, Any]]:
    return [
        {"dimension": name, "weight": 1.0, "levels": {"1": "missing", "3": "present and working", "5": "exemplary and explained"}}
        for name in RUBRIC_DIMENSIONS
    ]


def active_enrollment(session: Session, user: User) -> dict[str, Any] | None:
    enrollment = session.execute(
        select(ProjectEnrollment)
        .where(ProjectEnrollment.user_id == user.id, ProjectEnrollment.status.in_(["active", "paused", "submitted"]))
        .order_by(ProjectEnrollment.id.desc())
        .limit(1)
    ).scalars().first()
    return enrollment_payload(session, enrollment, include_log=True) if enrollment else None


def project_minutes(session: Session, enrollment_id: int, minutes: float) -> None:
    enrollment = session.get(ProjectEnrollment, int(enrollment_id))
    if enrollment is None:
        return
    enrollment.minutes_spent = round(float(enrollment.minutes_spent or 0.0) + max(0.0, float(minutes)), 1)
    session.commit()

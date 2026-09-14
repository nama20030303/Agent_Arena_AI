"""
Coding Lab (§20, §21).

Grading is *execution*, not opinion: the sandbox runs hidden unittest tests against the
learner's code. AI can author new tasks, but a generated task is only persisted when its
own reference solution passes its own tests inside the sandbox - so the lab cannot fill up
with broken, ungradeable exercises.
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.knowledge.textutil import content_hash
from app.models import CodingAttempt, CodingTask, LearningMemory, StudySession, User
from app.services.graph import SkillGraph
from app.services.sandbox import run_python

DIFFICULTY_LABELS = {1: "beginner", 2: "easy", 3: "medium", 4: "hard", 5: "senior"}


def task_payload(task: CodingTask, *, user_id: int | None = None, session: Session | None = None, with_solution: bool = False) -> dict[str, Any]:
    data = {
        "id": task.id,
        "slug": task.slug,
        "title": task.title,
        "description": task.description,
        "language": task.language,
        "difficulty": task.difficulty,
        "difficulty_label": DIFFICULTY_LABELS.get(int(task.difficulty or 2), "easy"),
        "level": task.level,
        "libraries": task.libraries or [],
        "skills": task.skill_codes or [],
        "topic": task.topic_code,
        "from_scratch": bool(task.from_scratch),
        "banned_imports": task.banned_imports or [],
        "starter_code": task.starter_code or "",
        "hints": (task.hints or []) if with_solution else [],
        "hint_count": len(task.hints or []),
        "est_minutes": task.est_minutes,
        "xp": task.xp,
        "tests_visible": bool(task.test_code),
        "test_count": _count_tests(task.test_code),
        "created_at": task.created_at.isoformat() if task.created_at else None,
    }
    if with_solution:
        data.update({"solution": task.solution, "solution_explanation": task.solution_explanation})
    if session is not None and user_id is not None:
        stats = session.execute(
            select(
                func.count(CodingAttempt.id),
                func.max(CodingAttempt.passed),
                func.max(CodingAttempt.created_at),
            ).where(CodingAttempt.user_id == user_id, CodingAttempt.task_id == task.id)
        ).one()
        data["attempts"] = int(stats[0] or 0)
        data["solved"] = bool(stats[1])
        data["last_attempt"] = stats[2].isoformat() if stats[2] else None
    return data


def _count_tests(test_code: str) -> int:
    """Number of test functions, computed once from the AST (shown in the task list)."""
    if not test_code:
        return 0
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return 0
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name.startswith("test"))


def list_tasks(
    session: Session,
    user: User,
    *,
    topic_code: str = "",
    skill: str = "",
    level: str = "",
    only_unsolved: bool = False,
    limit: int = 60,
    offset: int = 0,
) -> dict[str, Any]:
    stmt = select(CodingTask)
    if topic_code:
        stmt = stmt.where(CodingTask.topic_code == topic_code)
    if level:
        stmt = stmt.where(CodingTask.level == level)
    total = int(session.execute(select(func.count()).select_from(stmt.subquery())).scalar_one() or 0)
    rows = list(session.execute(stmt.order_by(CodingTask.difficulty.asc(), CodingTask.id.asc()).limit(500)).scalars())

    solved_ids = set(
        session.execute(
            select(CodingAttempt.task_id).where(CodingAttempt.user_id == user.id, CodingAttempt.passed.is_(True))
        ).scalars()
    )
    graph = SkillGraph(session)
    wanted: set[str] = set()
    if skill:
        wanted |= {skill}
    if topic_code:
        wanted |= set(graph.topic_skills.get(topic_code, []))

    def key(task: CodingTask) -> tuple[Any, ...]:
        solved_penalty = 1 if task.id in solved_ids else 0
        relevance = -2.0 * len(set(task.skill_codes or []) & wanted) if wanted else 0.0
        level_penalty = 0.0 if not level or task.level == level else 0.5
        return (solved_penalty, relevance + level_penalty, task.difficulty or 2, task.id)

    rows.sort(key=key)
    if only_unsolved:
        rows = [t for t in rows if t.id not in solved_ids]
    page = rows[offset : offset + limit]
    return {
        "total": total,
        "solved": len(solved_ids),
        "items": [task_payload(t, user_id=user.id, session=session) for t in page],
    }


def get_task(session: Session, user: User, ident: str | int) -> CodingTask | None:
    if isinstance(ident, int) or str(ident).isdigit():
        task = session.get(CodingTask, int(ident))
        if task is not None and task.id == int(ident):
            return task
    return session.execute(select(CodingTask).where(CodingTask.slug == str(ident))).scalar_one_or_none()


def run(session: Session, user: User, task: CodingTask, code: str, *, persist: bool = False, hints_used: int = 0) -> dict[str, Any]:
    """Execute the learner's code against the task's tests in the isolated sandbox."""
    if not task.test_code:
        result = run_python(code, "import unittest\n\nclass T(unittest.TestCase):\n    def test_smoke(self):\n        self.assertTrue(True)\n")
        payload = result.to_dict()
        payload["mode"] = "run-only"
        payload["note"] = "This task has no hidden tests yet - execution output only."
        return payload
    result = run_python(code, task.test_code, banned_imports=task.banned_imports or [])
    payload = result.to_dict()
    payload["mode"] = "tests"
    payload["passed_all"] = bool(result.ok)
    payload["solution_unlocked"] = bool(result.ok)
    if persist:
        _persist_attempt(session, user, task, code, result, hints_used=hints_used)
    return payload


def _persist_attempt(session: Session, user: User, task: CodingTask, code: str, result: Any, *, hints_used: int = 0) -> CodingAttempt:
    attempt = CodingAttempt(
        user_id=user.id,
        task_id=task.id,
        code=code[:60000],
        passed=bool(result.ok),
        total_tests=int(result.total_tests),
        passed_tests=int(result.passed_tests),
        test_results=[t if isinstance(t, dict) else asdict(t) for t in result.tests],
        stdout=(result.stdout or "")[:6000],
        error=(result.error or "")[:4000],
        runtime_ms=int(result.runtime_ms),
        hints_used=int(hints_used or 0),
        meta={"timed_out": bool(getattr(result, "timed_out", False)), "violated": getattr(result, "violated", [])},
    )
    session.add(attempt)
    task.run_count = (task.run_count or 0) + 1
    if result.ok:
        task.solve_count = (task.solve_count or 0) + 1
    session.flush()
    return attempt


def submit(session: Session, user: User, *, task_id: int, code: str, hints_used: int = 0, seconds: float = 0.0, session_id: int | None = None) -> dict[str, Any]:
    task = session.get(CodingTask, int(task_id))
    if task is None:
        raise ValueError(f"Coding task {task_id} not found")
    result = run_python(code, task.test_code or "import unittest\n\nclass T(unittest.TestCase):\n    def test_none(self):\n        pass\n", banned_imports=task.banned_imports or [])
    attempt = _persist_attempt(session, user, task, code, result, hints_used=hints_used)

    from app.services.user_knowledge import update_from_activity

    first_pass = bool(result.ok) and int(session.execute(
        select(func.count(CodingAttempt.id)).where(CodingAttempt.user_id == user.id, CodingAttempt.task_id == task.id, CodingAttempt.passed.is_(True))
    ).scalar_one()) <= 1

    updates = update_from_activity(
        session,
        user_id=user.id,
        skill_codes=task.skill_codes or [],
        activity="code",
        score=100.0 if result.ok else round(100.0 * result.passed_tests / max(1, result.total_tests)),
        correct=bool(result.ok),
        dimensions={
            "coding_score": 100.0 if result.ok else round(70.0 * result.passed_tests / max(1, result.total_tests)),
            "engineering_score": min(100.0, 40.0 + 20.0 * int(hints_used == 0) + (30.0 if result.ok else 0.0)),
            "problem_solving_score": 100.0 if (result.ok and first_pass) else 60.0,
        },
        difficulty=task.difficulty,
        xp=int(task.xp or 20) if result.ok else 2,
        minutes=max(0.5, (seconds or 0) / 60.0),
        error_type="" if result.ok else "code_bug",
    )

    from app.services.learning_engine import on_attempt_recorded

    loop = on_attempt_recorded(
        session, user, skill_codes=task.skill_codes or [], correct=bool(result.ok), score=float(result.passed_tests / max(1, result.total_tests) * 100), error_type="" if result.ok else "code_bug"
    )

    if session_id:
        study = session.get(StudySession, session_id)
        if study is not None:
            study.items_done = int(study.items_done or 0) + 1

    memory_note = ""
    if not result.ok and result.total_tests == 0 and result.error:
        memory_note = "Sandbox problem, not your code."
        session.add(
            LearningMemory(
                user_id=user.id,
                category="history",
                key=f"sandbox:{task.slug}:{datetime.utcnow().strftime('%H%M%S')}",
                value=f"execution error: {result.error[:200]}",
            )
        )

    session.commit()
    payload = result.to_dict()
    payload["test_results"] = payload.get("tests", [])
    payload.update(
        {
            "attempt_id": attempt.id,
            "task_id": task.id,
            "solved": bool(result.ok),
            "solution_unlocked": bool(result.ok),
            "skill_updates": [asdict(u) for u in updates],
            "learning_engine": loop,
            "note": memory_note,
        }
    )
    return payload


def reveal_solution(session: Session, user: User, task: CodingTask, *, force: bool = False) -> dict[str, Any]:
    solved = bool(
        session.execute(
            select(CodingAttempt.id).where(CodingAttempt.user_id == user.id, CodingAttempt.task_id == task.id, CodingAttempt.passed.is_(True)).limit(1)
        ).scalar()
    )
    attempts = int(
        session.execute(select(func.count(CodingAttempt.id)).where(CodingAttempt.user_id == user.id, CodingAttempt.task_id == task.id)).scalar_one() or 0
    )
    if not solved and not force and attempts < 3:
        return {
            "available": False,
            "reason": "Solve it (or run 3 attempts) to unlock the reference solution. Hints are unlimited.",
            "attempts": attempts,
            "solved": solved,
        }
    return {
        "available": True,
        "solution": task.solution,
        "explanation": task.solution_explanation,
        "tests": task.test_code,
        "attempts": attempts,
        "solved": solved,
    }


def next_task(session: Session, user: User, *, topic_code: str = "") -> dict[str, Any] | None:
    payload = list_tasks(session, user, topic_code=topic_code, only_unsolved=True, limit=1)
    if not payload["items"]:
        return None
    return payload["items"][0]


# --------------------------------------------------------------------------- #
#  Dynamic task generation (validated before it is stored)
# --------------------------------------------------------------------------- #
def generate_tasks(session: Session, user: User, *, topic_code: str, count: int = 2, allow_ai: bool = True) -> dict[str, Any]:
    """
    AI-authored coding tasks for a topic. Validation pipeline:
      1. parse the code (syntax)          -> reject junk
      2. run the reference solution vs the generated tests in the sandbox
                                          -> keep only tasks that actually pass
      3. run an empty solution            -> reject trivially-passing tests
    Rejected tasks are reported, never silently dropped, so the failure is visible.
    """
    graph = SkillGraph(session)
    label = graph.topics[topic_code].name if topic_code in graph.topics else topic_code.replace("_", " ")
    created: list[CodingTask] = []
    rejected: list[dict[str, Any]] = []

    if not allow_ai:
        return {"created": [], "rejected": [{"reason": "ai-disabled"}], "message": "Generation requires an AI provider."}

    from app.ai.manager import manager as ai_manager
    from app.knowledge.retrieval import Retriever, build_context

    if not ai_manager.is_configured():
        return {"created": [], "rejected": [{"reason": "no-ai"}], "message": "No AI provider configured - the offline engine cannot author and validate unit tests for new tasks. Existing tasks and the seeded bank still work."}

    results = Retriever(session).search(f"{label} {topic_code}", top_k=4)
    context, _citations = build_context(results, max_chars=6000)
    ai = ai_manager.invoke(
        "generate_practice",
        user_id=user.id,
        kwargs={
            "topic": f"{label} (CODING TASKS ONLY: implement from scratch with NumPy, no ML libraries)",
            "level": "intermediate",
            "context": context,
            "kind": "code",
            "learner_state": json.dumps({"level_index": user.level_index, "skills": graph.topic_skills.get(topic_code, [])[:6]}, ensure_ascii=False),
        },
        session=session,
    )
    if not ai.ok or not isinstance(ai.data, dict):
        return {"created": [], "rejected": [{"reason": "ai-error", "detail": (ai.error or ai.user_message())[:300]}], "message": "Generation failed; nothing was stored."}

    for raw in (ai.data or {}).get("tasks", [])[: count * 2]:
        spec = {
            "slug": f"gen-{content_hash(str(raw.get('title', '')) + str(raw.get('statement', ''))[:200])[:10]}",
            "title": str(raw.get("title") or "")[:200],
            "description": str(raw.get("statement") or raw.get("description") or "")[:6000],
            "starter_code": str(raw.get("starter_code") or "")[:6000],
            "test_code": str(raw.get("test_code") or raw.get("tests") or "")[:12000],
            "solution": str(raw.get("solution") or "")[:12000],
            "hints": [str(h)[:400] for h in (raw.get("hints") or [])][:4],
            "difficulty": max(1, min(5, int(raw.get("difficulty") or 3))),
        }
        if not spec["title"] or not spec["description"]:
            rejected.append({"title": spec["title"] or "(untitled)", "reason": "incomplete spec"})
            continue
        if not spec["test_code"] or not spec["solution"]:
            rejected.append({"title": spec["title"], "reason": "AI provided no tests or no reference solution"})
            continue
        for field in ("test_code", "solution", "starter_code"):
            try:
                ast.parse(spec[field] or "")
            except SyntaxError as exc:
                rejected.append({"title": spec["title"], "reason": f"{field} is not valid python: {exc}"})
                break
        else:
            ok_ref = run_python(spec["solution"], spec["test_code"], timeout=max(8, settings.code_execution_timeout_seconds - 4))
            if not ok_ref.ok:
                rejected.append(
                    {
                        "title": spec["title"],
                        "reason": "reference solution fails its own tests (task not stored)",
                        "detail": (ok_ref.error or f"{ok_ref.passed_tests}/{ok_ref.total_tests} tests passed")[:300],
                    }
                )
                continue
            empty = run_python(spec["starter_code"] or "raise NotImplementedError\n", spec["test_code"])
            if empty.ok:
                rejected.append({"title": spec["title"], "reason": "tests pass on the empty starter - useless tests"})
                continue
            task = CodingTask(
                slug=spec["slug"],
                title=spec["title"],
                description=spec["description"],
                starter_code=spec["starter_code"],
                test_code=spec["test_code"],
                solution=spec["solution"],
                solution_explanation=str(raw.get("explanation") or "")[:4000],
                hints=spec["hints"],
                difficulty=spec["difficulty"],
                level="intermediate",
                libraries=["numpy"],
                skill_codes=list(raw.get("skills") or graph.topic_skills.get(topic_code, []))[:5],
                topic_code=topic_code,
                from_scratch=True,
                xp=15 * spec["difficulty"],
                est_minutes=25,
            )
            session.add(task)
            created.append(task)

    if created:
        session.commit()
    return {
        "created": [task_payload(t) for t in created],
        "rejected": rejected,
        "message": f"{len(created)} task(s) validated and stored; {len(rejected)} rejected by validation.",
    }


def static_check(code: str, *, banned: list[str] | None = None) -> dict[str, Any]:
    """Fast pre-run feedback shown in the editor (syntax + blocked imports)."""
    from app.services.sandbox import screen_imports

    issues: list[dict[str, Any]] = []
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return {"ok": False, "syntax_error": f"{exc.msg} (line {exc.lineno})", "issues": []}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = node.names[0].name if isinstance(node, ast.Import) else (node.module or "")
            root = name.split(".")[0]
            if root in {"sys", "os", "open"} or root in (banned or []):
                issues.append({"line": node.lineno, "severity": "warning", "message": f"'{root}' is available but rarely allowed in exercise sandboxes; avoid it."})
        if isinstance(node, ast.Name) and node.id == "NotImplementedError" and isinstance(getattr(node, "ctx", None), ast.Del):
            issues.append({"line": node.lineno, "severity": "info", "message": "unfinished stub"})
    return {"ok": True, "syntax_error": "", "issues": issues, "imports_blocked": screen_imports(code, extra_banned=banned)}

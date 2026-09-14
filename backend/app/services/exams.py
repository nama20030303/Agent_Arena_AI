"""
Exams (§28).

Three gates - Junior, Middle, Senior. The Senior assessment is deliberately weighted to
system design, trade-offs, scalability, monitoring, cost and production incidents.

Objective items are graded deterministically. Open/architecture items use AI rubric
grading when available; without AI they are graded by the local criteria engine and the
report marks that clearly, so a self-administered exam can never pretend to be a proctored one.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    Exam,
    ExamAttempt,
    Question,
    StudySession,
    User,
)
from app.services.graph import SkillGraph
from app.services.questions import question_payload

EXAM_SKILL_BARS = {
    "junior": {"min_level": 1, "max_level": 4, "count": 12},
    "middle": {"min_level": 3, "max_level": 7, "count": 14},
    "senior": {"min_level": 7, "max_level": 10, "count": 10},
}


def list_exams(session: Session) -> list[dict[str, Any]]:
    rows = session.execute(select(Exam).order_by(Exam.min_level.asc())).scalars()
    return [
        {
            "id": e.id,
            "code": e.code,
            "title": e.title,
            "description": e.description,
            "target_level": e.target_level,
            "min_level": e.min_level,
            "duration_minutes": e.duration_minutes,
            "passing_score": e.passing_score,
            "sections": e.sections or [],
        }
        for e in rows
    ]


def exam_readiness(session: Session, user: User) -> list[dict[str, Any]]:
    graph = SkillGraph(session)
    scores = graph.user_scores(user.id)
    out: list[dict[str, Any]] = []
    for exam in session.execute(select(Exam).order_by(Exam.min_level.asc())).scalars():
        cfg = EXAM_SKILL_BARS.get(exam.target_level, {"min_level": exam.min_level, "max_level": 10, "count": 12})
        required = [
            code
            for code, node in graph.nodes.items()
            if cfg["min_level"] <= node.level <= cfg["max_level"]
        ]
        if not required:
            required = list(graph.nodes)
        bar = {"junior": 60.0, "middle": 70.0, "senior": 78.0}.get(exam.target_level, 68.0)
        covered = [c for c in required if scores.get(c, (0.0, 0.0))[0] >= bar]
        weak = sorted(required, key=lambda c: scores.get(c, (0.0, 0.0))[0])[:5]
        readiness = round(100.0 * len(covered) / max(1, len(required)), 1)
        advice = ""
        if readiness < 60:
            advice = "not ready - close these first: " + ", ".join(c.replace("_", " ") for c in weak[:3])
        elif readiness < 85:
            advice = "attempt it; expect to lose points on: " + ", ".join(c.replace("_", " ") for c in weak[:3])
        else:
            advice = "skills look ready - take it as a calibration check"
        out.append(
            {
                "exam_id": exam.id,
                "exam": exam.title,
                "code": exam.code,
                "readiness": readiness,
                "covered": len(covered),
                "total": len(required),
                "bar": bar,
                "weak": weak,
                "advice": advice,
            }
        )
    return out


def _section_pool(session: Session, section: dict[str, Any], *, used: set[int]) -> list[Question]:
    levels = [int(x) for x in (section.get("levels") or [])] or list(range(0, 11))
    types = [t for t in (section.get("types") or [])]
    stmt = select(Question).where(Question.level.in_(levels), Question.approved.is_(True))
    if types:
        stmt = stmt.where(Question.question_type.in_(types))
    rows = [q for q in session.execute(stmt.order_by(Question.difficulty.asc(), Question.id.asc())).scalars() if q.id not in used]
    return rows


def start(session: Session, user: User, exam_code: str) -> dict[str, Any]:
    exam = session.execute(select(Exam).where(Exam.code == exam_code)).scalar_one_or_none()
    if exam is None:
        raise ValueError(f"unknown exam: {exam_code}")
    running = session.execute(
        select(ExamAttempt).where(ExamAttempt.user_id == user.id, ExamAttempt.exam_id == exam.id, ExamAttempt.state == "in_progress")
        .order_by(ExamAttempt.id.desc())
    ).scalars().first()
    if running is not None:
        return payload_for(session, exam, running)

    used: set[int] = set()
    items: list[dict[str, Any]] = []
    missing: list[str] = []
    pool_by_section: dict[str, list[Question]] = {}
    for section in exam.sections or []:
        pool = _section_pool(session, section, used=used)
        pool_by_section[section["name"]] = pool
    for section in exam.sections or []:
        want = int(section.get("count") or max(3, (exam.duration_minutes // max(1, len(exam.sections or [1]))) // 4))
        pool = pool_by_section.get(section["name"], [])
        # spread difficulty: take alternating easy/hard
        picked: list[Question] = []
        if pool:
            step = max(1, len(pool) // max(1, want))
            for i in range(0, len(pool), step):
                if len(picked) >= want:
                    break
                picked.append(pool[i])
            for q in pool:
                if len(picked) >= want:
                    break
                if q not in picked:
                    picked.append(q)
        if len(picked) < want:
            missing.append(f"{section['name']}: {want - len(picked)} item(s) not available in the bank")
        for q in picked:
            used.add(q.id)
            items.append(
                {
                    "id": q.id,
                    "code": q.public_code,
                    "section": section["name"],
                    "weight": float(section.get("weight") or 1.0),
                    "question": question_payload(q),
                }
            )
    attempt = ExamAttempt(
        user_id=user.id,
        exam_id=exam.id,
        state="in_progress",
        items=items,
        answers={},
        started_at=datetime.utcnow(),
    )
    session.add(attempt)
    session.add(StudySession(user_id=user.id, activity="exam", item_ref=exam.code))
    session.commit()
    return payload_for(session, exam, attempt, missing=missing)


def payload_for(session: Session, exam: Exam, attempt: ExamAttempt, *, missing: list[str] | None = None) -> dict[str, Any]:
    return {
        "attempt_id": attempt.id,
        "exam": {"id": exam.id, "code": exam.code, "title": exam.title, "duration_minutes": exam.duration_minutes, "passing_score": exam.passing_score, "sections": exam.sections or []},
        "state": attempt.state,
        "items": attempt.items or [],
        "answers": attempt.answers or {},
        "started_at": attempt.started_at.isoformat() if attempt.started_at else None,
        "submitted_at": attempt.submitted_at.isoformat() if attempt.submitted_at else None,
        "elapsed_seconds": int(((attempt.submitted_at or datetime.utcnow()) - (attempt.started_at or datetime.utcnow())).total_seconds()),
        "item_count": len(attempt.items or []),
        "missing": missing or [],
        "report": _report_of(attempt),
    }


def _report_of(attempt: ExamAttempt) -> dict[str, Any]:
    if attempt.state not in {"submitted", "graded"}:
        return {}
    return {
        "score": attempt.score,
        "passed": attempt.passed,
        "verdict": attempt.verdict,
        "section_scores": attempt.section_scores or {},
        "item_results": attempt.item_results or [],
        "strengths": attempt.strengths or [],
        "gaps": attempt.gaps or [],
        "feedback": attempt.feedback,
        "duration_seconds": attempt.duration_seconds,
    }


def save_answers(session: Session, user: User, attempt_id: int, answers: dict[str, Any]) -> dict[str, Any]:
    attempt = session.get(ExamAttempt, int(attempt_id))
    if attempt is None or attempt.user_id != user.id:
        raise ValueError("attempt not found")
    if attempt.state != "in_progress":
        return payload_for(session, session.get(Exam, attempt.exam_id), attempt)
    merged = dict(attempt.answers or {})
    merged.update({str(k): v for k, v in (answers or {}).items()})
    attempt.answers = merged
    session.commit()
    return payload_for(session, session.get(Exam, attempt.exam_id), attempt)


def submit(session: Session, user: User, attempt_id: int, *, allow_ai: bool = True) -> dict[str, Any]:
    attempt = session.get(ExamAttempt, int(attempt_id))
    if attempt is None or attempt.user_id != user.id:
        raise ValueError("attempt not found")
    exam = session.get(Exam, attempt.exam_id)
    if exam is None:
        raise ValueError("exam missing")
    if attempt.state != "in_progress":
        return payload_for(session, exam, attempt)

    from app.services.questions import grade_locally

    ai_available = False
    if allow_ai:
        from app.ai.manager import manager as ai_manager

        ai_available = ai_manager.is_configured() and bool(user.ai_enabled)

    answers = attempt.answers or {}
    item_results: list[dict[str, Any]] = []
    section_scores: dict[str, list[float]] = {}
    open_items_ai: list[dict[str, Any]] = []

    for item in attempt.items or []:
        question = session.get(Question, int(item["id"]))
        if question is None:
            continue
        raw = answers.get(str(item["id"])) or answers.get(str(item.get("code"))) or {}
        if isinstance(raw, str):
            raw = {"answer": raw}
        answer = str(raw.get("answer") or "")
        selected = raw.get("selected_option")
        verdict = grade_locally(question=question, answer=answer, selected_option=selected)
        result = {
            "id": item["id"],
            "code": item.get("code"),
            "section": item.get("section"),
            "weight": float(item.get("weight") or 1.0),
            "type": question.question_type,
            "stem": question.stem[:400],
            "answered": bool(answer.strip()) or selected is not None,
            "correct": bool(verdict["correct"]),
            "local_score": float(verdict["score"]),
            "error_type": verdict.get("error_type", ""),
            "feedback": verdict.get("feedback", "")[:600],
            "missing_points": verdict.get("missing_points") or [],
            "skills": question.skill_codes or [],
            "graded_by": "local",
            "final_score": float(verdict["score"]),
        }
        needs_ai = (
            question.question_type in {"open", "architecture", "interview", "conceptual"}
            and answer.strip()
            and ai_available
        )
        if needs_ai:
            open_items_ai.append((item, question, answer, result))
        section_scores.setdefault(str(item.get("section") or "general"), []).append(result)
        item_results.append(result)

    if open_items_ai:
        _ai_grade_items(session, user, exam, open_items_ai)

    weights = {name: float((s or {}).get("weight") or 1.0) for name, s in zip(section_scores, exam.sections or [])}
    per_section: dict[str, float] = {}
    for name, results in section_scores.items():
        per_section[name] = round(sum(r["final_score"] for r in results) / max(1, len(results)), 1)
    answered_total = sum(1 for r in item_results if r["answered"])
    weighted_sum = sum(per_section.get(name, 0.0) * w for name, w in weights.items())
    weight_total = sum(weights.values()) or 1.0
    coverage_penalty = answered_total / max(1, len(item_results)) if item_results else 0.0
    score = round((weighted_sum / weight_total) * (0.6 + 0.4 * coverage_penalty), 1)

    passed = score >= float(exam.passing_score or 70.0)
    verdict = ("pass" if passed else "not yet") + (" · strong" if score >= 85 else (" · borderline" if score >= (exam.passing_score or 70) - 5 else ""))
    strengths = sorted((name for name, v in per_section.items() if v >= 75), key=lambda n: -per_section[n])[:4]
    gaps: list[str] = []
    for result in item_results:
        if not result["correct"]:
            for skill in result["skills"][:2]:
                if skill not in gaps:
                    gaps.append(skill)
    gaps = gaps[:8]

    attempt.item_results = item_results
    attempt.section_scores = {name: {"score": value, "weight": weights.get(name, 1.0), "items": len(section_scores.get(name, []))} for name, value in per_section.items()}
    attempt.score = score
    attempt.passed = passed
    attempt.verdict = verdict
    attempt.strengths = strengths
    attempt.gaps = gaps
    attempt.feedback = _narrative(exam, score, per_section, strengths, gaps, ai_available, len(item_results), answered_total)
    attempt.state = "graded"
    attempt.submitted_at = datetime.utcnow()
    attempt.duration_seconds = int((attempt.submitted_at - (attempt.started_at or attempt.submitted_at)).total_seconds())

    _apply_results(session, user, exam, item_results, score=score, passed=passed)

    study = session.execute(
        select(StudySession).where(StudySession.user_id == user.id, StudySession.activity == "exam", StudySession.item_ref == exam.code)
        .order_by(StudySession.id.desc())
    ).scalars().first()
    if study is not None:
        study.ended_at = datetime.utcnow()
        study.active_seconds = attempt.duration_seconds
        study.items_done = len(item_results)
        study.xp = int(score)
        study.ai_used = ai_available
    session.commit()
    return payload_for(session, exam, attempt)


def _ai_grade_items(session: Session, user: User, exam: Exam, entries: list[tuple[dict[str, Any], Question, str, dict[str, Any]]]) -> None:
    """One AI call per open item (cached by content), clamped to the daily budget."""
    from app.ai.manager import manager as ai_manager

    for _item, question, answer, result in entries:
        if ai_manager.budget_state(user.id)["limit_reached"]:
            result["ai_skipped"] = "daily AI limit reached"
            return
        graded = ai_manager.invoke(
            "evaluate_answer",
            user_id=user.id,
            kwargs={"question": {"stem": question.stem, "type": question.question_type, "difficulty": question.difficulty, "expected_points": question.expected_points or [], "expected_answer": question.expected_answer}, "answer": answer[:4000]},
            session=session,
        )
        if graded.ok and isinstance(graded.data, dict):
            data = graded.data
            result["final_score"] = float(data.get("score") or result["local_score"])
            result["correct"] = bool(data.get("correct", result["correct"]))
            result["feedback"] = (data.get("feedback") or result["feedback"])[:800]
            result["missing_points"] = data.get("missing_points") or result["missing_points"]
            result["error_type"] = data.get("error_type") or result["error_type"]
            result["graded_by"] = "ai"


def _narrative(exam: Exam, score: float, per_section: dict[str, float], strengths: list[str], gaps: list[str], ai_used: bool, total: int, answered: int) -> str:
    lines = [
        f"{exam.title}: {round(score)}/100 ({'pass' if score >= (exam.passing_score or 70) else 'below the bar of ' + str(exam.passing_score)}).",
        "",
        "Section scores: " + ", ".join(f"{name} {round(v)}" for name, v in per_section.items()),
    ]
    if strengths:
        lines.append("Strongest areas: " + ", ".join(strengths) + ".")
    if gaps:
        lines.append("Gaps to close: " + ", ".join(g.replace("_", " ") for g in gaps[:6]) + ".")
    if answered < total:
        lines.append(f"{total - answered} of {total} items were left unanswered - unanswered items are scored 0, which is the cheapest points to lose.")
    lines.append(
        "Open/architecture items were graded against the rubric by AI."
        if ai_used
        else "Open/architecture items were graded by the deterministic criteria engine (no AI configured): scores are conservative and keyword-based."
    )
    return "\n".join(lines)


def _apply_results(session: Session, user: User, exam: Exam, item_results: list[dict[str, Any]], *, score: float, passed: bool) -> None:
    from app.services.user_knowledge import update_from_activity

    for result in item_results:
        if not result["answered"]:
            continue
        update_from_activity(
            session,
            user_id=user.id,
            skill_codes=result.get("skills") or [],
            activity="exam",
            score=float(result["final_score"]),
            correct=bool(result["correct"]),
            difficulty=3 if exam.target_level == "junior" else (4 if exam.target_level == "middle" else 5),
            xp=8 if result["correct"] else 2,
            error_type=str(result.get("error_type") or ""),
        )
        result["skills_updated"] = True

    from app.models import LearningMemory

    session.add(
        LearningMemory(
            user_id=user.id,
            category="exam",
            key=f"{exam.code}:{datetime.utcnow().strftime('%Y%m%d%H%M')}",
            value=f"{round(score)}/100 - {exam.title}",
            payload={"score": score, "passed": passed, "gaps": [r.get("skills") for r in item_results if not r["correct"]][:6]},
        )
    )
    session.flush()


def history(session: Session, user: User) -> list[dict[str, Any]]:
    rows = session.execute(
        select(ExamAttempt, Exam).join(Exam, Exam.id == ExamAttempt.exam_id).where(ExamAttempt.user_id == user.id).order_by(ExamAttempt.id.desc()).limit(20)
    ).all()
    return [
        {
            "attempt_id": a.id,
            "exam": e.title,
            "code": e.code,
            "score": a.score,
            "passed": a.passed,
            "verdict": a.verdict,
            "state": a.state,
            "section_scores": a.section_scores or {},
            "started": a.started_at.isoformat() if a.started_at else None,
            "submitted": a.submitted_at.isoformat() if a.submitted_at else None,
            "duration_seconds": a.duration_seconds,
        }
        for a, e in rows
    ]

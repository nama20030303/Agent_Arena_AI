"""
Mock interview (§29).

Junior / Middle / Senior tracks. Each turn is graded, the follow-up is chosen from where
the answer was weakest, and the final report scores six axes with evidence. Without AI the
loop still runs (bank questions + deterministic scoring of the candidate's text); the report
states that it is a local, keyword-based evaluation.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import InterviewSession, LearningMemory, Question, StudySession, User
from app.services.graph import SkillGraph
from app.services.questions import grade_locally

AXES = (
    "technical_knowledge",
    "depth",
    "reasoning",
    "communication",
    "system_design",
    "production_thinking",
)

TRACKS = {
    "junior": {"levels": [0, 1, 2, 3], "turns": 6, "types": ["conceptual", "mcq", "open", "math"]},
    "middle": {"levels": [3, 4, 5, 6, 7], "turns": 8, "types": ["conceptual", "open", "math", "code", "debug"]},
    "senior": {"levels": [6, 7, 8, 9, 10], "turns": 8, "types": ["architecture", "open", "debug", "interview"]},
}

AXIS_HINTS = {
    "system_design": ("scal", "design", "architect", "shard", "partition", "throughput", "latency", "capacity"),
    "production_thinking": ("monitor", "drift", "alert", "rollback", "degrad", "sla", "slo", "incident", "cost", "latency p", "fall"),
    "depth": ("because", "derivative", "variance", "complexity", "proof", "gradient", "why"),
    "communication": ("first", "then", "however", "trade-off", "in short", "structure"),
}


def next_bank_question(session: Session, *, level: str = "middle", used: list[int] | None = None) -> Question | None:
    cfg = TRACKS.get(level, TRACKS["middle"])
    used_set = set(used or [])
    rows = session.execute(
        select(Question)
        .where(Question.level.in_(cfg["levels"]), Question.approved.is_(True))
        .order_by(func.random())
        .limit(60)
    ).scalars()
    for question in rows:
        if question.id in used_set:
            continue
        if question.question_type in cfg["types"]:
            return question
    for question in rows:
        if question.id not in used_set:
            return question
    return None


def start(session: Session, user: User, *, level: str = "middle", focus: str = "") -> dict[str, Any]:
    level = level if level in TRACKS else "middle"
    cfg = TRACKS[level]
    record = InterviewSession(user_id=user.id, role_level=level, focus=focus or "general", state="active", turns=[], questions_asked=[], scores={}, report={})
    session.add(record)
    session.flush()

    opening: dict[str, Any] = {"content": "", "question_id": None}
    from app.ai.manager import manager as ai_manager

    if ai_manager.is_configured() and user.ai_enabled:
        result = ai_manager.invoke(
            "interview",
            user_id=user.id,
            kwargs={
                "transcript": [],
                "stage": "question",
                "level": level,
                "context": f"Focus area: {focus or 'general ML engineering'}. Ask the opening technical question now.",
            },
            session=session,
            use_cache=False,
        )
        if result.ok:
            opening = {"content": _clean_question(result.text), "question_id": None}
    if not opening["content"]:
        question = next_bank_question(session, level=level)
        if question is None:
            raise ValueError("no questions available for the interview - seed the bank first")
        record.questions_asked = [question.id]
        session.flush()
        opening = {"content": question.stem, "question_id": question.id}

    record.turns = [{"role": "interviewer", "content": opening["content"], "question_id": opening["question_id"], "at": datetime.utcnow().isoformat()}]
    session.add(StudySession(user_id=user.id, activity="interview", item_ref=f"interview:{record.id}"))
    session.commit()
    return snapshot(session, user, record.id)


def _clean_question(text: str) -> str:
    text = (text or "").strip()
    for marker in ("NEXT:", "Next:"):
        if marker in text:
            text = text.split(marker, 1)[1].strip()
    return text.lstrip("#-* ").strip()[:2500]


def answer(session: Session, user: User, interview_id: int, text: str, *, seconds: float = 0.0) -> dict[str, Any]:
    record = session.get(InterviewSession, int(interview_id))
    if record is None or record.user_id != user.id:
        raise ValueError("interview not found")
    if record.state != "active":
        raise ValueError("interview already finished")

    turns = list(record.turns or [])
    last_question = next((t for t in reversed(turns) if t.get("role") == "interviewer"), {})
    qid = last_question.get("question_id")
    question = session.get(Question, int(qid)) if qid else None

    local = grade_locally(question=question or {"stem": last_question.get("content", ""), "type": "open", "difficulty": 3, "options": [], "correct_option": None, "expected_answer": "", "expected_value": "", "tolerance": 0.0, "expected_points": [], "explanation": ""}, answer=text)
    turn = {
        "role": "candidate",
        "content": text[:8000],
        "question_id": qid,
        "local_score": float(local["score"]),
        "correct": bool(local["correct"]),
        "dimensions": local["dimensions"],
        "error_type": local.get("error_type", ""),
        "seconds": float(seconds or 0),
        "at": datetime.utcnow().isoformat(),
    }
    engine = "local"
    ai_feedback = ""
    from app.ai.manager import manager as ai_manager

    if ai_manager.is_configured() and user.ai_enabled:
        result = ai_manager.invoke(
            "interview",
            user_id=user.id,
            kwargs={
                "transcript": turns + [{"role": "candidate", "content": text}],
                "stage": "turn",
                "level": record.role_level,
                "candidate_answer": text[:2000],
            },
            session=session,
            use_cache=False,
        )
        if result.ok:
            engine = "ai"
            ai_feedback = result.text.strip()
            turn["ai_response"] = ai_feedback[:6000]

    turns.append(turn)
    record.turns = turns
    session.flush()

    cfg = TRACKS.get(record.role_level, TRACKS["middle"])
    candidate_turns = sum(1 for t in turns if t.get("role") == "candidate")
    if candidate_turns >= cfg["turns"]:
        session.commit()
        return {"snapshot": snapshot(session, user, record.id), "asked": False, "finished_suggestion": True, "engine": engine, "feedback": ai_feedback}

    # next question: probe the weakest part of the last answer, else a new bank question
    next_content = ""
    next_qid: int | None = None
    if engine == "ai" and ai_feedback:
        for marker in ("NEXT:", "Next:"):
            if marker in ai_feedback:
                next_content = ai_feedback.split(marker, 1)[1].strip()[:2000]
                break
    if not next_content:
        nxt = _probe_locally(session, record, text, local)
        next_content = nxt["content"]
        next_qid = nxt.get("question_id")
    turns.append({"role": "interviewer", "content": next_content, "question_id": next_qid, "at": datetime.utcnow().isoformat()})
    record.turns = turns
    if next_qid:
        asked = list(record.questions_asked or [])
        if int(next_qid) not in asked:
            asked.append(int(next_qid))
        record.questions_asked = asked
    session.commit()
    return {"snapshot": snapshot(session, user, record.id), "asked": True, "question": next_content, "feedback": ai_feedback, "engine": engine, "finished_suggestion": False}


def _probe_locally(session: Session, record: InterviewSession, text: str, local: dict[str, Any]) -> dict[str, Any]:
    """Follow-up chosen from the shape of the answer: too short, skipped points, or a new question."""
    words = len((text or "").split())
    missing = local.get("missing_points") or []
    if words < 25:
        return {"content": "That was very short. Take it further: state the mechanism, then give one concrete example with numbers."}
    if missing:
        return {
            "content": "You covered the first part. Now the part you skipped - "
            + "; ".join(str(m)[:120] for m in missing[:2])
            + ". Why does it matter in production?"
        }
    if local.get("error_type") in {"math", "code_bug"}:
        return {"content": "Walk me through the computation again, line by line, and tell me which step is most likely to be wrong."}
    nxt = next_bank_question(session, level=record.role_level, used=list(record.questions_asked or []))
    if nxt is not None:
        return {"content": nxt.stem[:2000], "question_id": nxt.id}
    return {"content": "Give me a production story: where has this failure mode bitten you, what did you measure, and what would you do differently now?"}


def finish(session: Session, user: User, interview_id: int, *, allow_ai: bool = True) -> dict[str, Any]:
    record = session.get(InterviewSession, int(interview_id))
    if record is None or record.user_id != user.id:
        raise ValueError("interview not found")
    if record.state == "finished":
        return {"report": record.report, "scores": record.scores, "snapshot": snapshot(session, user, record.id)}

    turns = list(record.turns or [])
    candidate_turns = [t for t in turns if t.get("role") == "candidate"]
    if not candidate_turns:
        raise ValueError("no answers to evaluate")

    local_scores = _local_scores(record, candidate_turns)
    scores = dict(local_scores)
    report_text = ""
    engine = "local"
    recommendation = "practice_more"

    from app.ai.manager import manager as ai_manager

    if allow_ai and ai_manager.is_configured() and user.ai_enabled:
        result = ai_manager.invoke(
            "interview",
            user_id=user.id,
            kwargs={"transcript": turns, "stage": "report", "level": record.role_level, "context": f"focus={record.focus}"},
            session=session,
            use_cache=False,
        )
        if result.ok and isinstance(result.data, dict):
            data = result.data
            engine = "ai"
            scores = {k: float(v) for k, v in (data.get("scores") or {}).items() if k in AXES}
            recommendation = str(data.get("recommendation") or "neutral")
            report_text = json.dumps(
                {
                    "strengths": data.get("strengths") or [],
                    "gaps": data.get("gaps") or [],
                    "follow_ups": data.get("follow_ups_for_next_round") or [],
                    "study_plan": data.get("study_plan") or [],
                },
                ensure_ascii=False,
            )
            scores.setdefault("overall", float(data.get("overall_score") or 0))

    overall = round(sum(v for k, v in scores.items() if k in AXES) / max(1, len([k for k in scores if k in AXES])), 1)
    scores["overall"] = scores.get("overall", overall) if engine == "ai" else overall
    report = {
        "engine": engine,
        "level": record.role_level,
        "focus": record.focus,
        "turns": len(candidate_turns),
        "recommendation": recommendation,
        "narrative": report_text[:4000] if report_text else _local_narrative(candidate_turns, scores, record.role_level),
        "per_turn": [
            {"question_id": t.get("question_id"), "score": t.get("final_score", t.get("local_score")), "error_type": t.get("error_type", ""), "words": len(str(t.get("content") or "").split())}
            for t in candidate_turns
        ],
    }
    record.report = report
    record.scores = scores
    record.state = "finished"
    record.finished_at = datetime.utcnow()

    _apply(session, user, record, scores, report)
    session.commit()
    return {"report": report, "scores": scores, "snapshot": snapshot(session, user, record.id)}


def _local_scores(record: InterviewSession, candidate_turns: list[dict[str, Any]]) -> dict[str, float]:
    n = max(1, len(candidate_turns))
    avg_local = sum(float(t.get("local_score") or 0) for t in candidate_turns) / n
    words = [len(str(t.get("content") or "").split()) for t in candidate_turns]
    avg_words = sum(words) / n
    depth = min(100.0, 30.0 + avg_words / 3.0)
    reasoning = 0.0
    production = 0.0
    system = 0.0
    for turn in candidate_turns:
        text = str(turn.get("content") or "").lower()
        reasoning += sum(1 for marker in AXIS_HINTS["depth"] if marker in text)
        production += sum(1 for marker in AXIS_HINTS["production_thinking"] if marker in text)
        system += sum(1 for marker in AXIS_HINTS["system_design"] if marker in text)
    reasoning_score = min(100.0, 25.0 + 6.0 * reasoning / n)
    production_score = min(100.0, 20.0 + 9.0 * production / n)
    system_score = min(100.0, 15.0 + 11.0 * system / n)
    communication = min(100.0, 35.0 + min(40.0, avg_words / 4.0) + (15.0 if all(w > 20 for w in words) else 0.0))
    errors = sum(1 for t in candidate_turns if t.get("error_type") and t["error_type"] != "none")
    return {
        "technical_knowledge": round(min(100.0, avg_local), 1),
        "depth": round(depth, 1),
        "reasoning": round(reasoning_score, 1),
        "communication": round(communication, 1),
        "system_design": round(system_score, 1),
        "production_thinking": round(production_score, 1),
        "_errors": float(errors),
    }


def _local_narrative(turns: list[dict[str, Any]], scores: dict[str, float], level: str) -> str:
    lines = [f"{level.capitalize()} track · {len(turns)} answered question(s). Local (no-AI) evaluation of your written answers.", ""]
    for axis in AXES:
        value = scores.get(axis, 0.0)
        comment = "solid" if value >= 75 else ("adequate" if value >= 55 else "needs work")
        lines.append(f"- **{axis.replace('_', ' ')}**: {round(value)}/100 - {comment}")
    lines.append("")
    lines.append("_This is not a substitute for a human interview: keyword-based scoring cannot detect a subtly wrong but well-worded answer._")
    return "\n".join(lines)


def _apply(session: Session, user: User, record: InterviewSession, scores: dict[str, float], report: dict[str, Any]) -> None:
    from app.services.user_knowledge import update_from_activity

    graph = SkillGraph(session)
    asked = list(record.questions_asked or [])
    skills: list[str] = []
    if asked:
        rows = session.execute(select(Question).where(Question.id.in_(asked))).scalars()
        for question in rows:
            skills.extend(question.skill_codes or [])
    overall = float(scores.get("overall") or 0)
    update_from_activity(
        session,
        user_id=user.id,
        skill_codes=list(dict.fromkeys(skills))[:8],
        activity="interview",
        score=overall,
        correct=overall >= 70,
        dimensions={
            "theory_score": scores.get("technical_knowledge", overall),
            "problem_solving_score": scores.get("reasoning", overall),
            "engineering_score": scores.get("production_thinking", overall),
        },
        difficulty=4 if record.role_level == "senior" else (3 if record.role_level == "middle" else 2),
        xp=int(overall / 2),
    )
    existing = session.execute(
        select(LearningMemory).where(
            LearningMemory.user_id == user.id, LearningMemory.category == "exam", LearningMemory.key == f"interview:{record.id}"
        )
    ).scalar_one_or_none()
    memory_kwargs = dict(
        category="exam",
        key=f"interview:{record.id}",
        value=f"{record.role_level} mock interview: {round(overall)}/100 ({report.get('recommendation', 'n/a')})",
        payload={"scores": {k: v for k, v in scores.items() if k in AXES}, "engine": report.get("engine")},
    )
    if existing is None:
        session.add(LearningMemory(user_id=user.id, **memory_kwargs))
    else:
        for field_name, field_value in memory_kwargs.items():
            setattr(existing, field_name, field_value)
        existing.updated_at = datetime.utcnow()
    study = session.execute(
        select(StudySession)
        .where(StudySession.user_id == user.id, StudySession.activity == "interview", StudySession.item_ref == f"interview:{record.id}")
        .order_by(StudySession.id.desc())
    ).scalars().first()
    if study is not None:
        study.ended_at = datetime.utcnow()
        study.active_seconds = int((study.ended_at - study.started_at).total_seconds())
        study.items_done = report.get("turns", 0)
        study.xp = int(overall / 2)
        study.ai_used = report.get("engine") == "ai"
    session.flush()


def snapshot(session: Session, user: User, interview_id: int) -> dict[str, Any]:
    record = session.get(InterviewSession, int(interview_id))
    if record is None or record.user_id != user.id:
        raise ValueError("interview not found")
    turns = record.turns or []
    return {
        "id": record.id,
        "level": record.role_level,
        "focus": record.focus,
        "state": record.state,
        "turns": turns,
        "asked_count": len(record.questions_asked or []),
        "scores": record.scores or {},
        "report": record.report or {},
        "started_at": record.started_at.isoformat() if record.started_at else None,
        "finished_at": record.finished_at.isoformat() if record.finished_at else None,
    }


def history(session: Session, user: User, limit: int = 20) -> list[dict[str, Any]]:
    rows = session.execute(
        select(InterviewSession).where(InterviewSession.user_id == user.id, InterviewSession.state == "finished").order_by(InterviewSession.id.desc()).limit(limit)
    ).scalars()
    return [
        {
            "id": r.id,
            "level": r.role_level,
            "focus": r.focus,
            "scores": r.scores or {},
            "recommendation": (r.report or {}).get("recommendation"),
            "engine": (r.report or {}).get("engine"),
            "finished": r.finished_at.isoformat() if r.finished_at else None,
        }
        for r in rows
    ]

"""
Progress (§26), velocity (§27), study sessions, notes/memory/journal (§32-34).

Progress is deliberately *not* a single percentage: an ML engineer who can derive a
gradient but cannot ship a service is not 50% ready. We report a skill vector.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    CodingAttempt,
    LearningJournal,
    LearningMemory,
    PracticeAttempt,
    QuestionAttempt,
    StudySession,
    UserNote,
    UserSkill,
    Skill,
    Topic,
    User,
)

DIMENSIONS = {
    "theory": "theory_score",
    "math": "math_score",
    "coding": "coding_score",
    "problem_solving": "problem_solving_score",
    "engineering": "engineering_score",
}

# weights for the aggregate: a Senior-level bar cares about more than theory
DIMENSION_WEIGHTS = {"theory": 0.18, "math": 0.16, "coding": 0.26, "problem_solving": 0.2, "engineering": 0.2}


def _level_weights(level: int) -> float:
    """Higher roadmap levels count more toward the headline score (1.0 at level 0 -> 2.0 at 10)."""
    return 1.0 + 0.1 * max(0, min(10, int(level or 0)))


def progress_summary(session: Session, user: User) -> dict[str, Any]:
    rows = session.execute(
        select(UserSkill, Skill, Topic)
        .join(Skill, Skill.id == UserSkill.skill_id)
        .outerjoin(Topic, Topic.id == Skill.topic_id)
        .where(UserSkill.user_id == user.id)
    ).all()

    dim_totals: dict[str, list[float]] = defaultdict(list)
    level_scores: dict[int, list[float]] = defaultdict(list)
    domain_scores: dict[str, list[float]] = defaultdict(list)
    topic_scores: dict[str, list[float]] = defaultdict(list)  # skill + topic code -> scores
    covered_levels: set[int] = set()
    touched = 0
    xp_total = 0
    minutes_total = 0.0
    strong, weak, needs_review = [], [], []
    now = datetime.utcnow()

    for us, skill, topic in rows:
        score = float(us.knowledge_score or 0.0)
        weight = _level_weights(skill.level)
        for name, column in DIMENSIONS.items():
            value = float(getattr(us, column) or 0.0)
            if value > 0:
                dim_totals[name].append(value)
        level_scores[int(skill.level or 0)].append(score)
        topic_scores[skill.code].append(score)
        if topic is not None:
            domain_scores[topic.domain or topic.name].append(score)
            topic_scores[topic.code].append(score)
        if (us.attempts or 0) > 0:
            touched += 1
            covered_levels.add(int(skill.level or 0))
        xp_total += int(us.xp or 0)
        minutes_total += float(us.minutes_studied or 0.0)
        if score >= 78 and float(us.confidence or 0) >= 0.4:
            strong.append((score, skill.code, skill.name, skill.level))
        if 0 < score < 62:
            weak.append((score, skill.code, skill.name, skill.level, us.success_rate or 0))
        if us.next_review is not None and us.next_review <= now:
            needs_review.append((us.next_review, skill.code, skill.name, score))

    dims = {name: round(sum(vals) / len(vals), 1) if vals else 0.0 for name, vals in ((n, dim_totals.get(n, [])) for n in DIMENSIONS)}
    weighted = sum(dims.get(name, 0.0) * w for name, w in DIMENSION_WEIGHTS.items())
    # coverage: share of the 11 levels with at least one assessed skill
    coverage = len(covered_levels) / 11.0
    overall = round(weighted * (0.55 + 0.45 * coverage), 1)
    level_avg = {str(lvl): round(sum(v) / len(v), 1) for lvl, v in sorted(level_scores.items())}

    strong.sort(reverse=True)
    weak.sort()
    needs_review.sort()

    return {
        "ml_engineer_score": overall,
        "dimensions": dims,
        "dimension_weights": DIMENSION_WEIGHTS,
        "weighted_dimension_score": round(weighted, 1),
        "level_coverage": round(coverage * 100, 1),
        "levels_reached": sorted(covered_levels),
        "by_level": level_avg,
        "by_domain": {k: round(sum(v) / len(v), 1) for k, v in sorted(domain_scores.items())},
        "skills_assessed": touched,
        "skills_total": len(rows),
        "xp": xp_total,
        "minutes_studied": round(minutes_total, 1),
        "strong_skills": [
            {"code": c, "name": n, "score": round(s, 1), "level": lvl} for s, c, n, lvl in strong[:10]
        ],
        "weak_skills": [
            {"code": c, "name": n, "score": round(s, 1), "level": lvl, "success_rate": sr} for s, c, n, lvl, sr in weak[:10]
        ],
        "needs_review": [
            {"code": c, "name": n, "score": round(s, 1), "due": ts.isoformat()} for ts, c, n, s in needs_review[:10]
        ],
        "current_level": {"index": user.level_index, "label": user.level_label},
    }


def roadmap(session: Session, user: User) -> list[dict[str, Any]]:
    rows = session.execute(
        select(Topic, func.count(Skill.id), func.coalesce(func.avg(UserSkill.knowledge_score), 0), func.coalesce(func.avg(UserSkill.confidence), 0))
        .outerjoin(Skill, Skill.topic_id == Topic.id)
        .outerjoin(UserSkill, (UserSkill.skill_id == Skill.id) & (UserSkill.user_id == user.id))
        .group_by(Topic.id)
        .order_by(Topic.level.asc(), Topic.order_index.asc())
    ).all()
    by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for topic, n_skills, avg, conf in rows:
        by_level[int(topic.level or 0)].append(
            {
                "code": topic.code,
                "name": topic.name,
                "domain": topic.domain,
                "summary": topic.summary,
                "skills": int(n_skills or 0),
                "score": round(float(avg or 0.0), 1),
                "confidence": round(float(conf or 0.0), 2),
                "est_hours": topic.est_hours,
                "prerequisites": topic.prerequisites or [],
                "keywords": (topic.keywords or [])[:8],
                "state": "mastered" if float(avg or 0) >= 80 else ("in-progress" if float(avg or 0) > 0 else "locked" if int(avg or 0) == 0 and topic.level > (user.level_index or 0) + 1 else "new"),
            }
        )
    from app.seed.curriculum import LEVELS

    out = []
    for level in LEVELS:
        topics = by_level.get(level["index"], [])
        scores = [t["score"] for t in topics if t["skills"]]
        out.append(
            {
                **level,
                "topics": topics,
                "completion": round(sum(min(1.0, s / 80.0) for s in scores) / len(scores), 3) if scores else 0.0,
                "avg_score": round(sum(scores) / len(scores), 1) if scores else 0.0,
                "topic_count": len(topics),
            }
        )
    return out


def velocity(session: Session, user: User, *, days: int = 28) -> dict[str, Any]:
    since = datetime.utcnow() - timedelta(days=days)
    per_day = defaultdict(lambda: {"questions": 0, "coding": 0, "minutes": 0, "xp": 0, "correct": 0})
    for created, correct in session.execute(
        select(QuestionAttempt.created_at, QuestionAttempt.is_correct).where(QuestionAttempt.user_id == user.id, QuestionAttempt.created_at >= since)
    ).all():
        bucket = per_day[created.date()]
        bucket["questions"] += 1
        bucket["correct"] += 1 if correct else 0
    for created, _ in session.execute(
        select(CodingAttempt.created_at, CodingAttempt.passed).where(CodingAttempt.user_id == user.id, CodingAttempt.created_at >= since)
    ).all():
        per_day[created.date()]["coding"] += 1
    for started, seconds, xp, activity in session.execute(
        select(StudySession.started_at, StudySession.active_seconds, StudySession.xp, StudySession.activity).where(
            StudySession.user_id == user.id, StudySession.started_at >= since
        )
    ).all():
        bucket = per_day[started.date()]
        bucket["minutes"] += (seconds or 0) / 60.0
        bucket["xp"] += xp or 0
    series = [
        {
            "date": d.isoformat(),
            "questions": v["questions"],
            "correct": v["correct"],
            "coding": v["coding"],
            "minutes": round(v["minutes"], 1),
            "xp": v["xp"],
        }
        for d, v in sorted(per_day.items())
    ]
    weeks = max(1, days / 7.0)
    projects_month = int(
        session.execute(
            select(func.count(StudySession.id)).where(
                StudySession.user_id == user.id, StudySession.activity == "project", StudySession.started_at >= datetime.utcnow() - timedelta(days=30)
            )
        ).scalar_one()
        or 0
    )
    improved = int(
        session.execute(
            select(func.count(UserSkill.id)).where(
                UserSkill.user_id == user.id, UserSkill.updated_at >= since, UserSkill.knowledge_score > 0
            )
        ).scalar_one()
        or 0
    )
    totals = {k: sum(item[k] for item in series) for k in ("questions", "correct", "coding", "minutes", "xp")}
    return {
        "window_days": days,
        "series": series,
        "questions_per_week": round(totals["questions"] / weeks, 1),
        "coding_tasks_per_week": round(totals["coding"] / weeks, 1),
        "projects_per_month": projects_month,
        "study_minutes_total": round(totals["minutes"], 1),
        "minutes_per_week": round(totals["minutes"] / weeks, 1),
        "xp_total": totals["xp"],
        "xp_per_week": round(totals["xp"] / weeks, 1),
        "skills_improved": improved,
        "answer_accuracy": round(100.0 * totals["correct"] / totals["questions"], 1) if totals["questions"] else None,
        "days_active": len([s for s in series if s["questions"] or s["coding"] or s["minutes"] > 2]),
    }


# --------------------------------------------------------------------------- #
#  Study sessions
# --------------------------------------------------------------------------- #
def start_session(session: Session, user: User, *, activity: str = "learn", topic_code: str = "", item_ref: str = "") -> StudySession:
    row = StudySession(user_id=user.id, activity=activity, topic_code=topic_code, item_ref=item_ref)
    session.add(row)
    session.flush()
    return row


def end_session(session: Session, user: User, session_id: int | None, *, xp: int = 0, items_done: int = 0) -> dict[str, Any] | None:
    if not session_id:
        return None
    row = session.get(StudySession, int(session_id))
    if row is None or row.user_id != user.id:
        return None
    row.ended_at = datetime.utcnow()
    row.active_seconds = int((row.ended_at - row.started_at).total_seconds())
    row.xp = int(row.xp or 0) + int(xp or 0)
    row.items_done = int(row.items_done or 0) + int(items_done or 0)
    session.flush()
    return {
        "id": row.id,
        "activity": row.activity,
        "seconds": row.active_seconds,
        "xp": row.xp,
        "items_done": row.items_done,
        "topic": row.topic_code,
    }


def streak(session: Session, user_id: int) -> dict[str, Any]:
    days = {
        d.date() if isinstance(d, datetime) else d
        for d in session.execute(
            select(StudySession.started_at).where(StudySession.user_id == user_id, StudySession.active_seconds > 60)
        ).scalars()
    }
    today = date.today()
    cursor = today if today in days else today - timedelta(days=1)
    current = 0
    while cursor in days:
        current += 1
        cursor -= timedelta(days=1)
    longest = 0
    run = 0
    prev: date | None = None
    for d in sorted(days):
        run = run + 1 if prev and (d - prev).days == 1 else 1
        longest = max(longest, run)
        prev = d
    return {"current": current, "longest": max(longest, current), "days_studied": len(days)}


# --------------------------------------------------------------------------- #
#  Notes / bookmarks / mistakes
# --------------------------------------------------------------------------- #
def note_payload(row: UserNote) -> dict[str, Any]:
    return {
        "id": row.id,
        "kind": row.kind,
        "title": row.title,
        "body": row.body,
        "topic": row.topic_code,
        "document_id": row.document_id,
        "chunk_id": row.chunk_id,
        "excerpt": row.excerpt,
        "citation": row.citation or {},
        "tags": row.tags or [],
        "pinned": row.pinned,
        "created": row.created_at.isoformat() if row.created_at else None,
        "updated": row.updated_at.isoformat() if row.updated_at else None,
    }


def list_notes(session: Session, user: User, *, kind: str = "", topic: str = "", q: str = "", limit: int = 100) -> list[dict[str, Any]]:
    stmt = select(UserNote).where(UserNote.user_id == user.id)
    if kind:
        stmt = stmt.where(UserNote.kind == kind)
    if topic:
        stmt = stmt.where(UserNote.topic_code == topic)
    if q:
        like = f"%{q}%"
        stmt = stmt.where((UserNote.body.ilike(like)) | (UserNote.title.ilike(like)))
    rows = session.execute(stmt.order_by(UserNote.pinned.desc(), UserNote.updated_at.desc()).limit(limit)).scalars()
    return [note_payload(r) for r in rows]


# --------------------------------------------------------------------------- #
#  Long-term learning memory
# --------------------------------------------------------------------------- #
def memory_bundle(session: Session, user_id: int) -> dict[str, Any]:
    rows = session.execute(select(LearningMemory).where(LearningMemory.user_id == user_id)).scalars()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.category].append({"key": row.key, "value": row.value, "payload": row.payload, "weight": row.weight, "at": row.updated_at.isoformat() if row.updated_at else None})
    for entries in grouped.values():
        entries.sort(key=lambda e: -(e.get("weight") or 0))
    return {
        "strengths": grouped.get("strength", [])[:12],
        "weaknesses": grouped.get("weakness", [])[:12],
        "preferences": grouped.get("preference", [])[:8],
        "topics": grouped.get("topic", [])[:20],
        "projects": grouped.get("project", [])[:10],
        "exams": grouped.get("exam", [])[:10],
        "mistakes": grouped.get("mistake", [])[:20],
        "history": grouped.get("history", [])[:20],
        "facts": grouped.get("fact", [])[:10],
        "goal": grouped.get("goal", [])[:5],
    }


def journal_payload(row: LearningJournal) -> dict[str, Any]:
    return {
        "id": row.id,
        "date": row.entry_date.isoformat() if row.entry_date else None,
        "content": row.content,
        "mood": row.mood,
        "minutes": row.minutes,
        "analysis": row.ai_analysis or {},
        "gaps": row.gaps_detected or [],
        "created": row.created_at.isoformat() if row.created_at else None,
    }

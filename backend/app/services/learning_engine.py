"""
Learning Engine (§10, §14, §51, §52).

Decides *what* to study, *what* to review, *how hard* it should be and *when* to move
on - by combining the skill graph, the user knowledge model and the review schedule.
Deterministic by default: an AI planner can only re-order/refine the plan the engine
already produced, so the loop keeps working with no API access.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    CodingAttempt,
    CodingTask,
    ProjectEnrollment,
    DailyPlan,
    DiagnosticSession,
    LearningMemory,
    Question,
    QuestionAttempt,
    ReviewSchedule,
    StudySession,
    Skill,
    Topic,
    User,
    UserSkill,
)
from app.services.graph import SkillGraph
from app.services.user_knowledge import apply_forgetting, ensure_skill_rows

WEAK_SCORE_CEILING = 62.0
STRONG_SCORE_FLOOR = 78.0
MASTERED_FLOOR = 88.0
READINESS_TO_ADVANCE = 52.0

PLAN_KINDS = ("review", "learn", "practice", "coding", "project", "reflection", "exam_prep")


@dataclass
class TopicCandidate:
    code: str
    name: str
    level: int
    domain: str
    readiness: float
    weakness: float
    priority: float
    reasons: list[str] = field(default_factory=list)
    unmet: list[dict[str, Any]] = field(default_factory=list)
    assessed: int = 0
    coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
#  Learner state (shared by every AI prompt and by the planner)
# --------------------------------------------------------------------------- #
def learner_state(session: Session, user: User, *, graph: SkillGraph | None = None) -> dict[str, Any]:
    graph = graph or SkillGraph(session)
    rows = session.execute(select(UserSkill).where(UserSkill.user_id == user.id)).scalars()
    scored: list[tuple[str, float, float, int, str]] = []
    for row in rows:
        skill = graph.nodes.get(_skill_code(session, row.skill_id))
        if skill is None:
            continue
        scored.append((skill.code, float(row.knowledge_score or 0), float(row.confidence or 0), skill.level, row.mastery_state or "new"))

    weak = sorted([s for s in scored if s[1] < WEAK_SCORE_CEILING], key=lambda s: (s[1], -s[2]))[:10]
    strong = sorted([s for s in scored if s[1] >= STRONG_SCORE_FLOOR], key=lambda s: -s[1])[:10]
    plan = session.execute(
        select(DailyPlan).where(DailyPlan.user_id == user.id).order_by(DailyPlan.plan_date.desc()).limit(1)
    ).scalar_one_or_none()
    current_topic = _current_topic(session, user, plan=plan, graph=graph)

    recent_mistakes = [
        {"question": (q.stem or "")[:160], "error": a.error_type}
        for a, q in session.execute(
            select(QuestionAttempt, Question)
            .join(Question, Question.id == QuestionAttempt.question_id)
            .where(QuestionAttempt.user_id == user.id, QuestionAttempt.is_correct.is_(False))
            .order_by(QuestionAttempt.id.desc())
            .limit(4)
        ).all()
    ]
    preferences = [
        m.value for m in session.execute(
            select(LearningMemory).where(LearningMemory.user_id == user.id, LearningMemory.category == "preference")
        ).scalars()
    ]
    missing = current_topic["missing_prerequisites"] if current_topic else []
    return {
        "user_id": user.id,
        "level_index": user.level_index,
        "level_label": user.level_label or "Absolute Beginner",
        "goal": user.goal or user.target_role,
        "daily_minutes": user.daily_minutes,
        "strengths": [s[0] for s in strong],
        "weaknesses": [s[0] for s in weak],
        "weak_detail": [{"code": s[0], "score": round(s[1], 1), "confidence": round(s[2], 2)} for s in weak],
        "strong_detail": [{"code": s[0], "score": round(s[1], 1)} for s in strong],
        "missing_prerequisites": [m["code"] for m in missing],
        "recent_mistakes": recent_mistakes,
        "preferences": preferences,
        "current_topic": current_topic["code"] if current_topic else "",
        "velocity": velocity_summary(session, user.id),
        "web_allowed": bool(user.web_search_allowed),
    }


_code_cache: dict[int, str] = {}


def _skill_code(session: Session, skill_id: int) -> str:
    if skill_id not in _code_cache:
        code = session.execute(select(Skill.code).where(Skill.id == skill_id)).scalar()
        _code_cache[skill_id] = code or ""
    return _code_cache[skill_id]


def invalidate_caches() -> None:
    _code_cache.clear()


def demonstrated_level(session: Session, user: User, *, graph: SkillGraph | None = None) -> int:
    """
    The highest roadmap level the learner has *demonstrated* (not attended).
    Used so a strong Python user is not pushed back through Python basics (§52).
    """
    graph = graph or SkillGraph(session)
    rows = session.execute(
        select(Skill.level, UserSkill.knowledge_score, UserSkill.confidence, UserSkill.success_rate)
        .join(Skill, Skill.id == UserSkill.skill_id)
        .where(UserSkill.user_id == user.id, UserSkill.attempts > 0)
    ).all()
    qualified = {int(level) for level, score, conf, success in rows if float(score or 0) >= 72 and float(conf or 0) >= 0.25 and float(success or 0) >= 60}
    if not qualified:
        return int(user.level_index or 0)
    highest = max(qualified)
    return max(int(user.level_index or 0), min(10, highest))


def weak_skills(session: Session, user_id: int, *, limit: int = 8) -> list[dict[str, Any]]:
    rows = session.execute(
        select(UserSkill, Skill, Topic)
        .join(Skill, Skill.id == UserSkill.skill_id)
        .outerjoin(Topic, Topic.id == Skill.topic_id)
        .where(UserSkill.user_id == int(user_id))
        .order_by(UserSkill.knowledge_score.asc())
    ).all()
    out = []
    for us, skill, topic in rows:
        if float(us.knowledge_score or 0) >= WEAK_SCORE_CEILING:
            continue
        out.append(
            {
                "code": skill.code,
                "name": skill.name,
                "score": round(float(us.knowledge_score or 0), 1),
                "confidence": round(float(us.confidence or 0), 2),
                "attempts": us.attempts,
                "success_rate": us.success_rate,
                "level": skill.level,
                "topic": topic.code if topic else "",
                "topic_name": topic.name if topic else "",
                "failure_count": us.failure_count,
                "last_error_type": us.last_error_type,
                "next_review": us.next_review.isoformat() if us.next_review else None,
            }
        )
        if len(out) >= limit:
            break
    return out


def strong_skills(session: Session, user_id: int, *, limit: int = 8) -> list[dict[str, Any]]:
    rows = session.execute(
        select(UserSkill, Skill)
        .join(Skill, Skill.id == UserSkill.skill_id)
        .where(UserSkill.user_id == user_id, UserSkill.knowledge_score >= STRONG_SCORE_FLOOR)
        .order_by(UserSkill.knowledge_score.desc())
        .limit(limit)
    ).all()
    return [
        {
            "code": skill.code,
            "name": skill.name,
            "score": round(float(us.knowledge_score or 0), 1),
            "confidence": round(float(us.confidence or 0), 2),
            "level": skill.level,
            "category": skill.category,
            "attempts": us.attempts,
        }
        for us, skill in rows
    ]


def recently_learned(session: Session, user_id: int, *, days: int = 7, limit: int = 8) -> list[dict[str, Any]]:
    since = datetime.utcnow() - timedelta(days=days)
    rows = session.execute(
        select(UserSkill, Skill)
        .join(Skill, Skill.id == UserSkill.skill_id)
        .where(UserSkill.user_id == user_id, UserSkill.last_review >= since, UserSkill.knowledge_score > 0)
        .order_by(UserSkill.updated_at.desc())
        .limit(limit)
    ).all()
    return [
        {"code": skill.code, "name": skill.name, "score": round(float(us.knowledge_score or 0), 1),
         "delta": round(float(us.knowledge_score or 0) - float(us.theory_score or 0) * 0.0, 1),
         "at": us.last_review.isoformat() if us.last_review else None}
        for us, skill in rows
    ]


# --------------------------------------------------------------------------- #
#  Topic selection
# --------------------------------------------------------------------------- #
def candidate_topics(session: Session, user: User, *, graph: SkillGraph | None = None, limit: int = 12) -> list[TopicCandidate]:
    graph = graph or SkillGraph(session)
    scores = graph.user_scores(user.id)
    frontier = demonstrated_level(session, user, graph=graph)
    topics = sorted(graph.topics.values(), key=lambda t: (t.level, t.order_index))
    candidates: list[TopicCandidate] = []
    skipped: list[TopicCandidate] = []
    for topic in topics:
        skill_codes = graph.topic_skills.get(topic.code, [])
        if not skill_codes:
            continue
        assessed = graph.topic_assessment(user.id, topic.code, scores=scores)
        avg = assessed["avg"]
        evidence = assessed["assessed"]
        coverage = assessed["coverage"]
        prior = min(58.0, 6.0 * frontier)
        readiness_info = graph.readiness(user.id, topic.code, scores=scores, prior_score=prior)
        readiness = readiness_info["readiness"]
        weakness = max(0.0, 100.0 - avg) if evidence else 0.0
        priority = 0.0
        reasons: list[str] = []
        # 1) proximity to the learner's *demonstrated* frontier (personalised order inside the roadmap)
        distance = abs(topic.level - frontier)
        proximity = 40.0 / (1.0 + distance)
        priority += proximity
        if proximity > 20:
            reasons.append(f"level {topic.level} is at your demonstrated level {frontier}")
        elif topic.level < frontier:
            priority -= 8.0 * min(4, frontier - topic.level)
            reasons.append(f"below your demonstrated level {frontier}")
        # 2) readiness from prerequisites
        if readiness < READINESS_TO_ADVANCE:
            priority -= 22.0 * (1.0 - readiness / READINESS_TO_ADVANCE)
            reasons.append("prerequisites not solid yet")
        else:
            priority += min(14.0, readiness / 8.0)
        # 3) demonstrated weakness in this topic's own skills
        if evidence and avg < WEAK_SCORE_CEILING:
            priority += (WEAK_SCORE_CEILING - avg) * 0.6
            reasons.append(f"you score {round(avg)}/100 here over {evidence} assessed skill(s)")
        # 4) demonstrated mastery => excluded from the queue (the roadmap still shows it as done)
        skip = False
        dims = graph.topic_dimension_scores(user.id, topic.code)
        peak = max(dims.values()) if dims else 0.0
        demonstrated_well = peak >= 78.0 and assessed["confidence"] >= 0.2 and (coverage >= 0.34 or readiness >= 85.0)
        if demonstrated_well or (evidence and avg >= 75.0 and coverage >= 0.5 and assessed["confidence"] >= 0.25):
            skip = True
            reasons.append(f"already demonstrated (best dimension {round(peak)}/100 over {evidence} skill(s)) - not re-taught")
        elif evidence and avg >= MASTERED_FLOOR and coverage >= 0.2:
            skip = True
            reasons.append("already demonstrated - skipped")
        elif not evidence and topic.level < frontier and readiness >= 45.0:
            skip = True
            reasons.append(f"below your demonstrated level {frontier} and prerequisites look fine")
        if skip:
            skipped.append(
                TopicCandidate(
                    code=topic.code, name=topic.name, level=topic.level, domain=topic.domain,
                    readiness=readiness, weakness=round(weakness, 1), priority=-999.0,
                    reasons=reasons, assessed=evidence, coverage=coverage,
                )
            )
            continue
        candidates.append(
            TopicCandidate(
                code=topic.code,
                name=topic.name,
                level=topic.level,
                domain=topic.domain,
                readiness=readiness,
                weakness=round(weakness, 1),
                priority=round(priority, 2),
                reasons=reasons or ["next topic on the roadmap"],
                unmet=readiness_info["unmet"][:4],
                assessed=evidence,
                coverage=coverage,
            )
        )
    candidates.sort(key=lambda c: (-c.priority, c.level))
    # keep the skipped ones visible (the UI shows "why is X not suggested"), but never suggested
    return candidates[:limit]


def skipped_topics(session: Session, user: User, *, limit: int = 20) -> list[dict[str, Any]]:
    """Topics the engine deliberately does not serve, with the reason (transparency over silence)."""
    graph = SkillGraph(session)
    out: list[dict[str, Any]] = []
    for topic in graph.topics.values():
        assessment = graph.topic_assessment(user.id, topic.code)
        dims = graph.topic_dimension_scores(user.id, topic.code)
        peak = max(dims.values()) if dims else 0.0
        if assessment["assessed"] and (peak >= 78 or (assessment["avg"] >= 75 and assessment["coverage"] >= 0.5)):
            out.append({"code": topic.code, "name": topic.name, "level": topic.level, "score": assessment["avg"], "peak": peak, "reason": "demonstrated"})
    out.sort(key=lambda d: (d["level"], d["code"]))
    return out[:limit]


def recommend_next(session: Session, user: User) -> dict[str, Any]:
    ensure_skill_rows(session, user.id)
    apply_forgetting(session, user.id)
    graph = SkillGraph(session)
    candidates = candidate_topics(session, user, graph=graph, limit=6)
    if not candidates:
        return {"topic": None, "reason": "no curriculum loaded - run the seed"}
    best = candidates[0]
    if best.readiness < READINESS_TO_ADVANCE and best.unmet:
        fix = best.unmet[0]
        repair_topic = graph.nodes.get(fix["code"])
        return {
            "topic": {
                "code": best.code,
                "name": best.name,
                "level": best.level,
                "readiness": best.readiness,
                "reasons": best.reasons,
            },
            "remediation": {
                "skill": fix["code"],
                "name": fix["name"],
                "score": fix["score"],
                "why": f"{best.name} needs {fix['name']} first (you score {fix['score']}/100 there).",
            },
            "candidates": [c.to_dict() for c in candidates[1:]],
            "mode_suggestion": "practice",
        }
    return {
        "topic": {"code": best.code, "name": best.name, "level": best.level, "readiness": best.readiness, "reasons": best.reasons},
        "skills": graph.topic_skills.get(best.code, []),
        "candidates": [c.to_dict() for c in candidates[1:]],
        "mode_suggestion": "explain" if best.readiness >= 70 else "practice",
    }


def _current_topic(session: Session, user: User, *, plan: DailyPlan | None, graph: SkillGraph) -> dict[str, Any] | None:
    if plan is not None:
        for item in plan.items or []:
            if item.get("kind") == "learn" and item.get("code"):
                topic = graph.topics.get(item["code"])
                if topic is not None:
                    info = graph.readiness(user.id, topic.code)
                    return {
                        "code": topic.code,
                        "name": topic.name,
                        "level": topic.level,
                        "missing_prerequisites": info["unmet"],
                    }
    rec = recommend_next(session, user)
    topic = rec.get("topic")
    if not topic:
        return None
    return {
        "code": topic["code"],
        "name": topic["name"],
        "level": topic["level"],
        "missing_prerequisites": [
            {"code": u["code"], "name": u["name"], "score": u["score"]} for u in graph.readiness(user.id, topic["code"])["unmet"]
        ],
    }


# --------------------------------------------------------------------------- #
#  Daily plan
# --------------------------------------------------------------------------- #
def difficulty_for(session: Session, user_id: int, skill_codes: list[str]) -> int:
    """1..5 question difficulty target from the learner's own ladder."""
    if not skill_codes:
        return 2
    rows = session.execute(
        select(UserSkill.difficulty_level, UserSkill.success_rate)
        .join(Skill, Skill.id == UserSkill.skill_id)
        .where(UserSkill.user_id == user_id, Skill.code.in_(skill_codes))
    ).all()
    if not rows:
        return 2
    ladder = max(int(r[0] or 1) for r in rows)
    success = sum(float(r[1] or 0) for r in rows) / len(rows)
    base = {1: 1, 2: 2, 3: 3, 4: 4}.get(ladder, 2)
    if success >= 85:
        base += 1
    elif success < 45:
        base -= 1
    return max(1, min(5, base))


def build_daily_plan(
    session: Session,
    user: User,
    *,
    plan_date: date | None = None,
    reason: str = "engine",
    extra_directives: list[dict[str, Any]] | None = None,
) -> DailyPlan:
    """Local planner (always runs). `extra_directives` lets the AI reorder/refine it."""
    from app.services.srs import due_summary

    ensure_skill_rows(session, user.id)
    apply_forgetting(session, user.id)
    plan_date = plan_date or date.today()
    graph = SkillGraph(session)
    minutes_budget = max(20, int(user.daily_minutes or 45))
    items: list[dict[str, Any]] = []
    used_minutes = 0

    def add(kind: str, code: str, title: str, why: str, minutes: int, payload: dict[str, Any] | None = None) -> None:
        nonlocal used_minutes
        if used_minutes + minutes > minutes_budget * 1.35 and kind not in {"reflection"}:
            return
        item = {
            "id": f"{kind}:{code or len(items)}",
            "kind": kind,
            "code": code,
            "title": title,
            "why": why,
            "minutes": minutes,
            "done": False,
            "payload": payload or {},
        }
        items.append(item)
        used_minutes += minutes

    # 1) spaced-repetition reviews come first, always
    due = due_summary(session, user.id)
    urgent = due.get("urgent") or []
    for entry in urgent[:3]:
        add(
            "review",
            entry["code"],
            f"Review {entry['code']} ({entry['overdue_hours']:.0f}h overdue)",
            "Spaced repetition is due; forgetting is cheapest to fight here.",
            4,
            {"type": entry["type"], "item_id": entry["item_id"], "ease": entry["ease"]},
        )

    # 2) prerequisite repair beats new material
    rec = recommend_next(session, user)
    topic = rec.get("topic") or {}
    remediation = rec.get("remediation")
    if remediation:
        add(
            "learn",
            remediation["skill"],
            f"Fix prerequisite: {remediation['name']}",
            remediation["why"],
            min(18, max(10, minutes_budget // 4)),
            {"remediation": True, "target_topic": topic.get("code", "")},
        )
    if topic:
        add(
            "learn",
            topic.get("code", ""),
            f"Learn: {topic.get('name')}",
            "Next topic at your current readiness: " + "; ".join(topic.get("reasons", [])[:2]),
            min(22, max(12, minutes_budget // 3)),
            {"level": topic.get("level"), "skills": graph.topic_skills.get(topic.get("code", ""), [])},
        )

    skill_codes = graph.topic_skills.get(topic.get("code", ""), []) if topic else []
    target_difficulty = difficulty_for(session, user.id, skill_codes)

    # 3) practice: math/problem solving on the weakest area
    weak = weak_skills(session, user.id, limit=4)
    practice_focus = (weak[0]["code"] if weak else (skill_codes[0] if skill_codes else ""))
    add(
        "practice",
        practice_focus,
        f"Practice: 2 problems (difficulty {target_difficulty}/5)",
        "Practice is generated from your library + skill graph at your current difficulty ladder.",
        min(16, max(8, minutes_budget // 5)),
        {"count": 2, "difficulty": target_difficulty, "topic": topic.get("code", "")},
    )

    # 4) coding task matched to the topic / weakness
    coding = pick_coding_task(session, user, graph=graph, topic_code=topic.get("code", ""), weak=[w["code"] for w in weak])
    if coding:
        add(
            "coding",
            coding["slug"],
            f"Coding: {coding['title']}",
            "Selected for your skills; tests run in the isolated sandbox."
            + (" (from-scratch implementation - the internals matter)" if coding.get("from_scratch") else ""),
            min(25, max(15, minutes_budget // 3)),
            coding,
        )

    # 5) project work if enrolled
    enrollment = session.execute(
        select(StudySession).where(StudySession.user_id == user.id, StudySession.activity == "project").order_by(StudySession.id.desc()).limit(1)
    ).scalar_one_or_none()
    active = session.execute(
        select(ProjectEnrollment)
        .where(ProjectEnrollment.user_id == user.id, ProjectEnrollment.status == "active")
        .order_by(ProjectEnrollment.id.desc())
        .limit(1)
    ).scalars().first()
    if active is not None:
        add(
            "project",
            f"project:{active.id}",
            "Project: continue current milestone",
            "Consistent project work is what converts exercises into engineering skill.",
            min(30, max(20, minutes_budget // 3)),
            {"enrollment_id": active.id},
        )
    elif enrollment is not None and minutes_budget >= 60:
        add("project", "project:new", "Start a project matched to your level", "You have no active project.", 25, {})

    # 6) reflection - the spec's own example, and it feeds the memory model
    focus_label = (remediation or {}).get("name") or topic.get("name") or "today's topic"
    add(
        "reflection",
        topic.get("code", ""),
        f"Reflection: explain {focus_label} in your own words",
        "Explaining exposes gaps; the Teacher stores them in your learning memory.",
        5,
        {"prompt": f"Explain {focus_label} in your own words (5-8 sentences), then list what you are unsure about."},
    )

    if extra_directives:
        items = _merge_directives(items, extra_directives, minutes_budget=minutes_budget, graph=graph)

    rationale = (
        f"Budget {minutes_budget} min. Reviews first ({due['due_total']} due), "
        + ("prerequisite repair ahead of new material, " if remediation else "prerequisites clear, ")
        + f"difficulty target {target_difficulty}/5."
    )
    plan = session.execute(
        select(DailyPlan).where(DailyPlan.user_id == user.id, DailyPlan.plan_date == plan_date)
    ).scalar_one_or_none()
    if plan is None:
        plan = DailyPlan(user_id=user.id, plan_date=plan_date)
        session.add(plan)
    plan.items = items
    plan.rationale = rationale
    plan.minutes_budget = minutes_budget
    plan.generated_by = reason
    plan.completed_ids = [i["id"] for i in items if i.get("done")] if plan.completed_ids is None else plan.completed_ids
    plan.updated_at = datetime.utcnow()
    session.flush()
    return plan


def _merge_directives(items: list[dict[str, Any]], directives: list[dict[str, Any]], *, minutes_budget: int, graph: SkillGraph) -> list[dict[str, Any]]:
    """Apply AI planner output: reorder + insert items the engine did not consider."""
    order = {d.get("code") or d.get("kind", ""): idx for idx, d in enumerate(directives)}
    by_kind = defaultdict(list)
    for item in items:
        by_kind[item["kind"]].append(item)
    known_codes = {c for c in graph.topics} | {c for c in graph.nodes}
    added: list[dict[str, Any]] = []
    for directive in directives:
        code = directive.get("code", "")
        kind = directive.get("kind", "learn")
        if kind not in PLAN_KINDS:
            continue
        if any(i.get("kind") == kind and i.get("code") == code for i in items):
            continue
        if code and code not in known_codes:
            continue  # never let the model invent a topic and have the UI point nowhere
        added.append(
            {
                "id": f"{kind}:{code or 'ai'}",
                "kind": kind,
                "code": code,
                "title": directive.get("title") or f"{kind}: {code}",
                "why": directive.get("why") or "Suggested by the AI planner",
                "minutes": int(directive.get("minutes") or 10),
                "done": False,
                "payload": {"from_ai": True},
            }
        )
    merged = items + added[:3]
    merged.sort(key=lambda i: (order.get(i["code"], 99), order.get(i["kind"], 99)))
    total = sum(i["minutes"] for i in merged)
    if total > minutes_budget * 1.35:
        trimmed: list[dict[str, Any]] = []
        running = 0
        for item in merged:
            if running + item["minutes"] > minutes_budget * 1.35 and trimmed:
                continue
            trimmed.append(item)
            running += item["minutes"]
        merged = trimmed
    return merged


def pick_coding_task(session: Session, user: User, *, graph: SkillGraph, topic_code: str, weak: list[str]) -> dict[str, Any] | None:
    """Pick an unattempted (or failed) coding task relevant to the current topic/skills."""
    attempted = set(
        session.execute(
            select(CodingAttempt.task_id).where(CodingAttempt.user_id == user.id, CodingAttempt.passed.is_(True))
        ).scalars()
    )
    wanted = set(weak) | set(graph.topic_skills.get(topic_code, []))
    stmt = select(CodingTask).where(CodingTask.id.notin_(attempted or {0})).order_by(CodingTask.difficulty.asc(), CodingTask.id.asc())
    rows = list(session.execute(stmt).scalars())
    if not rows:
        rows = list(session.execute(select(CodingTask).order_by(CodingTask.id.asc())).scalars())
    scored = []
    for task in rows:
        tags = set(task.skill_codes or [])
        overlap = len(tags & wanted)
        score = overlap * 3.0 - 0.15 * abs((task.difficulty or 2) - (1 + user.level_index // 3))
        if task.from_scratch and overlap:
            score += 1.0
        scored.append((score, task))
    scored.sort(key=lambda kv: (-kv[0], kv[1].id))
    for _score, task in scored:
        return {
            "slug": task.slug,
            "id": task.id,
            "title": task.title,
            "difficulty": task.difficulty,
            "from_scratch": bool(task.from_scratch),
            "libraries": task.libraries or [],
        }
    return None


def get_or_build_plan(session: Session, user: User, *, plan_date: date | None = None, force: bool = False) -> DailyPlan:
    plan_date = plan_date or date.today()
    plan = session.execute(
        select(DailyPlan).where(DailyPlan.user_id == user.id, DailyPlan.plan_date == plan_date)
    ).scalar_one_or_none()
    if plan is None or force:
        plan = build_daily_plan(session, user, plan_date=plan_date, reason="engine")
    # recompute "done" flags from real events so the plan is never a lie
    done_ids = set(plan.completed_ids or [])
    for item in plan.items or []:
        item["done"] = item.get("id") in done_ids or _item_completed(session, user.id, item)
    plan.completed_ids = [i["id"] for i in (plan.items or []) if i.get("done")]
    return plan


def _item_completed(session: Session, user_id: int, item: dict[str, Any]) -> bool:
    kind, code = item.get("kind"), item.get("code") or ""
    since = datetime.combine(date.today(), datetime.min.time())
    if kind == "review":
        return bool(
            session.execute(
                select(ReviewSchedule.id).where(
                    ReviewSchedule.user_id == user_id,
                    ReviewSchedule.item_code == code,
                    ReviewSchedule.last_review >= since,
                )
            ).scalar()
        )
    if kind == "coding":
        return bool(
            session.execute(
                select(CodingAttempt.id)
                .join(CodingTask, CodingTask.id == CodingAttempt.task_id)
                .where(CodingAttempt.user_id == user_id, CodingTask.slug == code, CodingAttempt.passed.is_(True), CodingAttempt.created_at >= since)
            ).scalar()
        )
    if kind in {"practice", "learn", "reflection", "exam_prep", "project"}:
        return bool(
            session.execute(
                select(StudySession.id).where(
                    StudySession.user_id == user_id,
                    StudySession.started_at >= since,
                    StudySession.activity == ("learn" if kind in {"learn", "reflection"} else kind),
                )
            ).scalar()
        )
    return False


def mark_item_done(session: Session, user: User, item_id: str, *, plan_date: date | None = None) -> DailyPlan | None:
    plan = get_or_build_plan(session, user, plan_date=plan_date)
    ids = set(plan.completed_ids or [])
    ids.add(item_id)
    plan.completed_ids = sorted(ids)
    for item in plan.items or []:
        if item.get("id") == item_id:
            item["done"] = True
    plan.updated_at = datetime.utcnow()
    session.flush()
    return plan


# --------------------------------------------------------------------------- #
#  Diagnostics (§13)
# --------------------------------------------------------------------------- #
DIAGNOSTIC_DOMAINS = ["linalg", "calc", "prob", "python", "sql", "ml", "dl", "mlops", "stats", "engineering"]


def diagnostic_question_pool(session: Session, *, count: int = 28) -> list[Question]:
    """Balanced sample across levels/domains; falls back to the whole bank if thin."""
    rows = list(
        session.execute(
            select(Question).where(Question.approved.is_(True)).order_by(Question.level.asc(), Question.id.asc())
        ).scalars()
    )
    if not rows:
        return []
    buckets: dict[int, list[Question]] = defaultdict(list)
    for question in rows:
        buckets[int(question.level or 0)].append(question)
    levels = sorted(buckets)
    picked: list[Question] = []
    round_idx = 0
    while len(picked) < count and any(buckets.values()):
        for level in levels:
            if buckets[level]:
                picked.append(buckets[level].pop(round_idx % len(buckets[level])) if len(buckets[level]) > round_idx else buckets[level].pop(0))
        round_idx += 1
        if round_idx > 50:
            break
    seen: set[int] = set()
    unique: list[Question] = []
    for question in picked:
        if question.id in seen:
            continue
        seen.add(question.id)
        unique.append(question)
        if len(unique) >= count:
            break
    if len(unique) < min(count, len(rows)):
        for question in rows:
            if question.id in seen:
                continue
            unique.append(question)
            if len(unique) >= count:
                break
    return unique


def start_diagnostic(session: Session, user: User, *, count: int = 28) -> dict[str, Any]:
    existing = session.execute(
        select(DiagnosticSession).where(DiagnosticSession.user_id == user.id, DiagnosticSession.status == "in_progress")
    ).scalar_one_or_none()
    if existing:
        return _diagnostic_payload(existing)
    pool = diagnostic_question_pool(session, count=count)
    items = [
        {
            "id": q.id,
            "code": q.public_code,
            "type": q.question_type,
            "difficulty": q.difficulty,
            "stem": q.stem,
            "options": q.options or [],
            "skills": q.skill_codes or [],
            "topic": q.topic_code,
            "level": q.level,
        }
        for q in pool
    ]
    record = DiagnosticSession(user_id=user.id, items=items, answers={}, status="in_progress")
    session.add(record)
    session.flush()
    return diagnostic_payload(record)


def submit_diagnostic(session: Session, user: User, answers: dict[str, Any], *, record_id: int | None = None) -> dict[str, Any]:
    """answers: {question_id: {"answer": str, "selected_option": int|None, "seconds": float}}"""
    q = select(DiagnosticSession).where(DiagnosticSession.user_id == user.id)
    record = session.execute(q.where(DiagnosticSession.status == "in_progress").order_by(DiagnosticSession.id.desc()).limit(1)).scalar_one_or_none()
    if record is None:
        record = session.get(DiagnosticSession, record_id) if record_id else None
    if record is None:
        raise ValueError("No diagnostic in progress")

    from app.services.questions import grade_locally

    per_skill: dict[str, list[float]] = defaultdict(list)
    per_level: dict[int, list[float]] = defaultdict(list)
    graded: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for item in record.items or []:
        raw = answers.get(str(item["id"])) or answers.get(item.get("code", "")) or {}
        if not raw:
            continue
        question = session.get(Question, int(item["id"]))
        if question is None:
            continue
        verdict = grade_locally(question=question, answer=str(raw.get("answer") or ""), selected_option=raw.get("selected_option"))
        results[str(item["id"])] = {"answer": raw.get("answer"), "selected_option": raw.get("selected_option")}
        score = float(verdict["score"])
        graded.append(
            {
                "id": item["id"],
                "code": item.get("code"),
                "skills": item.get("skills") or [],
                "level": item.get("level", 0),
                "score": score,
                "correct": bool(verdict["correct"]),
                "error_type": verdict.get("error_type", ""),
                "seconds": float(raw.get("seconds") or 0),
            }
        )
        for skill in item.get("skills") or []:
            per_skill[skill].append(score)
        per_level[int(item.get("level", 0))].append(score)

    # initial belief per skill: mean score tempered by how little evidence we have
    skill_scores = {code: (sum(v) / len(v), len(v)) for code, v in per_skill.items()}
    level_scores = {lvl: (sum(v) / len(v), len(v)) for lvl, v in per_level.items()}
    reachable = [lvl for lvl, (mean, n) in sorted(level_scores.items()) if mean >= 62.0 and n >= 1]
    level_index = min(10, max(reachable) + 1 if reachable else 0)
    level_index = min(level_index, 10)

    from app.seed.curriculum import LEVELS

    graph = SkillGraph(session)
    ensure_skill_rows(session, user.id)
    applied: list[dict[str, Any]] = []
    for code, (mean, n) in skill_scores.items():
        node = graph.nodes.get(code)
        skill = session.execute(select(Skill).where(Skill.code == code)).scalar_one_or_none()
        if skill is None:
            continue
        row = session.execute(
            select(UserSkill).where(UserSkill.user_id == user.id, UserSkill.skill_id == skill.id)
        ).scalar_one_or_none()
        if row is None:
            continue
        evidence_weight = min(0.75, 0.25 * n + 0.15)
        row.knowledge_score = round(float(row.knowledge_score or 0) + (mean - float(row.knowledge_score or 0)) * evidence_weight, 2)
        row.theory_score = round(float(row.theory_score or 0) + (mean - float(row.theory_score or 0)) * evidence_weight, 2)
        if mean >= 70:
            row.coding_score = round(float(row.coding_score or 0) + (mean - float(row.coding_score or 0)) * evidence_weight, 2)
        row.confidence = round(min(0.6, 0.12 + 0.12 * n), 3)
        row.attempts += n
        row.successes += sum(1 for g in graded if code in (g["skills"] or []) and g["correct"])
        row.success_rate = round(100.0 * row.successes / max(1, row.attempts), 1)
        row.difficulty_level = 1 + min(3, int(mean // 30))
        row.mastery_state = "developing" if 50 <= mean < 78 else ("solid" if mean >= 78 else "learning")
        row.last_review = datetime.utcnow()
        applied.append({"code": code, "score": round(mean, 1), "questions": n, "name": node.name if node else code})

    strengths = [a for a in applied if a["score"] >= 75]
    weaknesses = [a for a in applied if a["score"] < 55]
    user.level_index = int(level_index)
    user.level_label = LEVELS[min(10, int(level_index))]["title"] if level_index < len(LEVELS) else "Assessing"
    user.diagnostic_done = True
    user.onboarded = True

    record.answers = results
    record.status = "completed"
    record.completed_at = datetime.utcnow()
    record.result = {
        "level_index": int(level_index),
        "level_label": user.level_label,
        "overall_score": round(sum(g["score"] for g in graded) / max(1, len(graded)), 1),
        "answered": len(graded),
        "skills": applied,
        "strengths": strengths[:8],
        "weaknesses": weaknesses[:8],
        "by_level": {str(lvl): {"mean": round(mean, 1), "n": n} for lvl, (mean, n) in sorted(level_scores.items())},
    }
    session.flush()

    memory: list[dict[str, Any]] = [
        {"category": "goal", "key": "start_level", "value": user.level_label, "payload": {"level_index": level_index}},
    ]
    for entry in weaknesses[:10]:
        memory.append({"category": "weakness", "key": entry["code"], "value": f"{entry['score']}/100 at diagnostic", "payload": entry})
    for entry in strengths[:10]:
        memory.append({"category": "strength", "key": entry["code"], "value": f"{entry['score']}/100 at diagnostic", "payload": entry})
    _write_memory(session, user.id, memory)
    build_daily_plan(session, user, reason="diagnostic")
    session.commit()
    return {"diagnostic_id": record.id, **record.result, "recommendation": recommend_next(session, user)}


def diagnostic_payload(record: DiagnosticSession) -> dict[str, Any]:
    return {
        "diagnostic_id": record.id,
        "status": record.status,
        "count": len(record.items or []),
        "items": [
            {k: v for k, v in item.items() if k not in {"answer", "correct_answer"}}
            for item in (record.items or [])
        ],
        "result": record.result or {},
        "progress": len(record.answers or {}),
    }


def _write_memory(session: Session, user_id: int, entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        row = session.execute(
            select(LearningMemory).where(
                LearningMemory.user_id == user_id, LearningMemory.category == entry["category"], LearningMemory.key == entry["key"]
            )
        ).scalar_one_or_none()
        if row is None:
            session.add(LearningMemory(user_id=user_id, **entry))
        else:
            row.value = entry["value"]
            row.payload = entry.get("payload", {})
            row.updated_at = datetime.utcnow()
    session.flush()


# --------------------------------------------------------------------------- #
#  Post-answer feedback loop (§51)
# --------------------------------------------------------------------------- #
def on_attempt_recorded(
    session: Session,
    user: User,
    *,
    skill_codes: list[str],
    correct: bool,
    score: float,
    error_type: str = "",
    graph: SkillGraph | None = None,
) -> dict[str, Any]:
    """
    Self-improvement hook. If the same skill fails repeatedly, schedule its
    prerequisite, lower the difficulty ladder, and tell the caller what changed.
    """
    graph = graph or SkillGraph(session)
    actions: list[dict[str, Any]] = []
    for code in dict.fromkeys(skill_codes or []):
        skill = graph.nodes.get(code)
        if skill is None:
            continue
        row = session.execute(
            select(UserSkill, Skill).join(Skill, Skill.id == UserSkill.skill_id).where(
                UserSkill.user_id == user.id, Skill.code == code
            )
        ).first()
        if row is None:
            continue
        us, _skill = row
        if not correct and (us.failure_count or 0) >= 2:
            prereqs = graph.prerequisites(code, depth=1)
            scores = graph.user_scores(user.id)
            weakest_prereq = None
            for prereq in prereqs:
                value = scores.get(prereq, (0.0, 0.0))[0]
                if value < 55.0 and (weakest_prereq is None or value < weakest_prereq[1]):
                    weakest_prereq = (prereq, value)
            if weakest_prereq:
                from app.services.srs import schedule_event

                pnode = graph.nodes.get(weakest_prereq[0])
                pskill = session.execute(select(Skill).where(Skill.code == weakest_prereq[0])).scalar_one_or_none()
                if pskill is not None:
                    schedule_event(session, user_id=user.id, item_type="skill", item_id=int(pskill.id), item_code=weakest_prereq[0], quality=1)
                    actions.append(
                        {
                            "type": "schedule_prerequisite",
                            "skill": code,
                            "prerequisite": weakest_prereq[0],
                            "prerequisite_name": pnode.name if pnode else weakest_prereq[0],
                            "prerequisite_score": round(weakest_prereq[1], 1),
                            "message": (
                                f"Two failures on {pnode.name if pnode else weakest_prereq[0]} are blocking {skill.name}. "
                                "I inserted a targeted review + practice of the prerequisite before we go back."
                            ),
                        }
                    )
            if (us.difficulty_level or 1) > 1:
                us.difficulty_level = max(1, int(us.difficulty_level) - 1)
                actions.append({"type": "lower_difficulty", "skill": code, "to": us.difficulty_level})
        if correct and score >= 92 and (us.difficulty_level or 1) < 4:
            us.difficulty_level = min(4, int(us.difficulty_level or 1) + 1)
            actions.append({"type": "raise_difficulty", "skill": code, "to": us.difficulty_level})
    _write_memory(
        session,
        user.id,
        [
            {
                "category": "mistake" if not correct else "topic",
                "key": f"attempt:{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                "value": f"{'miss' if not correct else 'hit'} on {', '.join(skill_codes[:3]) or 'general'} ({round(score)})",
                "payload": {"skills": skill_codes, "score": score, "error_type": error_type},
            }
        ],
    )
    session.flush()
    if actions:
        build_daily_plan(session, user, reason="engine")
    return {"actions": actions, "plan_updated": bool(actions)}


# --------------------------------------------------------------------------- #
#  Velocity (§27)
# --------------------------------------------------------------------------- #
def velocity_summary(session: Session, user_id: int, *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.utcnow()
    week = now - timedelta(days=7)
    month = now - timedelta(days=30)

    q_week = session.execute(
        select(func.count(QuestionAttempt.id)).where(QuestionAttempt.user_id == user_id, QuestionAttempt.created_at >= week)
    ).scalar_one()
    c_week = session.execute(
        select(func.count(CodingAttempt.id)).where(CodingAttempt.user_id == user_id, CodingAttempt.created_at >= week, CodingAttempt.passed.is_(True))
    ).scalar_one()
    p_month = session.execute(
        select(func.count(StudySession.id)).where(StudySession.user_id == user_id, StudySession.activity == "project", StudySession.started_at >= month)
    ).scalar_one()
    minutes = float(
        session.execute(
            select(func.coalesce(func.sum(StudySession.active_seconds), 0)).where(
                StudySession.user_id == user_id, StudySession.started_at >= week
            )
        ).scalar_one()
        or 0
    ) / 60.0
    xp_week = int(
        session.execute(
            select(func.coalesce(func.sum(StudySession.xp), 0)).where(StudySession.user_id == user_id, StudySession.started_at >= week)
        ).scalar_one()
        or 0
    )
    improved = session.execute(
        select(func.count(UserSkill.id)).where(UserSkill.user_id == user_id, UserSkill.updated_at >= week, UserSkill.knowledge_score > 0)
    ).scalar_one()
    accuracy = session.execute(
        select(func.avg(QuestionAttempt.score), func.count(QuestionAttempt.id)).where(
            QuestionAttempt.user_id == user_id, QuestionAttempt.created_at >= week
        )
    ).one()
    resolved = 0
    for row in session.execute(
        select(UserSkill, Skill).join(Skill, Skill.id == UserSkill.skill_id).where(
            UserSkill.user_id == user_id, UserSkill.knowledge_score >= STRONG_SCORE_FLOOR, UserSkill.updated_at >= month
        )
    ).all():
        resolved += 1
    streak_days = _study_streak(session, user_id)
    return {
        "questions_per_week": int(q_week),
        "coding_tasks_per_week": int(c_week),
        "project_sessions_per_month": int(p_month),
        "study_minutes_week": round(minutes, 1),
        "xp_week": xp_week,
        "avg_score_week": round(float(accuracy[0] or 0), 1),
        "skills_touched_week": int(improved),
        "weak_skills_resolved_month": int(resolved),
        "streak_days": streak_days,
        "accuracy_week": round(100.0 * _week_accuracy(session, user_id, week), 1),
    }


def _week_accuracy(session: Session, user_id: int, since: datetime) -> float:
    total = session.execute(
        select(func.count(QuestionAttempt.id)).where(QuestionAttempt.user_id == user_id, QuestionAttempt.created_at >= since)
    ).scalar_one()
    if not total:
        return 0.0
    correct = session.execute(
        select(func.count(QuestionAttempt.id)).where(
            QuestionAttempt.user_id == user_id, QuestionAttempt.created_at >= since, QuestionAttempt.is_correct.is_(True)
        )
    ).scalar_one()
    return correct / total


def _study_streak(session: Session, user_id: int) -> int:
    days = set(
        session.execute(
            select(func.distinct(StudySession.started_at)).where(StudySession.user_id == user_id, StudySession.active_seconds > 60)
        ).scalars()
    )
    dates = {d.date() if isinstance(d, datetime) else d for d in days}
    if not dates:
        return 0
    streak = 0
    cursor = date.today()
    if cursor not in dates:
        cursor -= timedelta(days=1)
    while cursor in dates:
        streak += 1
        cursor -= timedelta(days=1)
    return streak

"""
User knowledge model (§12).

Progress is *not* pages read. Each skill carries a belief about what the learner
knows per dimension (theory / math / coding / problem solving / engineering),
how confident that belief is, and how it decays when neglected.

Update rule (per attempt):
    alpha        = alpha_max * evidence_weight   (fresh evidence moves the score more)
    target       = clamp(score_achieved, 0, 100)
    dim_score   <- dim_score + alpha * (target - dim_score)
    knowledge    = weighted mean of dimension scores (weights depend on skill category)
    confidence   = f(attempts, agreement between dimensions, recency)
plus: streaks, success_rate, adaptive difficulty level, mastery state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Skill, UserSkill
from app.services.srs import schedule_event

DIMENSIONS = ("theory_score", "math_score", "coding_score", "problem_solving_score", "engineering_score")

CATEGORY_DIM_WEIGHTS: dict[str, dict[str, float]] = {
    "theory": {"theory_score": 0.6, "problem_solving_score": 0.25, "math_score": 0.15},
    "math": {"math_score": 0.55, "theory_score": 0.2, "coding_score": 0.15, "problem_solving_score": 0.1},
    "coding": {"coding_score": 0.6, "engineering_score": 0.2, "problem_solving_score": 0.2},
    "engineering": {"engineering_score": 0.55, "coding_score": 0.3, "problem_solving_score": 0.15},
    "problem_solving": {"problem_solving_score": 0.5, "theory_score": 0.2, "coding_score": 0.15, "engineering_score": 0.15},
}

ACTIVITY_TO_DIMENSION = {
    "concept": "theory_score",
    "theory": "theory_score",
    "math": "math_score",
    "code": "coding_score",
    "coding": "coding_score",
    "practice": "problem_solving_score",
    "problem": "problem_solving_score",
    "project": "engineering_score",
    "engineering": "engineering_score",
    "exam": "problem_solving_score",
    "interview": "theory_score",
    "review": "theory_score",
}

MASTERY_THRESHOLDS = (
    (0.0, "new"),
    (1.0, "learning"),
    (50.0, "developing"),
    (70.0, "solid"),
    (85.0, "strong"),
    (93.0, "expert"),
)


@dataclass
class SkillUpdate:
    skill_code: str
    before: float
    after: float
    delta: float
    attempts: int
    confidence: float
    mastery_state: str
    difficulty_level: int
    next_review: datetime | None


def get_or_create(session: Session, user_id: int, skill: Skill) -> UserSkill:
    row = session.execute(
        select(UserSkill).where(UserSkill.user_id == user_id, UserSkill.skill_id == skill.id)
    ).scalar_one_or_none()
    if row is None:
        row = UserSkill(
            user_id=user_id,
            skill_id=skill.id,
            knowledge_score=0.0,
            confidence=0.0,
            theory_score=0.0,
            math_score=0.0,
            coding_score=0.0,
            problem_solving_score=0.0,
            engineering_score=0.0,
            difficulty_level=1,
            mastery_state="new",
        )
        session.add(row)
        session.flush()
    return row


def get_by_code(session: Session, user_id: int, skill_code: str) -> UserSkill | None:
    return (
        session.execute(
            select(UserSkill)
            .join(Skill, Skill.id == UserSkill.skill_id)
            .where(UserSkill.user_id == user_id, Skill.code == skill_code)
        )
        .scalars()
        .first()
    )


def update_from_activity(
    session: Session,
    *,
    user_id: int,
    skill_codes: list[str],
    activity: str,
    score: float,
    correct: bool | None = None,
    dimensions: dict[str, float] | None = None,
    difficulty: int = 2,
    xp: int = 0,
    minutes: float = 0.0,
    error_type: str = "",
) -> list[SkillUpdate]:
    """Apply one graded attempt to every skill it exercised."""
    skills = {
        skill.code: skill
        for skill in session.execute(select(Skill).where(Skill.code.in_(skill_codes or []))).scalars()
    }
    updates: list[SkillUpdate] = []
    score = max(0.0, min(100.0, float(score)))
    if correct is None:
        correct = score >= 62
    dim_scores = dimensions or {}

    for code in dict.fromkeys(skill_codes or []):
        skill = skills.get(code)
        if skill is None:
            continue
        row = get_or_create(session, user_id, skill)
        before = float(row.knowledge_score or 0.0)

        primary = ACTIVITY_TO_DIMENSION.get((activity or "").lower(), "theory_score")
        # Only the dimensions this activity actually exercised move. A learner who solves a
        # coding task must not have their theory score pushed around.
        targets: dict[str, float] = {primary: score}
        for dim, value in (dim_scores or {}).items():
            if dim in DIMENSIONS and value:
                targets[dim] = max(0.0, min(100.0, float(value)))
        # weak transfer: a small share of the achieved score hints at neighbouring dimensions
        transfer = 0.30 * score
        for dim in DIMENSIONS:
            if dim not in targets and getattr(row, dim):
                current = float(getattr(row, dim))
                # transfer credit may never *reduce* an existing dimension score
                targets[dim] = max(current, current * 0.9 + transfer * 0.1)

        evidence = int(row.attempts or 0)
        # Bayesian-flavoured step: the first observation is decisive, later ones refine.
        # A flawless solve of a *hard* task is strong evidence, so it moves the belief further.
        first_step = 0.85 if (evidence == 0 and int(difficulty or 2) >= 4) else 0.7
        alpha = max(0.15, first_step / (1.0 + 0.35 * evidence))
        for dim, target in targets.items():
            current = float(getattr(row, dim) or 0.0)
            weight = alpha if dim == primary else alpha * 0.6
            setattr(row, dim, round(current + weight * (target - current), 2))

        weights = dict(CATEGORY_DIM_WEIGHTS.get(skill.category, {"theory_score": 0.4, "problem_solving_score": 0.3, "coding_score": 0.3}))
        # Evidence from another kind of activity still counts, with less authority: acing a coding
        # exercise about Big-O *is* evidence about the Big-O skill.
        for dim in DIMENSIONS:
            weights.setdefault(dim, 0.3)
        demonstrated = [d for d in weights if float(getattr(row, d) or 0.0) > 0]
        relevant = list(weights)
        if not demonstrated:
            knowledge = 0.0
        else:
            numerator = sum(float(getattr(row, d) or 0.0) * weights[d] for d in demonstrated)
            denominator = sum(weights[d] for d in demonstrated) or 1.0
            mean = numerator / denominator
            # narrow evidence is trusted a bit less: one strong dimension cannot mean "mastered",
            # but it must not bury the estimate either (two dimensions reach the full value).
            breadth = min(1.0, len(demonstrated) / 2.0)
            knowledge = min(100.0, mean * (0.78 + 0.22 * breadth))

        row.attempts = (row.attempts or 0) + 1
        row.successes = (row.successes or 0) + (1 if correct else 0)
        row.success_rate = round(100.0 * row.successes / max(1, row.attempts), 1)
        row.streak = (row.streak or 0) + 1 if correct else 0
        row.failure_count = (row.failure_count or 0) + (0 if correct else 1)
        row.last_error_type = "" if correct else (error_type or "unknown")
        row.xp = (row.xp or 0) + xp
        row.minutes_studied = round(float(row.minutes_studied or 0.0) + minutes, 2)
        row.knowledge_score = round(max(0.0, min(100.0, knowledge)), 2)
        row.confidence = round(
            min(
                1.0,
                0.12
                + 0.16 * row.attempts
                + 0.18 * _dimension_agreement(row)
                + (0.06 if row.attempts >= 2 else 0.0),
            ),
            3,
        )
        row.difficulty_level = _adapt_difficulty(row, correct=correct, score=score, requested_difficulty=difficulty)
        row.mastery_state = _mastery_state(row)
        row.last_review = datetime.utcnow()
        row.updated_at = datetime.utcnow()

        quality = _quality_from(correct=correct, score=score, confidence=row.confidence)
        sched = schedule_event(
            session,
            user_id=user_id,
            item_type="skill",
            item_id=int(skill.id),
            item_code=code,
            quality=quality,
            difficulty=5 + (5 - min(5, difficulty)),
        )
        row.next_review = sched.next_review
        updates.append(
            SkillUpdate(
                skill_code=code,
                before=round(before, 2),
                after=row.knowledge_score,
                delta=round(row.knowledge_score - before, 2),
                attempts=row.attempts,
                confidence=row.confidence,
                mastery_state=row.mastery_state,
                difficulty_level=row.difficulty_level,
                next_review=row.next_review,
            )
        )
    session.flush()
    return updates


def _dimension_agreement(row: UserSkill) -> float:
    values = [float(getattr(row, d) or 0.0) for d in DIMENSIONS if getattr(row, d) is not None]
    values = [v for v in values if v > 0]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    spread = max(values) - min(values)
    return max(0.0, 1.0 - spread / 60.0)


def _quality_from(*, correct: bool, score: float, confidence: float) -> int:
    """Map grading outcome to an SRS quality grade 0..5."""
    if not correct:
        return 1 if score >= 40 else 0
    if score >= 93 and confidence >= 0.5:
        return 5
    if score >= 82:
        return 4
    if score >= 70:
        return 3
    return 2


def _adapt_difficulty(row: UserSkill, *, correct: bool, score: float, requested_difficulty: int) -> int:
    """Difficulty ladder 1..4 (beginner .. production) driven by performance."""
    level = row.difficulty_level or 1
    if correct and score >= 85:
        level += 1
    elif correct and score >= 70:
        level += 0
    elif not correct and score < 45:
        level -= 1
    elif not correct:
        level -= 0
    if requested_difficulty >= 4 and correct and score >= 90:
        level = max(level, 4)
    return max(1, min(4, int(level)))


def _mastery_state(row: UserSkill) -> str:
    score = float(row.knowledge_score or 0.0)
    attempts = int(row.attempts or 0)
    if attempts == 0:
        return "new"
    state = "new"
    for threshold, name in MASTERY_THRESHOLDS:
        if score >= threshold:
            state = name
    if row.success_rate >= 80 and float(row.confidence or 0) >= 0.7:
        state = "strong" if score >= 85 else "solid"
    if (row.last_review and row.last_review < datetime.utcnow() - timedelta(days=30)) and score < 90:
        state = "dormant"
    return state


def apply_forgetting(session: Session, user_id: int, *, now: datetime | None = None, verbose: bool = False) -> int:
    """
    Deterministic decay for neglected skills (offline-safe, idempotent per day).
    Long gaps reduce the score slightly and pull the item back into review.
    """
    now = now or datetime.utcnow()
    rows = session.execute(select(UserSkill).where(UserSkill.user_id == user_id, UserSkill.last_review.isnot(None))).scalars()
    touched = 0
    for row in rows:
        last = row.last_review
        if not last:
            continue
        days = (now - last).days
        if days < 14:
            continue
        half_life = 120.0 if (row.mastery_state or "") in {"strong", "expert"} else 60.0
        factor = 0.5 ** (days / half_life)
        decayed = round(max(0.0, (row.knowledge_score or 0.0) * (1.0 - 0.35 * (1.0 - factor))), 2)
        if abs(decayed - float(row.knowledge_score or 0.0)) < 0.6:
            continue
        row.knowledge_score = decayed
        row.theory_score = round(float(row.theory_score or 0) * (1.0 - 0.2 * (1 - factor)), 2)
        row.coding_score = round(float(row.coding_score or 0) * (1.0 - 0.2 * (1 - factor)), 2)
        row.mastery_state = _mastery_state(row)
        if row.next_review is None or row.next_review > now:
            row.next_review = now
        touched += 1
    if touched:
        session.flush()
    return touched


def ensure_skill_rows(session: Session, user_id: int) -> int:
    """Create zeroed rows for all skills so the roadmap can render a full skill map."""
    existing = set(session.execute(select(UserSkill.skill_id).where(UserSkill.user_id == user_id)).scalars())
    created = 0
    for skill in session.execute(select(Skill)).scalars():
        if skill.id in existing:
            continue
        session.add(UserSkill(user_id=user_id, skill_id=skill.id, knowledge_score=0.0, confidence=0.0, mastery_state="new", difficulty_level=1))
        created += 1
    if created:
        session.flush()
    return created


def skill_detail(session: Session, user_id: int, code: str) -> dict[str, Any] | None:
    row = session.execute(
        select(UserSkill, Skill)
        .join(Skill, Skill.id == UserSkill.skill_id)
        .where(UserSkill.user_id == user_id, Skill.code == code)
    ).first()
    if row is None:
        return None
    us, skill = row
    return {
        "code": skill.code,
        "name": skill.name,
        "category": skill.category,
        "level": skill.level,
        "knowledge_score": us.knowledge_score,
        "confidence": us.confidence,
        "success_rate": us.success_rate,
        "attempts": us.attempts,
        "streak": us.streak,
        "failure_count": us.failure_count,
        "difficulty_level": us.difficulty_level,
        "mastery_state": us.mastery_state,
        "theory": us.theory_score,
        "math": us.math_score,
        "coding": us.coding_score,
        "problem_solving": us.problem_solving_score,
        "engineering": us.engineering_score,
        "last_review": us.last_review.isoformat() if us.last_review else None,
        "next_review": us.next_review.isoformat() if us.next_review else None,
        "xp": us.xp,
        "minutes_studied": us.minutes_studied,
        "last_error_type": us.last_error_type,
    }

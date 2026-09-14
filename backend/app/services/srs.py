"""
Spaced repetition (§18).

SM-2 derived, with three product-specific tweaks:
  * failures return the item *the same day* (so a mistake is re-tested while fresh);
  * interval length is scaled by how hard the item was for this learner;
  * reviews never depend on the network - the whole loop is offline.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models import ReviewSchedule

MIN_EASE = 1.3
MAX_EASE = 3.1
SAME_DAY_RETRY = timedelta(minutes=45)
MAX_INTERVAL_DAYS = 300.0


def get_or_create(session: Session, *, user_id: int, item_type: str, item_id: int, item_code: str = "") -> ReviewSchedule:
    row = session.execute(
        select(ReviewSchedule).where(
            ReviewSchedule.user_id == user_id,
            ReviewSchedule.item_type == item_type,
            ReviewSchedule.item_id == int(item_id or 0),
        )
    ).scalar_one_or_none()
    if row is None:
        row = ReviewSchedule(
            user_id=user_id,
            item_type=item_type,
            item_id=int(item_id or 0),
            item_code=item_code,
            interval_days=0.0,
            ease_factor=2.5,
            repetitions=0,
            lapses=0,
            streak=0,
            difficulty=5.0,
            next_review=datetime.utcnow(),
        )
        session.add(row)
        session.flush()
    elif item_code and not row.item_code:
        row.item_code = item_code
    return row


def compute_next(row: ReviewSchedule, quality: int, *, now: datetime | None = None) -> dict[str, Any]:
    """Pure function - unit tested separately."""
    now = now or datetime.utcnow()
    quality = max(0, min(5, int(quality)))
    ease = float(row.ease_factor or 2.5)
    interval = float(row.interval_days or 0.0)
    reps = int(row.repetitions or 0)
    lapses = int(row.lapses or 0)
    streak = int(row.streak or 0)
    difficulty = float(row.difficulty or 5.0)

    if quality < 3:
        reps = 0
        lapses += 1
        streak = 0
        ease = max(MIN_EASE, ease - (0.24 if quality <= 1 else 0.14))
        # failed: back within the same session if possible, else tomorrow at latest
        next_at = now + SAME_DAY_RETRY
        interval = 0.0
    else:
        reps += 1
        streak += 1
        ease = min(MAX_EASE, ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
        if reps == 1:
            interval = 1.0
        elif reps == 2:
            interval = 3.0 if quality < 5 else 4.0
        else:
            difficulty_scale = max(0.55, min(1.25, 1.0 - 0.07 * (difficulty - 5.0)))
            interval = (interval * ease * difficulty_scale) if quality >= 3 else interval * 0.6
            if quality == 5:
                interval *= 1.12
        interval = min(MAX_INTERVAL_DAYS, max(0.5, interval))
        next_at = now + timedelta(days=interval)
        # never schedule a "due" review in the past for a success
        if next_at <= now:
            next_at = now + timedelta(days=max(0.5, interval))

    return {
        "interval_days": round(interval, 2),
        "ease_factor": round(ease, 3),
        "repetitions": reps,
        "lapses": lapses,
        "streak": streak,
        "next_review": next_at,
        "quality_last": quality,
        "last_review": now,
    }


def schedule_event(
    session: Session,
    *,
    user_id: int,
    item_type: str,
    item_id: int,
    item_code: str = "",
    quality: int,
    difficulty: float | None = None,
    now: datetime | None = None,
) -> ReviewSchedule:
    row = get_or_create(session, user_id=user_id, item_type=item_type, item_id=item_id, item_code=item_code)
    if difficulty is not None:
        # 1 (very easy for this learner) .. 9 (brutal) - smoothed with an EMA
        row.difficulty = round(0.7 * float(row.difficulty or 5.0) + 0.3 * float(difficulty), 2)
    update = compute_next(row, quality, now=now)
    for key, value in update.items():
        setattr(row, key, value)
    row.updated_at = datetime.utcnow()
    session.flush()
    return row


def due_rows(session: Session, user_id: int, *, now: datetime | None = None, item_type: str | None = None, limit: int = 60):
    now = now or datetime.utcnow()
    stmt = select(ReviewSchedule).where(
        ReviewSchedule.user_id == user_id,
        ReviewSchedule.suspended.is_(False),
        or_(ReviewSchedule.next_review.is_(None), ReviewSchedule.next_review <= now),
    )
    if item_type:
        stmt = stmt.where(ReviewSchedule.item_type == item_type)
    stmt = stmt.order_by(ReviewSchedule.next_review.asc().nullsfirst(), ReviewSchedule.ease_factor.asc()).limit(limit)
    return list(session.execute(stmt).scalars())


def due_summary(session: Session, user_id: int, *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.utcnow()
    rows = due_rows(session, user_id, now=now, limit=500)
    overdue_hours = [max(0.0, (now - (r.next_review or now)).total_seconds() / 3600.0) for r in rows]
    return {
        "due_total": len(rows),
        "due_skills": sum(1 for r in rows if r.item_type == "skill"),
        "due_questions": sum(1 for r in rows if r.item_type == "question"),
        "max_overdue_hours": round(max(overdue_hours), 1) if overdue_hours else 0.0,
        "urgent": [
            {
                "code": r.item_code,
                "type": r.item_type,
                "item_id": r.item_id,
                "due": r.next_review.isoformat() if r.next_review else None,
                "overdue_hours": round(h, 1),
                "ease": round(r.ease_factor, 2),
                "lapses": r.lapses,
                "streak": r.streak,
            }
            for r, h in sorted(zip(rows, overdue_hours), key=lambda kv: -kv[1])[:12]
        ],
    }


def schedule_payload(row: ReviewSchedule) -> dict[str, Any]:
    data = {
        "id": row.id,
        "item_type": row.item_type,
        "item_id": row.item_id,
        "item_code": row.item_code,
        "interval_days": row.interval_days,
        "ease_factor": row.ease_factor,
        "repetitions": row.repetitions,
        "lapses": row.lapses,
        "streak": row.streak,
        "difficulty": row.difficulty,
        "quality_last": row.quality_last,
        "last_review": row.last_review.isoformat() if row.last_review else None,
        "next_review": row.next_review.isoformat() if row.next_review else None,
        "suspended": row.suspended,
    }
    return data


def set_suspended(session: Session, user_id: int, schedule_id: int, suspended: bool) -> dict[str, Any] | None:
    row = session.get(ReviewSchedule, schedule_id)
    if row is None or row.user_id != user_id:
        return None
    row.suspended = suspended
    session.flush()
    return schedule_payload(row)


def stats(session: Session, user_id: int) -> dict[str, Any]:
    rows = session.execute(select(ReviewSchedule).where(ReviewSchedule.user_id == user_id)).scalars()
    schedules = list(rows)
    if not schedules:
        return {"items": 0, "avg_interval_days": 0.0, "avg_ease": 0.0, "total_lapses": 0, "mature": 0, "young": 0}
    intervals = [float(r.interval_days or 0.0) for r in schedules]
    return {
        "items": len(schedules),
        "avg_interval_days": round(sum(intervals) / len(intervals), 2),
        "avg_ease": round(sum(float(r.ease_factor or 0) for r in schedules) / len(schedules), 2),
        "total_lapses": int(sum(r.lapses or 0 for r in schedules)),
        "mature": int(sum(1 for i in intervals if i >= 21)),
        "young": int(sum(1 for i in intervals if 0 < i < 21)),
    }

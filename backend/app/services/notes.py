"""
Notes / bookmarks / mistakes (§32), learning journal with gap detection (§33),
learning memory (§34).

The journal is not a diary: its analysis feeds the Learning Engine (scheduling + memory),
which is what makes the system self-improving rather than merely recorded.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    Document,
    DocumentChunk,
    JournalEntry,
    LearningMemory,
    Question,
    QuestionAttempt,
    ReviewSchedule,
    Skill,
    Topic,
    User,
    UserNote,
)
from app.services.graph import SkillGraph

NOTE_KINDS = ("note", "bookmark", "mistake", "concept", "question")

GAP_MARKERS = (
    "не понимаю", "не понял", "забыл", "путал", "путаница", "неясно", "не могу объяснить", "не уверен",
    "don't understand", "do not understand", "didn't get", "confused", "keep forgetting", "not sure",
    "vague", "blurry", "can't explain", "cannot explain", "mixing up", "mixed up", "gap",
)
STRONG_MARKERS = (
    "now i get", "finally understood", "makes sense now", "understand it", "it clicks",
    "теперь понятно", "наконец понял", "сложилось",
)
CODE_SMELL_MARKERS = (
    "hardcode", "hard-code", "костыль", "copy-paste", "copy paste", "no tests", "без тестов",
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")


def _topic_candidates(text: str, graph: SkillGraph, session: Session) -> list[tuple[str, int]]:
    """(code, hits) for skills/topics mentioned in a free-text note, ranked."""
    low = text.lower()
    hits: dict[str, int] = {}
    for code, node in graph.nodes.items():
        score = 0
        if node.name and node.name.lower() in low:
            score += 3
        for kw in node.keywords or []:
            if len(kw) > 3 and re.search(rf"\b{re.escape(kw.lower())}\b", low):
                score += 1
        if score:
            hits[code] = hits.get(code, 0) + score
    for code, topic in graph.topics.items():
        score = 0
        if topic.name and topic.name.lower() in low:
            score += 3
        for kw in topic.keywords or []:
            if len(kw) > 3 and re.search(rf"\b{re.escape(kw.lower())}\b", low):
                score += 1
        if score:
            hits[code] = hits.get(code, 0) + score
    return sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))


# --------------------------------------------------------------------------- #
#  Notes CRUD
# --------------------------------------------------------------------------- #
def list_notes(session: Session, user: User, *, kind: str = "", topic: str = "", q: str = "", limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    stmt = select(UserNote).where(UserNote.user_id == user.id)
    if kind:
        stmt = stmt.where(UserNote.kind == kind)
    if topic:
        stmt = stmt.where(UserNote.topic_code == topic)
    if q:
        like = f"%{q}%"
        stmt = stmt.where((UserNote.title.ilike(like)) | (UserNote.body.ilike(like)) | (UserNote.excerpt.ilike(like)))
    rows = session.execute(stmt.order_by(UserNote.pinned.desc(), UserNote.updated_at.desc()).limit(min(limit, 300)).offset(offset)).scalars()
    return [note_payload(r) for r in rows]


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
        "pinned": bool(row.pinned),
        "created": row.created_at.isoformat() if row.created_at else None,
        "updated": row.updated_at.isoformat() if row.updated_at else None,
    }


def create_note(
    session: Session,
    user: User,
    *,
    kind: str = "note",
    title: str = "",
    body: str = "",
    topic_code: str = "",
    chunk_id: int | None = None,
    document_id: int | None = None,
    tags: list[str] | None = None,
    excerpt: str = "",
    citation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kind = kind if kind in NOTE_KINDS else "note"
    if chunk_id:
        chunk = session.get(DocumentChunk, int(chunk_id))
        if chunk is not None:
            document_id = document_id or chunk.document_id
            excerpt = excerpt or chunk.text[:1500]
            document = session.get(Document, chunk.document_id)
            citation = citation or {
                "document": document.title if document else "",
                "author": document.author if document else "",
                "chapter": chunk.chapter,
                "section": chunk.section,
                "page": chunk.page_start,
                "chunk_id": chunk.id,
            }
            topic_code = topic_code or chunk.topic
    row = UserNote(
        user_id=user.id,
        kind=kind,
        title=(title or (body.strip().split("\n")[0] if body.strip() else "Untitled note"))[:300],
        body=body[:40000],
        topic_code=topic_code,
        document_id=document_id,
        chunk_id=chunk_id,
        excerpt=excerpt[:8000],
        citation=citation or {},
        tags=[str(t)[:40] for t in (tags or [])][:12],
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return note_payload(row)


def update_note(
    session: Session,
    user: User,
    note_id: int,
    *,
    title: str | None = None,
    body: str | None = None,
    tags: list[str] | None = None,
    pinned: bool | None = None,
    kind: str | None = None,
    topic_code: str | None = None,
) -> dict[str, Any] | None:
    row = session.get(UserNote, int(note_id))
    if row is None or row.user_id != user.id:
        return None
    if title is not None:
        row.title = title[:300]
    if body is not None:
        row.body = body[:40000]
    if tags is not None:
        row.tags = [str(t)[:40] for t in tags][:12]
    if pinned is not None:
        row.pinned = bool(pinned)
    if kind is not None and kind in NOTE_KINDS:
        row.kind = kind
    if topic_code is not None:
        row.topic_code = topic_code
    row.updated_at = datetime.utcnow()
    session.commit()
    return note_payload(row)


def delete_note(session: Session, user: User, note_id: int) -> bool:
    row = session.get(UserNote, int(note_id))
    if row is None or row.user_id != user.id:
        return False
    session.delete(row)
    session.commit()
    return True


def note_from_mistake(session: Session, user: User, attempt_id: int) -> dict[str, Any] | None:
    """Persist a wrong answer as a mistake note, with the source that was being tested."""
    attempt = session.get(QuestionAttempt, int(attempt_id))
    if attempt is None or attempt.user_id != user.id:
        return None
    question = session.get(Question, attempt.question_id)
    if question is None:
        return None
    body = (
        f"Q: {question.stem}\n\n"
        f"My answer: {attempt.answer or '(none)'}\n\n"
        f"What was off: {attempt.feedback or attempt.correction or 'no detail recorded'}\n\n"
        f"Missed points: {'; '.join(attempt.missing_points or []) or '-'}\n"
    )
    return create_note(
        session,
        user,
        kind="mistake",
        title=f"Mistake · {question.public_code or question.id}",
        body=body[:20000],
        topic_code=question.topic_code,
        tags=list(question.skill_codes or [])[:6],
        citation=question.citation or {},
    )


# --------------------------------------------------------------------------- #
#  Journal (§33)
# --------------------------------------------------------------------------- #
def write_journal(session: Session, user: User, *, content: str, mood: int = 3, minutes: int = 0, entry_date: str | None = None) -> dict[str, Any]:
    graph = SkillGraph(session)
    parsed_date = datetime.fromisoformat(entry_date).date() if entry_date else date.today()
    entry = session.execute(
        select(JournalEntry).where(JournalEntry.user_id == user.id, JournalEntry.entry_date == parsed_date)
    ).scalar_one_or_none()
    if entry is None:
        entry = JournalEntry(user_id=user.id, entry_date=parsed_date)
        session.add(entry)
    entry.content = content[:20000]
    entry.mood = max(1, min(5, int(mood or 3)))
    entry.minutes = max(0, int(minutes or 0))
    entry.updated_at = datetime.utcnow()

    analysis = analyse_journal(session, user, content, graph=graph)
    entry.topics = analysis["topics"]
    entry.gaps = analysis["gaps"]
    entry.strengths = analysis["strengths"]
    entry.risk_flags = analysis["risk_flags"]
    entry.word_count = len(content.split())
    session.flush()

    _apply_journal_effects(session, user, analysis, graph=graph)
    session.commit()
    session.refresh(entry)
    return journal_payload(entry, analysis=analysis)


def analyse_journal(session: Session, user: User, content: str, *, graph: SkillGraph | None = None) -> dict[str, Any]:
    """
    Deterministic journal analysis: which topics the entry touches, whether the learner reports
    confusion about them, and what the engine should do about it.
    """
    graph = graph or SkillGraph(session)
    candidates = _topic_candidates(content, graph, session)
    scores = graph.user_scores(user.id)
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(content) if s.strip()]
    weak_sentences = [s for s in sentences if any(marker in s.lower() for marker in GAP_MARKERS)]
    strong_sentences = [s for s in sentences if any(marker in s.lower() for marker in STRONG_MARKERS)]
    risky = [s for s in sentences if any(marker in s.lower() for marker in CODE_SMELL_MARKERS)]

    def _signal(sentences: list[str], label: str, keywords: list[str]) -> bool:
        """
        Does any of these sentences talk about this skill/topic?
        Matched on any meaningful word of the name or its keywords - not just the first word,
        which is how "I don't understand the gradient of cross entropy" must hit `info.entropy`.
        """
        terms = {w for w in re.findall(r"[a-zа-яё0-9]{4,}", (label or "").lower())}
        terms |= {str(k).lower() for k in (keywords or []) if len(str(k)) >= 4}
        terms -= {"and", "the", "with", "from", "that", "this", "what", "when", "why", "how", "does", "into", "over", "are"}
        if not terms:
            first = (label or "").lower().split(" ")[0]
            terms = {first} if len(first) >= 4 else set()
        for sentence in sentences:
            lowered = sentence.lower()
            if any(re.search(rf"\b{re.escape(term)}\w*", lowered) for term in terms):
                return True
        return False

    topics, gaps, strengths = [], [], []
    for code, hits in candidates[:8]:
        node = graph.nodes.get(code)
        topic = graph.topics.get(code)
        label = node.name if node else (topic.name if topic else code)
        level = node.level if node else (topic.level if topic else 0)
        knowledge = scores.get(code, (0.0, 0.0))[0] if node else 0.0
        keywords = list((node.keywords if node else (topic.keywords if topic else [])) or [])
        mentions_gap = _signal(weak_sentences, label, keywords)
        mentions_strong = _signal(strong_sentences, label, keywords)
        entry = {
            "code": code,
            "name": label,
            "level": level,
            "hits": hits,
            "knowledge_score": round(float(knowledge), 1),
            "confusion_signal": bool(mentions_gap),
            "clarity_signal": bool(mentions_strong),
        }
        topics.append(entry)
        if mentions_gap or (node and knowledge < 55 and hits >= 2):
            gaps.append({**entry, "why": "reported confusion" if mentions_gap else "low demonstrated score"})
        elif mentions_strong and knowledge >= 70:
            strengths.append(entry)

    return {
        "topics": topics,
        "gaps": gaps[:6],
        "strengths": strengths[:6],
        "risk_flags": risky[:3],
        "sentiment": {
            "mood_words": len(strong_sentences),
            "confusion_words": len(weak_sentences),
            "words": len(content.split()),
        },
        "engine": "local",
        "summary": _summary_line(topics, gaps, strengths),
    }


def _summary_line(topics: list[dict[str, Any]], gaps: list[dict[str, Any]], strengths: list[dict[str, Any]]) -> str:
    parts = []
    if topics:
        parts.append("touched: " + ", ".join(t["name"] for t in topics[:4]))
    if gaps:
        parts.append("needs work: " + ", ".join(g["name"] for g in gaps[:3]))
    if strengths:
        parts.append("solid: " + ", ".join(s["name"] for s in strengths[:3]))
    if not parts:
        parts.append("no curriculum topic was identified in this entry")
    return " · ".join(parts)


def _apply_journal_effects(session: Session, user: User, analysis: dict[str, Any], *, graph: SkillGraph) -> None:
    """Journal gaps become review schedules + memory, so reflection changes the next plan."""
    from app.services.srs import get_or_create

    for gap in analysis["gaps"]:
        code = gap["code"]
        if code in graph.nodes:
            skill = session.execute(select(Skill).where(Skill.code == code)).scalar_one_or_none()
            if skill is not None:
                schedule = get_or_create(session, user_id=user.id, item_type="skill", item_id=int(skill.id), item_code=code)
                tomorrow = datetime.utcnow() + timedelta(hours=20)
                if schedule.next_review is None or schedule.next_review > tomorrow:
                    schedule.next_review = tomorrow
                schedule.difficulty = round(min(9.0, float(schedule.difficulty or 5.0) + 0.5), 2)
                session.flush()
        _remember(session, user.id, "weakness", f"journal:{code}", gap["name"] + " - reported confusion in the journal", {"from": "journal"})
    for strength in analysis["strengths"]:
        _remember(session, user.id, "strength", f"journal:{strength['code']}", strength["name"] + " - reported as understood", {"from": "journal"})
    for risk in analysis["risk_flags"]:
        _remember(session, user.id, "preference", f"risk:{abs(hash(risk)) % 10**8}", risk[:200], {"from": "journal"})
    if analysis["topics"]:
        _remember(
            session,
            user.id,
            "topic",
            f"journal:{date.today().isoformat()}",
            "; ".join(t["code"] for t in analysis["topics"][:6]),
            {"gaps": [g["code"] for g in analysis["gaps"]]},
        )
    from app.models import DailyPlan

    plan = session.execute(
        select(DailyPlan).where(DailyPlan.user_id == user.id, DailyPlan.plan_date == date.today())
    ).scalar_one_or_none()
    if plan is not None and analysis["gaps"]:
        plan.rationale = (plan.rationale or "") + " | journal gaps: " + ", ".join(g["name"] for g in analysis["gaps"][:3])
        session.flush()


def journal_payload(entry: JournalEntry, *, analysis: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": entry.id,
        "date": entry.entry_date.isoformat() if entry.entry_date else None,
        "content": entry.content,
        "mood": entry.mood,
        "minutes": entry.minutes,
        "word_count": entry.word_count,
        "topics": entry.topics or [],
        "gaps": entry.gaps or [],
        "strengths": entry.strengths or [],
        "risk_flags": entry.risk_flags or [],
        "analysis": analysis or {},
        "updated": entry.updated_at.isoformat() if entry.updated_at else None,
    }


def list_journal(session: Session, user: User, *, days: int = 30, limit: int = 30) -> list[dict[str, Any]]:
    since = date.today() - timedelta(days=days)
    rows = session.execute(
        select(JournalEntry).where(JournalEntry.user_id == user.id, JournalEntry.entry_date >= since).order_by(JournalEntry.entry_date.desc()).limit(min(limit, 120))
    ).scalars()
    return [journal_payload(r) for r in rows]


def delete_journal(session: Session, user: User, entry_id: int) -> bool:
    row = session.get(JournalEntry, int(entry_id))
    if row is None or row.user_id != user.id:
        return False
    session.delete(row)
    session.commit()
    return True


# --------------------------------------------------------------------------- #
#  Long-term learning memory (§34)
# --------------------------------------------------------------------------- #
def _remember(session: Session, user_id: int, category: str, key: str, value: str, payload: dict[str, Any] | None = None) -> None:
    row = session.execute(
        select(LearningMemory).where(LearningMemory.user_id == user_id, LearningMemory.category == category, LearningMemory.key == key)
    ).scalar_one_or_none()
    if row is None:
        session.add(LearningMemory(user_id=user_id, category=category, key=key[:160], value=value[:1000], payload=payload or {}))
    else:
        row.value = value[:1000]
        row.payload = payload or {}
        row.hits = (row.hits or 0) + 1
        row.updated_at = datetime.utcnow()
    session.flush()


def memory_bundle(session: Session, user_id: int) -> dict[str, Any]:
    rows = session.execute(select(LearningMemory).where(LearningMemory.user_id == user_id).order_by(LearningMemory.updated_at.desc())).scalars()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.category, []).append(
            {"key": row.key, "value": row.value, "payload": row.payload or {}, "weight": row.weight, "hits": row.hits, "at": row.updated_at.isoformat() if row.updated_at else None}
        )
    limits = {
        "strengths": ("strength", 12),
        "weaknesses": ("weakness", 14),
        "preferences": ("preference", 8),
        "topics": ("topic", 12),
        "projects": ("project", 8),
        "exams": ("exam", 8),
        "mistakes": ("mistake", 15),
        "history": ("history", 20),
        "facts": ("fact", 10),
        "risks": ("risk", 8),
    }
    bundle = {name: grouped.get(category, [])[: count] for name, (category, count) in limits.items()}
    recent_attempts = session.execute(
        select(QuestionAttempt.created_at, QuestionAttempt.score, QuestionAttempt.is_correct)
        .where(QuestionAttempt.user_id == user_id)
        .order_by(QuestionAttempt.id.desc())
        .limit(400)
    ).all()
    return {
        "summary": {
            "categories": {k: len(v) for k, v in grouped.items()},
            "total_rows": sum(len(v) for v in grouped.values()),
            "recent_attempts": len(recent_attempts),
        },
        **bundle,
    }


def forget(session: Session, user: User, *, category: str, key: str) -> int:
    rows = session.execute(
        select(LearningMemory).where(LearningMemory.user_id == user.id, LearningMemory.category == category, LearningMemory.key == key)
    ).scalars()
    items = list(rows)
    for row in items:
        session.delete(row)
    session.commit()
    return len(items)


def journal_streak(session: Session, user_id: int) -> int:
    dates = {row for row in session.execute(select(JournalEntry.entry_date).where(JournalEntry.user_id == user_id)).scalars()}
    if not dates:
        return 0
    streak, cursor = 0, date.today()
    while cursor in dates:
        streak += 1
        cursor -= timedelta(days=1)
    return streak

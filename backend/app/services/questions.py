"""
Question Engine (§16, §17, §30).

Two sources of questions, both persisted and both deduplicated:
  * the bank (seed + everything generated before), selected by topic/skill/difficulty;
  * the generator - AI when available, template-over-retrieved-text when not.

A new question is only served if it does not duplicate an existing one, matches the
learner's difficulty ladder, and (for KB-grounded questions) cites a real chunk.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.knowledge.textutil import content_hash, tokenize
from app.models import Document, DocumentChunk, Question, QuestionAttempt, StudySession, User
from app.services.graph import SkillGraph

MIN_JACCARD_DUP = 0.82
TYPES = ("conceptual", "mcq", "open", "math", "code", "debug", "architecture", "interview", "from_scratch")


def question_payload(question: Question, *, with_answer: bool = False, citation: dict[str, Any] | None = None) -> dict[str, Any]:
    data = {
        "id": question.id,
        "code": question.public_code,
        "type": question.question_type,
        "difficulty": question.difficulty,
        "level": question.level,
        "stem": question.stem,
        "options": question.options or [],
        "starter_code": question.starter_code or "",
        "hints": (question.hints or []) if with_answer else [],
        "topic": question.topic_code,
        "skills": question.skill_codes or [],
        "generated_by": question.generated_by,
        "citation": citation or question.citation or {},
        "est_seconds": 40 + 25 * int(question.difficulty or 2),
    }
    if with_answer:
        data.update(
            {
                "correct_option": question.correct_option,
                "expected_answer": question.expected_answer,
                "expected_value": question.expected_value,
                "tolerance": question.tolerance,
                "expected_points": question.expected_points or [],
                "explanation": question.explanation,
                "test_code": question.test_code or "",
            }
        )
    return data


def question_as_dict(question: Question) -> dict[str, Any]:
    """Shape used by both graders (AI prompt + local)."""
    return {
        "id": question.id,
        "stem": question.stem,
        "type": question.question_type,
        "question_type": question.question_type,
        "difficulty": question.difficulty,
        "options": question.options or [],
        "correct_option": question.correct_option,
        "expected_answer": question.expected_answer,
        "expected_value": question.expected_value,
        "tolerance": question.tolerance,
        "expected_points": question.expected_points or [],
        "explanation": question.explanation,
        "skills": question.skill_codes or [],
        "topic": question.topic_code,
        "citation": question.citation or {},
    }


# --------------------------------------------------------------------------- #
#  Selection
# --------------------------------------------------------------------------- #
def recent_question_ids(session: Session, user_id: int, *, days: int = 45, limit: int = 400) -> list[int]:
    return list(
        session.execute(
            select(QuestionAttempt.question_id)
            .where(QuestionAttempt.user_id == user_id, QuestionAttempt.created_at >= datetime.utcnow() - timedelta(days=days))
            .group_by(QuestionAttempt.question_id)
            .order_by(func.count(QuestionAttempt.id).desc())
            .limit(limit)
        ).scalars()
    )


def select_next(
    session: Session,
    user: User,
    *,
    topic_code: str = "",
    skill_codes: list[str] | None = None,
    types: list[str] | None = None,
    difficulty: int | None = None,
    include_answered: bool = False,
    allow_generation: bool = True,
) -> dict[str, Any] | None:
    graph = SkillGraph(session)
    skills = skill_codes or (graph.topic_skills.get(topic_code, []) if topic_code else [])
    if difficulty is None:
        from app.services.learning_engine import difficulty_for

        difficulty = difficulty_for(session, user.id, skills or [])

    stmt = select(Question).where(Question.approved.is_(True))
    if topic_code:
        stmt = stmt.where(or_(Question.topic_code == topic_code, Question.level == int(user.level_index or 0)))
    if types:
        stmt = stmt.where(Question.question_type.in_([t for t in types if t in TYPES]))

    candidates = list(session.execute(stmt.order_by(Question.id.asc()).limit(400)).scalars())
    if skills:
        preferred = [q for q in candidates if set(q.skill_codes or []) & set(skills)]
        if preferred:
            candidates = preferred
    if not include_answered:
        answered = set(recent_question_ids(session, user.id))
        fresh = [q for q in candidates if q.id not in answered]
        if fresh:
            candidates = fresh

    def key(q: Question) -> tuple[float, float]:
        diff_penalty = abs((q.difficulty or 2) - difficulty) * 1.4
        level_bonus = 0.0 if q.level <= (user.level_index or 0) + 1 else 1.5
        use_penalty = 0.12 * (q.use_count or 0)
        skill_bonus = 2.0 if skills and set(q.skill_codes or []) & set(skills) else 0.0
        grounded_bonus = 0.6 if q.source_chunk_id else 0.0
        return (-(skill_bonus + grounded_bonus) + diff_penalty + use_penalty + level_bonus, float(q.id))

    candidates.sort(key=key)
    if candidates:
        question = candidates[0]
        question.use_count = (question.use_count or 0) + 1
        session.flush()
        return {
            "question": question_payload(question),
            "origin": "bank",
            "context": _context_for(session, question),
        }

    if not allow_generation:
        return None
    generated = generate_questions(
        session,
        user,
        topic_code=topic_code or (skills[0] if skills else ""),
        skill_codes=skills,
        count=1,
        difficulty=difficulty,
        types=types or ["conceptual", "mcq", "open", "math"],
    )
    if not generated:
        return None
    question = generated[0]
    question.use_count = (question.use_count or 0) + 1
    session.flush()
    return {"question": question_payload(question), "origin": "generated", "context": _context_for(session, question)}


def _context_for(session: Session, question: Question) -> dict[str, Any]:
    if not question.source_chunk_id:
        return {}
    chunk = session.get(DocumentChunk, question.source_chunk_id)
    if chunk is None:
        return {}
    document = session.get(Document, chunk.document_id)
    return {
        "document": document.title if document else "",
        "chapter": chunk.chapter,
        "section": chunk.section,
        "page": chunk.page_start,
        "excerpt": chunk.text[:600],
    }


# --------------------------------------------------------------------------- #
#  Generation (AI -> bank; no AI -> templates over retrieved text)
# --------------------------------------------------------------------------- #
def generate_questions(
    session: Session,
    user: User,
    *,
    topic_code: str = "",
    skill_codes: list[str] | None = None,
    count: int = 3,
    difficulty: int = 2,
    types: list[str] | None = None,
    persist: bool = True,
    allow_ai: bool = True,
) -> list[Question]:
    from app.ai.manager import manager as ai_manager
    from app.knowledge.retrieval import Retriever, build_context
    from app.services.learning_engine import learner_state

    graph = SkillGraph(session)
    skills = skill_codes or (graph.topic_skills.get(topic_code, []) if topic_code else [])
    topic_name = topic_code
    if topic_code in graph.topics:
        topic_name = graph.topics[topic_code].name
    retriever = Retriever(session)
    results = retriever.search(topic_name or " ".join(skills), top_k=6) if (topic_name or skills) else []
    context, citations = build_context(results)

    existing_hashes = set(
        session.execute(select(Question.content_hash).where(Question.content_hash != "")).scalars()
    )
    existing_stems = list(
        session.execute(
            select(Question.stem).where(Question.topic_code == topic_code).limit(80) if topic_code
        else select(Question.stem).limit(80)
        ).scalars()
    )
    created: list[Question] = []

    ai_result = None
    if allow_ai and ai_manager.is_configured() and user.ai_enabled:
        ai_result = ai_manager.invoke(
            "generate_question",
            user_id=user.id,
            kwargs={
                "spec": {
                    "count": count,
                    "topic": topic_name or (skills[0] if skills else "the learner's weak area"),
                    "types": [t for t in (types or []) if t in TYPES] or ["conceptual", "mcq", "open"],
                    "difficulty": difficulty,
                    "level": user.level_index,
                    "skills": skills[:6],
                },
                "context": context,
                "avoid": [s[:160] for s in existing_stems],
            },
            session=session,
        )

    if ai_result is not None and ai_result.ok and isinstance(ai_result.data, dict):
        for raw in (ai_result.data or {}).get("questions", [])[:count]:
            question = _build_from_ai(session, raw, topic_code=topic_code, skills=skills, user=user, citations=citations)
            if question is None:
                continue
            if question.content_hash in existing_hashes:
                continue
            existing_hashes.add(question.content_hash)
            created.append(question)

    if len(created) < count:
        from app.ai.fallback import local_questions_from_chunks

        specs = local_questions_from_chunks(
            topic=topic_name or (skills[0] if skills else ""),
            results=results,
            count=max(1, count - len(created)) + 1,
            difficulty=difficulty,
        )
        for raw in specs:
            question = _build_from_ai(session, raw, topic_code=topic_code, skills=skills, user=user, citations=citations, template=True)
            if question is None or question.content_hash in existing_hashes:
                continue
            existing_hashes.add(question.content_hash)
            created.append(question)

    if persist and created:
        session.add_all(created)
        session.flush()
    return created


def _build_from_ai(
    session: Session,
    raw: dict[str, Any],
    *,
    topic_code: str,
    skills: list[str],
    user: User,
    citations: list[dict[str, Any]],
    template: bool = False,
) -> Question | None:
    stem = str(raw.get("stem") or "").strip()
    if len(stem) < 15:
        return None
    qtype = str(raw.get("type") or "conceptual").lower()
    if qtype not in TYPES:
        qtype = "conceptual"
    difficulty = max(1, min(5, int(raw.get("difficulty") or 2)))
    options = raw.get("options") or []
    normalised_options: list[dict[str, Any]] = []
    if qtype == "mcq" and isinstance(options, list):
        for i, option in enumerate(options[:6]):
            if isinstance(option, dict):
                normalised_options.append({"id": int(option.get("id", i)), "text": str(option.get("text", ""))[:400]})
            else:
                normalised_options.append({"id": i, "text": str(option)[:400]})
        if len(normalised_options) < 2:
            qtype = "open"
    correct_option = raw.get("correct_option")
    if qtype == "mcq":
        try:
            correct_option = int(correct_option) if correct_option is not None else None
        except (TypeError, ValueError):
            correct_option = None
        if correct_option is None or correct_option < 0 or correct_option >= len(normalised_options):
            qtype = "open"
            correct_option = None

    ref_indexes = [i for i in (raw.get("source_refs") or []) if isinstance(i, int) and 1 <= i <= len(citations)]
    citation = citations[ref_indexes[0] - 1] if ref_indexes else {}
    chunk_id = int(citation.get("chunk_id")) if citation.get("chunk_id") else None
    if not template:
        # integrity rule: a non-grounded AI question must not pretend to cite the library
        if raw.get("source_refs") and not citation:
            citation = {}
            chunk_id = None

    text_for_hash = stem + "||" + str(raw.get("expected_answer") or "")
    digest = content_hash(text_for_hash)
    level = int(user.level_index or 0)
    if topic_code in _topic_levels(session):
        level = _topic_levels(session)[topic_code]

    return Question(
        public_code=f"q{digest[:8]}",
        topic_code=topic_code,
        skill_codes=list(raw.get("skills") or skills)[:6],
        level=level,
        question_type=qtype,
        difficulty=difficulty,
        stem=stem[:4000],
        options=normalised_options,
        correct_option=correct_option,
        expected_answer=str(raw.get("expected_answer") or "")[:3000],
        expected_value=str(raw.get("expected_value") or "")[:180],
        tolerance=float(raw.get("tolerance") or 0.0),
        expected_points=[str(p)[:300] for p in (raw.get("expected_points") or [])][:8],
        rubric=raw.get("rubric") or [],
        explanation=str(raw.get("explanation") or "")[:3000],
        starter_code=str(raw.get("starter_code") or ""),
        hints=[str(h)[:400] for h in (raw.get("hints") or [])][:4],
        source_chunk_id=chunk_id,
        source_document_id=citation.get("document_id"),
        citation=citation,
        generated_by="ai" if not template else "template",
        content_hash=digest,
        embedding_text=stem,
        approved=True,
    )


_topic_level_cache: dict[int, dict[str, int]] = {}


def _topic_levels(session: Session) -> dict[str, int]:
    key = id(session)
    if key not in _topic_level_cache:
        from app.models import Topic

        _topic_level_cache[key] = {t.code: t.level for t in session.execute(select(Topic)).scalars()}
        if len(_topic_level_cache) > 8:
            _topic_level_cache.pop(next(iter(_topic_level_cache)), None)
    return _topic_level_cache[key]


def find_duplicate(session: Session, stem: str, *, threshold: float = MIN_JACCARD_DUP) -> Question | None:
    """Near-duplicate guard (§16) using token Jaccard over existing stems."""
    new_tokens = set(tokenize(stem))
    if not new_tokens:
        return None
    best: tuple[float, Question] | None = None
    for question in session.execute(select(Question).order_by(Question.id.desc()).limit(600)).scalars():
        other = set(tokenize(question.embedding_text or question.stem))
        if not other:
            continue
        inter = len(new_tokens & other)
        if inter == 0:
            continue
        jaccard = inter / len(new_tokens | other)
        if jaccard >= threshold and (best is None or jaccard > best[0]):
            best = (jaccard, question)
    return best[1] if best else None


# --------------------------------------------------------------------------- #
#  Grading
# --------------------------------------------------------------------------- #
def grade_locally(*, question: Any, answer: str, selected_option: int | None = None, time_seconds: float = 0.0) -> dict[str, Any]:
    """
    Deterministic grading. Question may be an ORM row or a dict.
    Same output schema as the AI evaluator so nothing downstream branches on it.
    """
    from app.ai.fallback import evaluate_locally

    data = question_as_dict(question) if isinstance(question, Question) else dict(question)
    if selected_option is not None and data.get("options"):
        data["type"] = "mcq"
        answer = answer or str(selected_option)
    return evaluate_locally(question=data, answer=answer, time_seconds=time_seconds)


def grade(
    session: Session,
    user: User,
    *,
    question: Question,
    answer: str,
    selected_option: int | None = None,
    code: str = "",
    time_seconds: float = 0.0,
    allow_ai: bool = True,
) -> tuple[dict[str, Any], str]:
    """AI grading when configured & affordable, always with a local safety net."""
    from app.ai.manager import manager as ai_manager
    from app.knowledge.retrieval import Retriever, build_context
    from app.services.learning_engine import learner_state

    local = grade_locally(question=question, answer=answer, selected_option=selected_option, time_seconds=time_seconds)
    data = question_as_dict(question)
    if code:
        data["learner_code"] = code

    if not allow_ai or not user.ai_enabled or not ai_manager.is_configured():
        return local, "local"

    context = ""
    if question.source_chunk_id:
        chunk = session.get(DocumentChunk, question.source_chunk_id)
        if chunk is not None:
            document = session.get(Document, chunk.document_id)
            context = (
                f"[1] {document.title if document else 'library'} · {chunk.chapter or chunk.section or ''} "
                f"{'Page ' + str(chunk.page_start) if chunk.page_start else ''}\n{chunk.text[:2500]}"
            )
    elif settings.retrieval_top_k:
        results = Retriever(session).search(question.stem, top_k=3)
        if results:
            context, _ = build_context(results, max_chars=4000)

    result = ai_manager.invoke(
        "evaluate_answer",
        user_id=user.id,
        kwargs={
            "question": data,
            "answer": (answer or "") + (("\n\nCODE:\n" + code) if code else ""),
            "context": context,
            "learner_state": json.dumps(learner_state(session, user), ensure_ascii=False)[:2500],
        },
        session=session,
    )
    if not result.ok or not isinstance(result.data, dict):
        local["ai_note"] = result.user_message()
        local["ai_error_code"] = result.error_code
        return local, "local"

    payload = result.data
    merged = {
        "correct": bool(payload.get("correct", local["correct"])),
        "score": float(payload.get("score", local["score"]) or 0),
        "dimensions": {**local["dimensions"], **(payload.get("dimensions") or {})},
        "error_type": payload.get("error_type") or local["error_type"],
        "what_is_wrong": payload.get("what_is_wrong") or "",
        "correction": payload.get("correction") or local.get("correction") or "",
        "feedback": payload.get("feedback") or local["feedback"],
        "missing_points": payload.get("missing_points") or local["missing_points"],
        "check_question": payload.get("check_question") or local.get("check_question"),
        "evaluated_by": "ai",
        "engine": "ai",
        "time_seconds": time_seconds,
        "sources": _citations_for(question, context),
        "usage": asdict(result.usage) if result.usage else {},
    }
    # sanity guard: a model that returns nonsense should not destroy the local grade
    if not 0 <= merged["score"] <= 100:
        merged["score"] = local["score"]
    return merged, "ai"


def _citations_for(question: Question, context: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if question.citation:
        out.append(dict(question.citation))
    return out


# --------------------------------------------------------------------------- #
#  Answer submission (the loop that moves state)
# --------------------------------------------------------------------------- #
def submit_answer(
    session: Session,
    user: User,
    *,
    question_id: int,
    answer: str,
    selected_option: int | None = None,
    code: str = "",
    time_seconds: float = 0.0,
    session_id: int | None = None,
    allow_ai: bool = True,
    source: str = "learn",
) -> dict[str, Any]:
    question = session.get(Question, int(question_id))
    if question is None:
        raise ValueError(f"Question {question_id} not found")

    verdict, engine = grade(
        session, user, question=question, answer=answer, selected_option=selected_option,
        code=code, time_seconds=time_seconds, allow_ai=allow_ai,
    )
    attempt = QuestionAttempt(
        user_id=user.id,
        question_id=question.id,
        answer=(answer or "")[:8000],
        selected_option=selected_option,
        code=code[:8000] if code else "",
        is_correct=bool(verdict["correct"]),
        score=float(verdict["score"]),
        correctness=float(verdict["dimensions"].get("correctness", 0)),
        depth=float(verdict["dimensions"].get("depth", 0)),
        reasoning=float(verdict["dimensions"].get("reasoning", 0)),
        precision=float(verdict["dimensions"].get("precision", 0)),
        confidence=float(verdict["dimensions"].get("confidence", 0)),
        error_type=str(verdict.get("error_type") or ""),
        missing_points=verdict.get("missing_points") or [],
        feedback=str(verdict.get("feedback") or "")[:6000],
        correction=str(verdict.get("correction") or "")[:4000],
        evaluated_by=engine,
        time_seconds=float(time_seconds or 0),
        session_id=session_id,
        meta={"source": source, "ai_error": verdict.get("ai_error_code", "")},
    )
    session.add(attempt)
    question.answer_count = (question.answer_count or 0) + 1
    question.correct_count = (question.correct_count or 0) + (1 if verdict["correct"] else 0)
    session.flush()

    from app.services.user_knowledge import update_from_activity

    activity = "code" if (question.question_type in {"code", "from_scratch", "debug"} or code) else (
        "math" if question.question_type == "math" else "concept"
    )
    updates = update_from_activity(
        session,
        user_id=user.id,
        skill_codes=question.skill_codes or [],
        activity=activity,
        score=float(verdict["score"]),
        correct=bool(verdict["correct"]),
        dimensions={
            "theory_score": float(verdict["dimensions"].get("correctness", 0)),
            "math_score": float(verdict["dimensions"].get("precision", 0)) if question.question_type == "math" else float(verdict["dimensions"].get("correctness", 0) * 0.6),
            "coding_score": float(verdict["dimensions"].get("reasoning", 0)) if code else 0.0,
            "problem_solving_score": (float(verdict["dimensions"].get("depth", 0)) + float(verdict["dimensions"].get("reasoning", 0))) / 2,
            "engineering_score": float(verdict["dimensions"].get("depth", 0)) if question.question_type in {"architecture", "interview"} else 0.0,
        },
        difficulty=int(question.difficulty or 2),
        xp=_xp_for(question, correct=bool(verdict["correct"])),
        error_type=str(verdict.get("error_type") or ""),
    )

    from app.services.learning_engine import on_attempt_recorded

    feedback = on_attempt_recorded(
        session,
        user,
        skill_codes=question.skill_codes or [],
        correct=bool(verdict["correct"]),
        score=float(verdict["score"]),
        error_type=str(verdict.get("error_type") or ""),
    )

    next_question = None
    if not verdict["correct"] and verdict.get("check_question"):
        next_question = _persist_check_question(session, user, question, verdict["check_question"])

    xp = _xp_for(question, correct=bool(verdict["correct"]))
    if session_id:
        study = session.get(StudySession, session_id)
        if study is not None:
            study.xp = int(study.xp or 0) + xp
            study.items_done = int(study.items_done or 0) + 1
            study.ai_used = study.ai_used or engine == "ai"
            session.flush()

    session.commit()
    return {
        "attempt_id": attempt.id,
        "question_id": question.id,
        "verdict": verdict,
        "engine": engine,
        "skill_updates": [asdict(u) for u in updates],
        "learning_engine": feedback,
        "retry_question": next_question,
        "xp": xp,
        "sources": verdict.get("sources") or ([question.citation] if question.citation else []),
    }


def _xp_for(question: Question, *, correct: bool) -> int:
    base = {1: 6, 2: 9, 3: 13, 4: 18, 5: 25}.get(int(question.difficulty or 2), 10)
    return base if correct else max(1, base // 4)


def _persist_check_question(session: Session, user: User, parent: Question, check: dict[str, Any]) -> dict[str, Any] | None:
    stem = str(check.get("stem") or "").strip()
    if len(stem) < 12:
        return None
    digest = content_hash(stem)
    existing = session.execute(select(Question).where(Question.content_hash == digest)).scalar_one_or_none()
    if existing is None:
        options = check.get("options") or []
        normalised = []
        for i, option in enumerate(options[:6] if isinstance(options, list) else []):
            normalised.append(option if isinstance(option, dict) else {"id": i, "text": str(option)})
        qtype = str(check.get("type") or "conceptual").lower()
        if qtype not in TYPES:
            qtype = "conceptual"
        if qtype == "mcq" and len(normalised) < 2:
            qtype = "open"
        existing = Question(
            public_code=f"q{digest[:8]}",
            topic_code=parent.topic_code,
            skill_codes=parent.skill_codes or [],
            level=parent.level,
            question_type=qtype,
            difficulty=max(1, int(parent.difficulty or 2) - 1),
            stem=stem[:4000],
            options=normalised,
            correct_option=check.get("correct_option") if qtype == "mcq" else None,
            expected_answer=str(check.get("expected_answer") or "")[:3000],
            expected_points=[str(p)[:300] for p in (check.get("expected_points") or [])][:6],
            explanation=str(check.get("explanation") or parent.explanation or "")[:3000],
            citation=parent.citation or {},
            source_document_id=parent.source_document_id,
            generated_by="ai" if parent.generated_by == "ai" else "template",
            content_hash=digest,
            embedding_text=stem,
        )
        session.add(existing)
        session.flush()
    existing.use_count = (existing.use_count or 0) + 1
    session.flush()
    return question_payload(existing)


# --------------------------------------------------------------------------- #
#  Review-my-answer (used by the AI Teacher)
# --------------------------------------------------------------------------- #
def review_free_text(
    session: Session,
    user: User,
    *,
    prompt: str,
    answer: str,
    topic_code: str = "",
    context: str = "",
    allow_ai: bool = True,
) -> dict[str, Any]:
    """Grade an arbitrary written explanation (§17 without a stored question)."""
    from app.ai.fallback import evaluate_locally

    pseudo = {
        "stem": prompt,
        "type": "open",
        "difficulty": 3,
        "expected_points": [],
        "expected_answer": "",
        "options": [],
        "correct_option": None,
        "expected_value": "",
        "tolerance": 0.0,
        "explanation": "",
        "skills": [],
        "citation": {},
    }
    local = evaluate_locally(question=pseudo, answer=answer)
    engine = "local"
    result = None
    if allow_ai:
        from app.ai.manager import manager as ai_manager

        if ai_manager.is_configured() and user.ai_enabled:
            result = ai_manager.invoke(
                "evaluate_answer",
                user_id=user.id,
                kwargs={"question": pseudo, "answer": answer, "context": context},
                session=session,
            )
            if result.ok and isinstance(result.data, dict):
                payload = result.data
                local = {
                    "correct": bool(payload.get("correct")),
                    "score": float(payload.get("score") or 0),
                    "dimensions": {**local["dimensions"], **(payload.get("dimensions") or {})},
                    "error_type": payload.get("error_type") or "none",
                    "what_is_wrong": payload.get("what_is_wrong") or "",
                    "correction": payload.get("correction") or "",
                    "feedback": payload.get("feedback") or local["feedback"],
                    "missing_points": payload.get("missing_points") or [],
                    "check_question": payload.get("check_question"),
                    "evaluated_by": "ai",
                }
                engine = "ai"
    if topic_code:
        graph = SkillGraph(session)
        from app.services.user_knowledge import update_from_activity

        update_from_activity(
            session,
            user_id=user.id,
            skill_codes=graph.topic_skills.get(topic_code, [])[:3],
            activity="concept",
            score=float(local["score"]),
            correct=bool(local["correct"]),
            difficulty=3,
            xp=6 if local["correct"] else 2,
        )
        session.commit()
    local["engine"] = engine
    local["usage"] = (asdict(result.usage) if result and result.usage else {})
    return local


def list_questions(session: Session, *, topic_code: str = "", qtype: str = "", limit: int = 60, offset: int = 0) -> dict[str, Any]:
    stmt = select(Question)
    if topic_code:
        stmt = stmt.where(Question.topic_code == topic_code)
    if qtype:
        stmt = stmt.where(Question.question_type == qtype)
    total = session.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    rows = session.execute(stmt.order_by(Question.id.desc()).limit(limit).offset(offset)).scalars()
    return {"total": int(total), "items": [question_payload(q) for q in rows]}

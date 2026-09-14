"""Knowledge-base search + AI Teacher endpoints (§5, §8, §9, §30, §31)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import db_dep, get_current_user
from app.config import settings
from app.knowledge.retrieval import Retriever, build_context, detect_conflicts, format_sources_for_user
from app.knowledge.vectorstore import get_vector_store
from app.models import Document, DocumentChunk, Question, TeacherConversation, TeacherMessage, User
from app.services import teacher
from app.services.graph import SkillGraph

router = APIRouter(tags=["kb & teacher"])


class SearchIn(BaseModel):
    query: str = Field(min_length=2, max_length=600)
    top_k: int = 8
    topic_codes: list[str] = []
    document_ids: list[int] = []
    include_full_text: bool = False
    detect_conflicts: bool = True


@router.post("/kb/search")
def kb_search(payload: SearchIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    retriever = Retriever(session)
    results = retriever.search(
        payload.query,
        top_k=max(1, min(25, payload.top_k)),
        topic_codes=payload.topic_codes or None,
        document_ids=payload.document_ids or None,
    )
    hits = []
    for res in results:
        item: dict[str, Any] = {
            "document_id": res.document_id,
            "document": res.document_title,
            "author": res.author,
            "chunk_id": res.chunk_id,
            "chapter": res.chapter,
            "section": res.section,
            "page_start": res.page_start,
            "page_end": res.page_end,
            "score": round(res.score, 4),
            "lexical": round(res.lexical_score, 4),
            "vector": round(res.vector_score, 4),
            "tier": res.tier,
            "url": res.url,
            "snippet": res.text[:800],
            "citation": res.citation,
        }
        if payload.include_full_text:
            chunk = session.get(DocumentChunk, res.chunk_id)
            item["text"] = chunk.text if chunk else ""
        hits.append(item)
    return {
        "query": payload.query,
        "hits": hits,
        "count": len(hits),
        "sources": format_sources_for_user([h["citation"] for h in hits]),
        "conflicts": (detect_conflicts(results, term=payload.query.split()[0].lower()) if payload.detect_conflicts else []),
        "store": get_vector_store().health(),
    }


@router.get("/kb/chunks/{document_id}")
def document_chunks(document_id: int, limit: int = 50, offset: int = 0, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    doc = session.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    total = int(session.execute(select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == doc.id)).scalar_one() or 0)
    rows = session.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index.asc()).limit(min(limit, 300)).offset(offset)
    ).scalars()
    return {
        "document": {"id": doc.id, "title": doc.title, "author": doc.author, "pages": doc.page_count},
        "total": total,
        "items": [
            {
                "id": c.id,
                "index": c.chunk_index,
                "chapter": c.chapter,
                "section": c.section,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "topic": c.topic,
                "quality": c.quality,
                "keywords": c.keywords or [],
                "text": c.text,
            }
            for c in rows
        ],
    }


@router.get("/kb/context")
def kb_context(query: str, topic: str = "", user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    """Exactly the context block an AI request would receive - for transparency/debugging."""
    retriever = Retriever(session)
    results = retriever.search(f"{topic} {query}".strip(), top_k=settings.retrieval_top_k)
    context, citations = build_context(results)
    return {
        "query": query,
        "context_chars": len(context),
        "context": context,
        "citations": citations,
        "sufficient": retriever.sufficient(query)[0],
        "coverage": round(retriever.coverage(query, results), 3),
    }


# --------------------------------------------------------------------------- #
#  Teacher
# --------------------------------------------------------------------------- #
class TeacherIn(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    mode: str = "explain"
    conversation_id: int | None = None
    topic_code: str = ""
    allow_web: bool | None = None
    code: str = ""
    question_id: int | None = None


@router.get("/teacher/modes")
def teacher_modes(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    from app.ai.manager import manager as ai_manager

    ai_on = ai_manager.is_configured() and user.ai_enabled
    descriptions = {
        "explain": "Intuition, definition, example, then a check question.",
        "practice": "Generate tasks for the current topic at your difficulty ladder.",
        "quiz": "Short adaptive quiz built from your library and skill graph.",
        "deep_dive": "Formal treatment, failure modes, what to learn next.",
        "socratic": "No answers - leading questions until you derive it yourself.",
        "code_review": "Static + AI review of code you paste or run in the Lab.",
        "project_mentor": "Architecture, risks and next steps; never your whole project.",
        "interview": "Technical interview turn-taking with judgement and follow-ups.",
        "exam_prep": "Readiness per exam and what to close first.",
        "review_answer": "Grade a written answer on 5 dimensions and re-test you.",
    }
    return {
        "ai_configured": ai_on,
        "modes": [
            {
                "id": mode,
                "label": teacher.MODE_LABELS.get(mode, mode.title()),
                "description": text,
                "engine": "ai+kb" if ai_on else "local+kb",
            }
            for mode, text in descriptions.items()
        ],
    }


@router.post("/teacher/chat")
def teacher_chat(payload: TeacherIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    try:
        return teacher.respond(
            session,
            user,
            text=payload.text,
            mode=payload.mode,
            conversation_id=payload.conversation_id,
            topic_code=payload.topic_code,
            allow_web=payload.allow_web,
            code=payload.code,
            answer_to_question_id=payload.question_id,
        )
    except Exception as exc:  # noqa: BLE001 - surface the real reason in the chat, do not 500 silently
        raise HTTPException(status_code=500, detail=f"Teacher error: {type(exc).__name__}: {exc}") from exc


@router.get("/teacher/conversations")
def list_conversations(limit: int = 30, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> list[dict[str, Any]]:
    return teacher.list_conversations(session, user, limit=limit)


@router.get("/teacher/conversations/{conversation_id}")
def get_conversation(conversation_id: int, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    payload = teacher.get_conversation(session, user, conversation_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return payload


@router.delete("/teacher/conversations/{conversation_id}")
def delete_conversation(conversation_id: int, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    conv = session.get(TeacherConversation, int(conversation_id))
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="conversation not found")
    session.execute(delete(TeacherMessage).where(TeacherMessage.conversation_id == conv.id))
    session.delete(conv)
    session.commit()
    return {"deleted": conversation_id}


class ExplainIn(BaseModel):
    topic_code: str = ""
    free_text: str = ""
    mode: str = "explain"
    conversation_id: int | None = None


@router.post("/teacher/explain")
def teacher_explain(payload: ExplainIn, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    """Convenience endpoint used by Learn / Roadmap cards ('Explain this topic')."""
    graph = SkillGraph(session)
    topic = payload.topic_code
    text = payload.free_text.strip()
    if not text:
        label = graph.topics[topic].name if topic in graph.topics else topic.replace("_", " ")
        text = f"Explain {label}"
    return teacher.respond(session, user, text=text, mode=payload.mode, topic_code=topic, conversation_id=payload.conversation_id)


@router.get("/teacher/pending")
def teacher_pending(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    """Last unanswered item in the most recent conversation (lets the UI resume a check question)."""
    conv = session.execute(
        select(TeacherConversation).where(TeacherConversation.user_id == user.id).order_by(TeacherConversation.updated_at.desc()).limit(1)
    ).scalars().first()
    if conv is None:
        return {}
    message = session.execute(
        select(TeacherMessage)
        .where(TeacherMessage.conversation_id == conv.id, TeacherMessage.role == "assistant", TeacherMessage.pending_item != {})
        .order_by(TeacherMessage.id.desc())
        .limit(1)
    ).scalars().first()
    if message is None or not (message.pending_item or {}):
        return {}
    pending = dict(message.pending_item)
    ids = [int(i) for i in (pending.get("question_ids") or [])]
    if ids:
        rows = session.execute(select(Question).where(Question.id.in_(ids))).scalars()
        from app.services.questions import question_payload

        pending["questions"] = [question_payload(q) for q in sorted(rows, key=lambda q: ids.index(q.id))]
    return {"conversation_id": conv.id, "mode": conv.mode, "pending": pending}

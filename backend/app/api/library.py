"""
Document library + knowledge-base endpoints (§3-§6, §41).

Upload -> parse -> clean -> chunk -> metadata -> embed -> index, with visible status so
the Library page always tells the truth about what is indexed and what is not.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session

from app.api.deps import db_dep, get_current_user
from app.config import settings
from app.db import session_scope
from app.knowledge import ingest
from app.knowledge.ingest import PROGRESS, clear_progress, get_progress
from app.knowledge.retrieval import Retriever, ensure_lexical_index, invalidate_lexical_index
from app.knowledge.vectorstore import get_vector_store
from app.knowledge.web import WebFetchError, fetch_url, guess_tier_from_url
from app.models import Document, DocumentChunk, LearningMemory, Source, User

router = APIRouter(tags=["library"])

KINDS = {"book", "docs", "paper", "course", "article", "notes"}


class DocumentPatch(BaseModel):
    title: str | None = None
    author: str | None = None
    kind: str | None = None
    tier: int | None = None
    topics: list[str] | None = None
    tags: list[str] | None = None
    notes: str | None = None
    year: int | None = None
    trust: float | None = None


class UrlIn(BaseModel):
    url: str = Field(min_length=8, max_length=1200)
    title: str = ""
    kind: str = "article"
    tier: int | None = None
    topics: list[str] = []
    tags: list[str] = []


def doc_payload(session: Session, doc: Document) -> dict[str, Any]:
    progress = get_progress(doc.id)
    return {
        "id": doc.id,
        "title": doc.title,
        "filename": doc.filename,
        "doc_type": doc.doc_type,
        "author": doc.author,
        "year": doc.year,
        "kind": doc.kind,
        "tier": doc.tier,
        "trust": doc.trust,
        "size_bytes": doc.size_bytes,
        "size_human": human_size(doc.size_bytes),
        "page_count": doc.page_count,
        "word_count": doc.word_count,
        "chunk_count": doc.chunk_count or len(doc.chunks),
        "status": doc.status,
        "error": doc.error_message or "",
        "topics": doc.topics or [],
        "tags": doc.tags or [],
        "notes": doc.notes or "",
        "url": doc.source_url or "",
        "language": doc.language,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "indexed_at": doc.indexed_at.isoformat() if doc.indexed_at else None,
        "progress": progress,
    }


def human_size(num: int) -> str:
    value = float(num or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _schedule_ingest(document_id: int) -> None:
    with session_scope() as session:
        ingest.ingest_document(session, document_id)


@router.get("/library/documents")
def list_documents(
    status: str = "",
    q: str = "",
    kind: str = "",
    limit: int = 200,
    offset: int = 0,
    user: User = Depends(get_current_user),
    session: Session = Depends(db_dep),
) -> dict[str, Any]:
    stmt = select(Document).where(or_(Document.owner_id == user.id, Document.owner_id.is_(None)))
    if status:
        stmt = stmt.where(Document.status == status)
    if kind:
        stmt = stmt.where(Document.kind == kind)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            or_(Document.title.ilike(like), Document.author.ilike(like), Document.filename.ilike(like), Document.notes.ilike(like))
        )
    total = int(session.execute(select(func.count()).select_from(stmt.subquery())).scalar_one() or 0)
    rows = session.execute(stmt.order_by(Document.id.desc()).limit(min(limit, 500)).offset(offset)).scalars()
    items = [doc_payload(session, d) for d in rows]
    counts = dict(
        session.execute(select(Document.status, func.count(Document.id)).group_by(Document.status)).all()
    )
    return {
        "total": total,
        "items": items,
        "status_counts": {k: int(v) for k, v in counts.items()},
        "chunks": int(session.execute(select(func.count(DocumentChunk.id))).scalar_one() or 0),
        "vector_store": get_vector_store().health(),
    }


@router.post("/library/upload", status_code=201)
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(""),
    author: str = Form(""),
    kind: str = Form("book"),
    tier: int = Form(1),
    topics: str = Form(""),
    tags: str = Form(""),
    wait: bool = Form(False),
    user: User = Depends(get_current_user),
    session: Session = Depends(db_dep),
) -> dict[str, Any]:
    """Multipart upload. `wait=true` indexes synchronously (tests/CLI); otherwise in the background."""
    content = await file.read()
    limit = settings.max_upload_mb * 1024 * 1024
    if len(content) > limit:
        raise HTTPException(status_code=413, detail=f"File is larger than the {settings.max_upload_mb} MB limit")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    try:
        document = ingest.create_document_from_upload(
            session,
            owner_id=user.id,
            filename=file.filename or "document.txt",
            content=content,
            title=title,
            author=author,
            kind=kind if kind in KINDS else "book",
            tier=max(1, min(5, int(tier or 1))),
            topics=[t.strip() for t in topics.split(",") if t.strip()],
            tags=[t.strip() for t in tags.split(",") if t.strip()],
        )
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    if wait:
        result = ingest.ingest_document(session, document.id)
        clear_progress(document.id)
        return {"document": doc_payload(session, document), "ingest": result.to_dict()}

    document.status = Document.STATUS_UPLOADING
    session.commit()
    background.add_task(_schedule_ingest, document.id)
    return {
        "document": doc_payload(session, document),
        "ingest": {"queued": True, "document_id": document.id, "hint": "poll GET /api/library/documents for progress"},
    }


@router.post("/library/url", status_code=201)
def add_url(
    payload: UrlIn,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    session: Session = Depends(db_dep),
) -> dict[str, Any]:
    if not settings.web_ingestion_enabled:
        raise HTTPException(status_code=403, detail="Web ingestion is disabled (WEB_INGESTION_ENABLED=false)")
    url = payload.url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Only http(s) URLs are supported")
    try:
        document = ingest.create_document_from_url(
            session,
            owner_id=user.id,
            url=url,
            title=payload.title,
            kind=payload.kind if payload.kind in KINDS else "article",
            tier=payload.tier,
            topics=payload.topics,
            tags=payload.tags,
        )
    except WebFetchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background.add_task(_schedule_ingest, document.id)
    return {
        "document": doc_payload(session, document),
        "tier": guess_tier_from_url(url),
        "ingest": {"queued": True, "document_id": document.id},
    }


@router.post("/library/url/preview")
def preview_url(payload: dict[str, Any]) -> dict[str, Any]:
    url = str(payload.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    try:
        data = fetch_url(url)
    except WebFetchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "url": data["final_url"],
        "title": data["title"],
        "chars": len(data.get("text") or ""),
        "words": len((data.get("text") or "").split()),
        "tier": guess_tier_from_url(url),
        "meta": data.get("meta") or {},
        "preview": (data.get("text") or "")[:1200],
    }


@router.get("/library/documents/{document_id}")
def get_document(document_id: int, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    doc = session.get(Document, document_id)
    if doc is None or (doc.owner_id not in (None, user.id)):
        raise HTTPException(status_code=404, detail="document not found")
    chunks = session.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index.asc()).limit(400)
    ).scalars()
    return {
        **doc_payload(session, doc),
        "chunks": [
            {
                "id": c.id,
                "index": c.chunk_index,
                "chapter": c.chapter,
                "section": c.section,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "chars": c.char_count,
                "quality": c.quality,
                "topic": c.topic,
                "keywords": c.keywords or [],
                "text": c.text,
            }
            for c in chunks
        ],
        "sources": [
            {"id": s.id, "kind": s.kind, "tier": s.tier, "url": s.url, "publisher": s.publisher, "title": s.title}
            for s in session.execute(select(Source).where(Source.document_id == doc.id)).scalars()
        ],
    }


@router.patch("/library/documents/{document_id}")
def patch_document(document_id: int, payload: DocumentPatch, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    doc = session.get(Document, document_id)
    if doc is None or (doc.owner_id not in (None, user.id)):
        raise HTTPException(status_code=404, detail="document not found")
    data = payload.model_dump(exclude_none=True)
    if "kind" in data and data["kind"] not in KINDS:
        data.pop("kind")
    if "tier" in data:
        data["tier"] = max(1, min(5, int(data["tier"])))
    for field, value in data.items():
        setattr(doc, field, value)
    session.commit()
    return doc_payload(session, doc)


@router.delete("/library/documents/{document_id}")
def delete_document(document_id: int, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    doc = session.get(Document, document_id)
    if doc is None or (doc.owner_id not in (None, user.id)):
        raise HTTPException(status_code=404, detail="document not found")
    path = doc.file_path
    get_vector_store().delete_document(doc.id)
    session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))
    session.execute(delete(Source).where(Source.document_id == doc.id))
    session.delete(doc)
    session.commit()
    clear_progress(document_id)
    if path:
        file_path = Path(path)
        try:
            if file_path.is_file() and file_path.resolve().is_relative_to(settings.upload_dir.resolve()):
                file_path.unlink(missing_ok=True)
        except OSError:
            pass
    invalidate_lexical_index()
    return {"deleted": document_id, "file_removed": bool(path)}


@router.post("/library/documents/{document_id}/index")
def index_document(
    document_id: int,
    background: BackgroundTasks,
    wait: bool = False,
    user: User = Depends(get_current_user),
    session: Session = Depends(db_dep),
) -> dict[str, Any]:
    doc = session.get(Document, document_id)
    if doc is None or (doc.owner_id not in (None, user.id)):
        raise HTTPException(status_code=404, detail="document not found")
    if doc.status == Document.STATUS_PROCESSING:
        return {"document": doc_payload(session, doc), "ingest": {"already_running": True}}
    doc.status = Document.STATUS_UPLOADING
    doc.error_message = ""
    session.commit()
    if wait:
        result = ingest.ingest_document(session, doc.id)
        return {"document": doc_payload(session, doc), "ingest": result.to_dict()}
    background.add_task(_schedule_ingest, doc.id)
    return {"document": doc_payload(session, doc), "ingest": {"queued": True}}


@router.post("/library/reindex-all")
def reindex_all(background: BackgroundTasks, wait: bool = False, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    ids = [
        int(row)
        for row in session.execute(
            select(Document.id).where(or_(Document.owner_id == user.id, Document.owner_id.is_(None)), Document.status != Document.STATUS_PROCESSING)
        ).scalars()
    ]
    if wait:
        results = []
        with session_scope() as scoped:
            for doc_id in ids:
                results.append(ingest.ingest_document(scoped, doc_id).to_dict())
        return {"queued": 0, "done": results}
    background.add_task(_reindex_many, ids)
    return {"queued": len(ids), "document_ids": ids}


def _reindex_many(ids: list[int]) -> None:
    for doc_id in ids:
        _schedule_ingest(doc_id)


@router.post("/library/embeddings/rebuild")
def rebuild_embeddings(background: BackgroundTasks, wait: bool = False, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    """Recompute vectors for all chunks (needed after changing EMBEDDING_PROVIDER / dim)."""
    if wait:
        with session_scope() as scoped:
            updated = ingest.reembed_all(scoped)
        return {"reembedded": updated, "index": get_vector_store().health()}
    background.add_task(_reembed_bg)
    return {"queued": True, "hint": "poll /api/library/stats"}


def _reembed_bg() -> None:
    with session_scope() as session:
        ingest.reembed_all(session)


@router.post("/library/vectors/rebuild")
def rebuild_vectors(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    """Rebuild the vector index from the vectors stored in SQLite (no re-embedding)."""
    rows = ingest.rebuild_vector_index(session)
    return {"indexed_vectors": rows, "store": get_vector_store().health()}


@router.get("/library/search")
def library_search(
    q: str,
    top_k: int = 10,
    document_ids: str = "",
    mode: str = "hybrid",
    user: User = Depends(get_current_user),
    session: Session = Depends(db_dep),
) -> dict[str, Any]:
    """Search *inside* the whole library (title/author match + full-text chunk match)."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="q is required")
    ids = [int(x) for x in document_ids.split(",") if x.strip().isdigit()]
    results: list[dict[str, Any]] = []
    hits: list[Any] = []
    retriever = None
    if mode in {"hybrid", "semantic"}:
        retriever = Retriever(session)
        hits = retriever.search(q, top_k=max(1, min(40, top_k)), document_ids=ids or None)
        for hit in hits:
            results.append(
                {
                    "match_type": "content",
                    "score": round(hit.score, 4),
                    "lexical": round(hit.lexical_score, 4),
                    "vector": round(hit.vector_score, 4),
                    "document_id": hit.document_id,
                    "document": hit.document_title,
                    "author": hit.author,
                    "chunk_id": hit.chunk_id,
                    "chapter": hit.chapter,
                    "section": hit.section,
                    "page_start": hit.page_start,
                    "page_end": hit.page_end,
                    "snippet": hit.text[:700],
                    "tier": hit.tier,
                    "url": hit.url,
                }
            )
    coverage = round(retriever_coverage(session, q, hits), 3) if mode in {"hybrid", "semantic"} else 0.0
    like = f"%{q}%"
    docs = session.execute(
        select(Document).where(or_(Document.title.ilike(like), Document.author.ilike(like), Document.notes.ilike(like))).limit(15)
    ).scalars()
    for doc in docs:
        results.insert(
            0,
            {
                "match_type": "document",
                "score": 1.0,
                "document_id": doc.id,
                "document": doc.title,
                "author": doc.author,
                "chunk_id": None,
                "chapter": "",
                "page_start": None,
                "snippet": (doc.notes or "")[:400] or f"{doc.chunk_count} indexed chunks",
                "tier": doc.tier,
                "url": doc.source_url or "",
            },
        )
    return {
        "query": q,
        "results": results[: max(1, top_k) + 15],
        "count": len(results),
        "coverage": coverage,
    }


def retriever_coverage(session: Session, query: str, hits: list[Any]) -> float:
    """How much of the query vocabulary the retrieved passages actually contain."""
    if not hits or not isinstance(hits[0], dict):
        return 0.0
    from app.knowledge.textutil import tokenize

    terms = set(tokenize(query))
    if not terms:
        return 1.0
    text = " ".join(h.get("snippet", "") for h in hits if isinstance(h, dict))
    found = set(tokenize(text))
    return len(terms & found) / len(terms)


@router.get("/library/stats")
def library_stats(user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    index = ensure_lexical_index(session)
    by_kind = dict(session.execute(select(Document.kind, func.count(Document.id)).group_by(Document.kind)).all())
    by_status = dict(session.execute(select(Document.status, func.count(Document.id)).group_by(Document.status)).all())
    return {
        "documents": int(session.execute(select(func.count(Document.id)).where(or_(Document.owner_id == user.id, Document.owner_id.is_(None)))).scalar_one() or 0),
        "chunks": int(session.execute(select(func.count(DocumentChunk.id))).scalar_one() or 0),
        "indexed": int(session.execute(select(func.count(Document.id)).where(Document.status == Document.STATUS_INDEXED)).scalar_one() or 0),
        "errors": int(session.execute(select(func.count(Document.id)).where(Document.status == Document.STATUS_ERROR)).scalar_one() or 0),
        "by_kind": {k: int(v) for k, v in by_kind.items()},
        "by_status": {k: int(v) for k, v in by_status.items()},
        "lexical_docs": index.n_docs,
        "lexical_terms": len(index.df),
        "vector_store": get_vector_store().health(),
        "progress": dict(PROGRESS),
    }


@router.get("/library/progress/{document_id}")
def document_progress(document_id: int, user: User = Depends(get_current_user), session: Session = Depends(db_dep)) -> dict[str, Any]:
    doc = session.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    return {"document_id": document_id, "status": doc.status, **get_progress(document_id)}

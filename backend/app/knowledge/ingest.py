"""
Ingestion pipeline:

  file/url -> parser -> clean -> chunk -> metadata -> embed -> vector DB -> KB

Runs synchronously (for tests / small files) or in a background task (for books).
Every stage records progress on the Document row so the Library UI can show
Uploading -> Processing -> Indexed / Error.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import session_scope
from app.knowledge.chunking import chunk_document, guess_topic
from app.knowledge.embeddings import get_embedder, persist_idf
from app.knowledge.extractors import ExtractedDocument, extract_file, extract_text_blob
from app.knowledge.retrieval import invalidate_lexical_index
from app.knowledge.vectorstore import get_vector_store
from app.knowledge.web import fetch_url
from app.models import Document, DocumentChunk, LearningMemory, Skill, Topic, User

logger = logging.getLogger(__name__)

_progress_lock = threading.Lock()
PROGRESS: dict[int, dict[str, Any]] = {}


@dataclass
class IngestResult:
    document_id: int
    status: str
    chunk_count: int = 0
    page_count: int = 0
    word_count: int = 0
    char_count: int = 0
    topics: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _set_progress(document_id: int, stage: str, *, percent: int, message: str = "") -> None:
    with _progress_lock:
        PROGRESS[document_id] = {
            "stage": stage,
            "percent": max(0, min(100, percent)),
            "message": message,
            "updated_at": time.time(),
        }


def get_progress(document_id: int) -> dict[str, Any]:
    with _progress_lock:
        return dict(PROGRESS.get(document_id) or {"stage": "idle", "percent": 0, "message": "", "updated_at": None})


def clear_progress(document_id: int) -> None:
    with _progress_lock:
        PROGRESS.pop(document_id, None)


def topic_vocabulary(session: Session) -> dict[str, list[str]]:
    vocab: dict[str, list[str]] = {}
    for topic in session.execute(select(Topic)).scalars():
        vocab[topic.code] = list(topic.keywords or [])
    for skill in session.execute(select(Skill)).scalars():
        bucket = vocab.setdefault(skill.code, [])
        bucket.extend(skill.keywords or [])
        if skill.topic_id:
            topic_code = session.get(Topic, skill.topic_id)
            if topic_code:
                vocab.setdefault(topic_code.code, []).extend(skill.keywords or [])
    return {code: sorted(set(terms))[:40] for code, terms in vocab.items() if terms}


def _link_document_topics(session: Session, document: Document) -> list[str]:
    """Tag the document with curriculum topics that its text actually mentions."""
    from app.knowledge.textutil import keywords

    sample = "\n".join(c.text for c in document.chunks[: max(1, len(document.chunks) or 1)])
    vocab = topic_vocabulary(session)
    if not vocab or not sample:
        return []
    scored: dict[str, int] = {}
    doc_terms = set(keywords(sample, limit=400))
    for code, terms in vocab.items():
        hits = len(doc_terms & {t.lower() for t in terms})
        if hits:
            scored[code] = hits
    ranked = sorted(scored, key=lambda c: -scored[c])[:8]
    return ranked


def ingest_document(session: Session, document_id: int, *, embed: bool = True) -> IngestResult:
    """Full pipeline for one already-persisted Document row."""
    started = time.perf_counter()
    document = session.get(Document, document_id)
    if document is None:
        return IngestResult(document_id=document_id, status="error", error="document not found")

    document.status = Document.STATUS_PROCESSING
    session.commit()
    _set_progress(document_id, "extract", percent=5)

    try:
        if document.source_url and not document.file_path:
            fetched = fetch_url(document.source_url)
            extracted = extract_text_blob(
                fetched["html"], title=document.title or fetched["title"], doc_type="url", url=document.source_url
            )
            document.title = document.title or fetched["title"][:380]
        elif document.file_path and Path(document.file_path).is_file():
            extracted = extract_file(Path(document.file_path))
        else:
            raise FileNotFoundError(f"No file or URL for document {document_id}")

        if not extracted.pages or extracted.char_count < 80:
            raise ValueError("Could not extract readable text from this file (scanned PDF without OCR text?)")

        if not document.title or document.title in {Path(document.file_path).stem}:
            document.title = (extracted.title or document.title or Path(document.file_path or "untitled").stem)[:380]
        if not document.author:
            document.author = extracted.author[:290]
        document.language = document.language or extracted.language

        _set_progress(document_id, "chunk", percent=30)
        drafts = chunk_document(extracted)
        if not drafts:
            raise ValueError("Chunking produced no usable chunks")

        _replace_chunks(session, document, drafts, extracted)
        session.commit()

        _set_progress(document_id, "embed", percent=55)
        if embed:
            _embed_document(session, document, progress=lambda pct: _set_progress(document_id, "embed", percent=pct))

        _set_progress(document_id, "tag", percent=88)
        topics = _link_document_topics(session, document)
        document.topics = topics
        document.status = Document.STATUS_INDEXED
        document.error_message = ""
        document.indexed_at = _utcnow()
        document.chunk_count = len(document.chunks)
        document.page_count = extracted.page_count or document.page_count
        document.word_count = sum(len(c.text.split()) for c in document.chunks)
        document.char_count = extracted.char_count
        session.commit()

        invalidate_lexical_index()
        _remember_corpus_growth(session, document)
        clear_progress(document_id)
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info("Indexed document %s (%s chunks, %sms)", document_id, document.chunk_count, elapsed)
        return IngestResult(
            document_id=document_id,
            status=document.status,
            chunk_count=document.chunk_count,
            page_count=document.page_count,
            word_count=document.word_count,
            char_count=document.char_count,
            topics=topics,
            elapsed_ms=elapsed,
        )
    except Exception as exc:  # noqa: BLE001 - ingestion failures must be visible, not fatal
        logger.exception("Ingestion failed for document %s", document_id)
        document = session.get(Document, document_id)
        if document is not None:
            document.status = Document.STATUS_ERROR
            document.error_message = f"{type(exc).__name__}: {exc}"[:1500]
            session.commit()
        _set_progress(document_id, "error", percent=100, message=str(exc)[:200])
        return IngestResult(document_id=document_id, status="error", error=str(exc)[:500], elapsed_ms=int((time.perf_counter() - started) * 1000))


def _utcnow() -> Any:
    from datetime import datetime

    return datetime.utcnow()


def _replace_chunks(session: Session, document: Document, drafts: list[Any], extracted: ExtractedDocument) -> None:
    session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
    session.flush()
    get_vector_store().delete_document(document.id)

    rows = []
    for draft in drafts:
        rows.append(
            {
                "document_id": document.id,
                **draft.to_db_kwargs(),
                "meta": {
                    **(draft.meta or {}),
                    "doc_type": extracted.doc_type,
                    "language": extracted.language,
                    "source_kind": document.kind,
                },
            }
        )
    session.bulk_insert_mappings(DocumentChunk, rows)
    session.flush()


def _embed_document(session: Session, document: Document, *, progress: Any | None = None) -> None:
    embedder = get_embedder()
    chunks = list(document.chunks)
    batch = max(4, settings.embedding_batch_size)
    vectors: list[tuple[int, int, np.ndarray, dict]] = []
    for start in range(0, len(chunks), batch):
        window = chunks[start : start + batch]
        matrix = embedder.embed([c.text for c in window])
        for chunk, vec in zip(window, matrix):
            chunk.embedding = np.asarray(vec, dtype="float32").tobytes()
            chunk.embedding_dim = int(vec.shape[0])
            chunk.embedding_model = getattr(embedder, "name", "unknown")
            vectors.append((chunk.id, document.id, vec, {"topic": chunk.topic or "", "tier": document.tier}))
        if progress:
            progress(55 + int(30 * min(1.0, (start + batch) / max(1, len(chunks)))))
    session.commit()

    store = get_vector_store()
    if store.name == "local-numpy":
        store.rebuild(_all_vector_rows(session))  # single write, keeps npz consistent
    else:
        store.upsert_many(vectors)

    all_texts = list(session.execute(select(DocumentChunk.text)).scalars())
    persist_idf(all_texts)


def _all_vector_rows(session: Session) -> list[tuple[int, int, np.ndarray]]:
    rows = session.execute(
        select(DocumentChunk.id, DocumentChunk.document_id, DocumentChunk.embedding, DocumentChunk.embedding_dim)
        .where(DocumentChunk.embedding.isnot(None))
    ).all()
    out: list[tuple[int, int, np.ndarray]] = []
    for chunk_id, doc_id, blob, dim in rows:
        if not blob:
            continue
        vec = np.frombuffer(blob, dtype="float32")
        expected = dim or settings.embedding_dim
        if vec.shape[0] != expected:
            continue
        out.append((int(chunk_id), int(doc_id), vec))
    return out


def rebuild_vector_index(session: Session) -> int:
    """Recreate the vector index from SQLite (after a backend change / embedder swap)."""
    rows = _all_vector_rows(session)
    store = get_vector_store()
    if store.name == "local-numpy":
        store.rebuild(rows)
    else:
        meta = {
            int(cid): {"topic": topic or "", "tier": tier}
            for cid, topic, tier in session.execute(
                select(DocumentChunk.id, DocumentChunk.topic, Document.tier).join(Document, Document.id == DocumentChunk.document_id)
            ).all()
        }
        store.upsert_many([(cid, doc_id, vec, meta.get(cid, {})) for cid, doc_id, vec in rows])
    return len(rows)


def reembed_all(session: Session, *, batch: int | None = None) -> int:
    """Recompute embeddings for every chunk with the currently configured embedder."""
    embedder = get_embedder()
    batch = batch or max(8, settings.embedding_batch_size)
    updated = 0
    ids = [r[0] for r in session.execute(select(DocumentChunk.id).order_by(DocumentChunk.id)).all()]
    vectors: list[tuple[int, int, np.ndarray, dict]] = []
    for start in range(0, len(ids), batch):
        window = ids[start : start + batch]
        chunks = list(session.execute(select(DocumentChunk).where(DocumentChunk.id.in_(window))).scalars())
        matrix = embedder.embed([c.text for c in chunks])
        for chunk, vec in zip(chunks, matrix):
            chunk.embedding = np.asarray(vec, dtype="float32").tobytes()
            chunk.embedding_dim = int(vec.shape[0])
            chunk.embedding_model = getattr(embedder, "name", "unknown")
            vectors.append((chunk.id, chunk.document_id, vec, {"topic": chunk.topic or ""}))
        updated += len(window)
        session.commit()
    store = get_vector_store()
    if store.name == "local-numpy":
        store.rebuild(_all_vector_rows(session))
    else:
        store.upsert_many(vectors)
    return updated


def _remember_corpus_growth(session: Session, document: Document) -> None:
    """Track the knowledge horizon in long-term memory (used by the Teacher prompt)."""
    user_id = document.owner_id or (session.execute(select(User.id).limit(1)).scalar() or 0)
    if not user_id:
        return
    row = session.execute(
        select(LearningMemory).where(
            LearningMemory.user_id == user_id, LearningMemory.category == "fact", LearningMemory.key == "corpus"
        )
    ).scalar_one_or_none()
    total_chunks = int(session.execute(select(func.count()).select_from(DocumentChunk)).scalar() or 0)
    total_docs = int(session.execute(select(func.count()).select_from(Document)).scalar() or 0)
    payload = {"documents": total_docs, "chunks": total_chunks, "last_document": document.title}
    if row is None:
        session.add(
            LearningMemory(user_id=user_id, category="fact", key="corpus", value=f"{total_docs} documents indexed", payload=payload)
        )
    else:
        row.payload = payload
        row.value = f"{total_docs} documents / {total_chunks} chunks indexed"
    session.commit()


# --------------------------------------------------------------------------- #
#  Convenience wrappers used by the API + seed
# --------------------------------------------------------------------------- #
def create_document_from_upload(
    session: Session,
    *,
    owner_id: int,
    filename: str,
    content: bytes,
    title: str = "",
    author: str = "",
    kind: str = "book",
    tier: int = 1,
    topics: list[str] | None = None,
    tags: list[str] | None = None,
) -> Document:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    safe = re_slug(filename or "document")
    path = settings.upload_dir / f"{int(time.time() * 1000)}_{safe}"
    path.write_bytes(content)
    ext = Path(safe).suffix.lower().lstrip(".") or "txt"
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported file type '.{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXT))}")
    document = Document(
        owner_id=owner_id,
        title=(title or Path(safe).stem)[:380],
        doc_type=ext,
        author=author[:290],
        filename=filename[:380],
        file_path=str(path),
        size_bytes=len(content),
        kind=kind,
        tier=tier,
        trust=1.0 if tier == 1 else 0.9,
        topics=topics or [],
        tags=tags or [],
        status=Document.STATUS_UPLOADING,
        ingest_options={"original_name": filename, "bytes": len(content)},
    )
    session.add(document)
    session.commit()
    session.refresh(document)
    return document


def create_document_from_url(
    session: Session,
    *,
    owner_id: int,
    url: str,
    title: str = "",
    kind: str = "article",
    tier: int | None = None,
    topics: list[str] | None = None,
    tags: list[str] | None = None,
) -> Document:
    from app.knowledge.web import _assert_public_host, guess_tier_from_url

    _assert_public_host(url)  # reject non-http(s) and private/loopback targets before creating anything
    resolved_tier = tier or guess_tier_from_url(url)
    document = Document(
        owner_id=owner_id,
        title=(title or url)[:380],
        doc_type="url",
        source_url=url[:1200],
        kind=kind,
        tier=resolved_tier,
        trust=0.75 if resolved_tier >= 4 else 0.95,
        topics=topics or [],
        tags=tags or [],
        status=Document.STATUS_UPLOADING,
        ingest_options={"url": url},
    )
    session.add(document)
    session.commit()
    session.refresh(document)
    return document


def ingest_file_on_disk(
    session: Session,
    *,
    owner_id: int | None,
    path: Path,
    title: str = "",
    kind: str = "docs",
    tier: int = 1,
    tags: list[str] | None = None,
    author: str = "",
) -> Document:
    document = Document(
        owner_id=owner_id,
        title=(title or path.stem)[:380],
        doc_type=path.suffix.lower().lstrip(".") or "md",
        author=author[:290],
        filename=path.name,
        file_path=str(path),
        size_bytes=path.stat().st_size if path.is_file() else 0,
        kind=kind,
        tier=tier,
        topics=[],
        tags=tags or [],
        status=Document.STATUS_UPLOADING,
    )
    session.add(document)
    session.commit()
    session.refresh(document)
    return document


def ingest_blob(session: Session, *, owner_id: int | None, name: str, text: str, doc_type: str = "md",
                kind: str = "docs", tier: int = 1, tags: list[str] | None = None) -> IngestResult:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    path = settings.upload_dir / f"inline_{re_slug(name)}"
    path.write_text(text, encoding="utf-8")
    document = Document(
        owner_id=owner_id,
        title=name[:380],
        doc_type=doc_type if doc_type in SUPPORTED_EXT else "md",
        filename=path.name,
        file_path=str(path),
        size_bytes=len(text),
        kind=kind,
        tier=tier,
        tags=tags or [],
        status=Document.STATUS_UPLOADING,
    )
    session.add(document)
    session.commit()
    session.refresh(document)
    return ingest_document(session, document.id)


SUPPORTED_EXT = {"pdf", "epub", "txt", "md", "markdown", "html", "htm", "rst", "org", "csv", "json", "py", "ipynb"}


def re_slug(value: str) -> str:
    import re as _re

    slug = _re.sub(r"[^A-Za-z0-9._\-]+", "_", (value or "document").strip())
    slug = _re.sub(r"_+", "_", slug).strip("._-") or "document"
    return slug[:120]


def ingest_in_background(document_id: int) -> None:  # pragma: no cover - threaded
    with session_scope() as session:
        ingest_document(session, document_id)

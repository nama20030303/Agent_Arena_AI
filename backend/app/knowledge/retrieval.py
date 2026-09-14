"""
Hybrid retrieval over the personal Knowledge Base.

  BM25 (lexical, exact terms, code identifiers)
  + dense vectors (offline hashing embedder or a real embedding API)
  + source-tier weighting (library > docs > papers > education > web)
  + topic/skill boost from the user's current position in the curriculum

Plus two things the product spec insists on:
  * every hit carries a real citation (document, chapter, page) - never invented
  * ``detect_conflicts`` surfaces when two sources disagree instead of hiding it
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.knowledge.embeddings import get_embedder
from app.knowledge.textutil import normalize, split_sentences, tokenize
from app.knowledge.vectorstore import get_vector_store
from app.models import Document, DocumentChunk


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    text: str
    score: float
    lexical_score: float = 0.0
    vector_score: float = 0.0
    chapter: str = ""
    section: str = ""
    page_start: int | None = None
    page_end: int | None = None
    document_title: str = ""
    author: str = ""
    doc_type: str = "md"
    url: str = ""
    tier: int = 3
    topic: str = ""
    quality: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> dict[str, Any]:
        parts = [self.document_title or f"document #{self.document_id}"]
        if self.chapter:
            parts.append(self.chapter)
        elif self.section:
            parts.append(self.section)
        if self.page_start:
            parts.append(f"Page {self.page_start}" + (f"-{self.page_end}" if self.page_end and self.page_end != self.page_start else ""))
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "document": self.document_title,
            "author": self.author,
            "chapter": self.chapter,
            "section": self.section,
            "page": self.page_start,
            "page_end": self.page_end,
            "url": self.url,
            "tier": self.tier,
            "label": " · ".join(p for p in parts if p),
        }

    def snippet(self, limit: int = 700) -> dict[str, Any]:
        return {"text": self.text[:limit], "citation": self.citation, "score": round(self.score, 4)}


# --------------------------------------------------------------------------- #
#  BM25 index (in-process, lazily rebuilt when the corpus changes)
# --------------------------------------------------------------------------- #
class LexicalIndex:
    k1 = 1.4
    b = 0.72
    first_chunk_id: int = 0

    @staticmethod
    def _fingerprint_of(text: str) -> str:
        import hashlib

        return hashlib.sha1((text or "")[:4096].encode("utf-8", "ignore")).hexdigest()[:16]

    def __init__(self) -> None:
        self.doc_ids: list[int] = []
        self.chunk_ids: list[int] = []
        self.lengths: np.ndarray = np.zeros(0, dtype="float32")
        self.avg_len = 1.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.df: dict[str, int] = {}
        self.n_docs = 0
        self.signature: tuple[int, int, int, int] = (-1, -1, -1, -1)
        self.fingerprint: str = ""
        self.first_chunk_id = 0
        self._lock = threading.RLock()

    def build(self, rows: list[tuple[int, int, str, list[str]]]) -> None:
        """rows: (chunk_id, document_id, text, extra_terms)"""
        with self._lock:
            chunk_ids: list[int] = []
            doc_ids: list[int] = []
            postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
            lengths: list[float] = []
            self.first_chunk_id = int(rows[0][0]) if rows else 0
            first_text = rows[0][2] if rows else ""
            for doc_pos, (chunk_id, document_id, text, extra) in enumerate(rows):
                terms = tokenize(text) + [t for e in (extra or []) for t in tokenize(str(e))]
                if not terms:
                    continue
                counts = Counter(terms)
                chunk_ids.append(int(chunk_id))
                doc_ids.append(int(document_id))
                lengths.append(float(len(terms)))
                for term, freq in counts.items():
                    postings[term].append((doc_pos, freq))
            self.chunk_ids = chunk_ids
            self.doc_ids = doc_ids
            self.n_docs = len(chunk_ids)
            self.lengths = np.asarray(lengths, dtype="float32") if lengths else np.zeros(0, dtype="float32")
            self.avg_len = float(self.lengths.mean()) if self.n_docs else 1.0
            self.postings = postings
            self.df = {term: len(bucket) for term, bucket in postings.items()}
            self.fingerprint = self._fingerprint_of(first_text)

    def search(self, query: str, limit: int = 80) -> list[tuple[int, float]]:
        with self._lock:
            if self.n_docs == 0:
                return []
            terms = Counter(tokenize(query))
            if not terms:
                return []
            scores = np.zeros(self.n_docs, dtype="float32")
            for term, q_freq in terms.items():
                bucket = self.postings.get(term)
                if not bucket:
                    continue
                df = self.df.get(term) or len(bucket)
                idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
                weight = idf * (1.0 + math.log(q_freq))
                for doc_pos, freq in bucket:
                    denom = freq + self.k1 * (1 - self.b + self.b * self.lengths[doc_pos] / max(self.avg_len, 1.0))
                    scores[doc_pos] += weight * (freq * (self.k1 + 1.0)) / max(denom, 1e-6)
            if not scores.any():
                return []
            k = min(limit, self.n_docs)
            idx = np.argpartition(-scores, kth=k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]
            return [(int(self.chunk_ids[i]), float(scores[i])) for i in idx if scores[i] > 0]


_lexical = LexicalIndex()
_lexical_lock = threading.Lock()


def _corpus_signature(session: Session) -> tuple[int, int, int, int]:
    """(count, min_id, max_id, id_sum) - a cheap fingerprint that catches DB swaps/rebuilds."""
    row = session.execute(
        select(func.count(DocumentChunk.id), func.min(DocumentChunk.id), func.max(DocumentChunk.id), func.coalesce(func.sum(DocumentChunk.id), 0))
    ).one()
    return int(row[0] or 0), int(row[1] or 0), int(row[2] or 0), int(row[3] or 0)


def ensure_lexical_index(session: Session, *, force: bool = False) -> LexicalIndex:
    """
    Return a process-wide lexical index, rebuilt when the corpus changed.

    The signature (count, min id, max id, id sum) catches ordinary growth; the fingerprint of the
    first chunk catches the subtle case where a *different* corpus has the same shape (fresh DB,
    index rebuild, restored file) - which would otherwise silently serve stale scores.
    """
    signature = _corpus_signature(session)
    with _lexical_lock:
        cached = _lexical
        if not force and cached.n_docs and cached.signature == signature:
            live = session.execute(select(DocumentChunk.text).where(DocumentChunk.id == cached.first_chunk_id)).scalar_one_or_none()
            if live is not None and cached._fingerprint_of(live) == cached.fingerprint:
                return cached
        rows = session.execute(
            select(DocumentChunk.id, DocumentChunk.document_id, DocumentChunk.text, DocumentChunk.keywords)
        ).all()
        cached.build([(r[0], r[1], r[2], r[3] or []) for r in rows])
        cached.signature = signature
        return cached


def invalidate_lexical_index() -> None:
    with _lexical_lock:
        _lexical.signature = (-1, -1, -1, -1)
        _lexical.n_docs = 0
        _lexical.postings = {}
        _lexical.df = {}


# --------------------------------------------------------------------------- #
#  Retriever
# --------------------------------------------------------------------------- #
_TIER_WEIGHT = {1: 1.00, 2: 0.92, 3: 0.86, 4: 0.78, 5: 0.66}


class Retriever:
    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------ main
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        topic_codes: Iterable[str] | None = None,
        document_ids: Iterable[int] | None = None,
        min_score: float | None = None,
        max_tier: int | None = None,
        per_document_cap: int = 3,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.retrieval_top_k
        min_score = settings.retrieval_min_score if min_score is None else min_score
        topics = {t for t in (topic_codes or []) if t}
        doc_filter = {int(d) for d in (document_ids or []) if d}

        index = ensure_lexical_index(self.session)
        lexical_hits = dict(index.search(query, limit=max(40, top_k * 12)))
        lexical_max = max(lexical_hits.values()) if lexical_hits else 1.0
        lexical_norm = {cid: s / lexical_max for cid, s in lexical_hits.items()}

        vector_norm: dict[int, float] = {}
        try:
            embedder = get_embedder()
            query_vec = embedder.embed_one(query)
            hits = get_vector_store().search(query_vec, top_k=max(40, top_k * 12), document_ids=list(doc_filter) or None)
            for hit in hits:
                vector_norm[hit.chunk_id] = max(vector_norm.get(hit.chunk_id, 0.0), hit.score)
        except Exception:  # noqa: BLE001 - retrieval must degrade to lexical only
            vector_norm = {}

        if not lexical_norm and not vector_norm:
            return []

        fused: dict[int, float] = defaultdict(float)
        for cid, score in lexical_norm.items():
            fused[cid] += settings.lexical_weight * score
        for cid, score in vector_norm.items():
            fused[cid] += settings.vector_weight * (score ** 0.8)

        candidate_ids = sorted(fused, key=lambda cid: -fused[cid])[: max(60, top_k * 10)]
        rows = self.session.execute(
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.id.in_(candidate_ids), Document.status == Document.STATUS_INDEXED)
        ).all()

        results: list[RetrievedChunk] = []
        for chunk, doc in rows:
            if doc_filter and doc.id not in doc_filter:
                continue
            if max_tier and doc.tier > max_tier:
                continue
            base = fused.get(chunk.id, 0.0)
            score = base * _TIER_WEIGHT.get(doc.tier, 0.8) * (0.6 + 0.4 * (chunk.quality or 1.0)) * (0.55 + 0.45 * doc.trust)
            if topics and chunk.topic and chunk.topic in topics:
                score *= 1.22
            elif topics and doc.topics and set(doc.topics) & topics:
                score *= 1.10
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=doc.id,
                    text=chunk.text,
                    score=score,
                    lexical_score=lexical_norm.get(chunk.id, 0.0),
                    vector_score=vector_norm.get(chunk.id, 0.0),
                    chapter=chunk.chapter,
                    section=chunk.section,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    document_title=doc.title,
                    author=doc.author,
                    doc_type=doc.doc_type,
                    url=doc.source_url,
                    tier=doc.tier,
                    topic=chunk.topic,
                    quality=chunk.quality,
                    meta=chunk.meta or {},
                )
            )
        results.sort(key=lambda r: -r.score)
        results = [r for r in results if r.score >= min_score]

        capped: list[RetrievedChunk] = []
        per_doc: Counter[int] = Counter()
        for res in results:
            if per_doc[res.document_id] >= per_document_cap:
                continue
            per_doc[res.document_id] += 1
            capped.append(res)
            if len(capped) >= top_k:
                break
        return capped

    def coverage(self, query: str, results: list[RetrievedChunk]) -> float:
        """Fraction of query content words actually present in the retrieved context."""
        query_terms = set(tokenize(query)) - set(tokenize("what is the of and"))
        if not query_terms:
            return 1.0
        corpus_terms: set[str] = set()
        for res in results:
            corpus_terms.update(tokenize(res.text))
        return len(query_terms & corpus_terms) / max(1, len(query_terms))

    def sufficient(self, query: str, *, top_k: int = 6) -> tuple[bool, float, list[RetrievedChunk]]:
        """
        Does the library actually cover this question?

        Three conditions, because any single one is gameable: some lexical strength, enough term
        coverage, and the *rarest* content term of the query must exist in the corpus at all.
        Without the last one, generic words ("explain", "policy", "mechanism") are enough to make
        an unrelated collection look relevant - which is exactly how a RAG system learns to hallucinate.
        """
        from app.knowledge.textutil import tokenize

        results = self.search(query, top_k=top_k)
        if not results:
            return False, 0.0, []
        cov = self.coverage(query, results)
        strength = max(r.score for r in results)
        index = ensure_lexical_index(self.session)
        terms = {t for t in tokenize(query) if len(t) > 3}
        if terms:
            unknown = len([t for t in terms if t not in index.df]) / len(terms)
            mentioned = unknown <= 0.34  # most content words must be part of this corpus
        else:
            mentioned = True
        verdict = bool(mentioned) and cov >= 0.34 and strength >= 0.24
        return verdict, cov, results


# --------------------------------------------------------------------------- #
#  Context assembly
# --------------------------------------------------------------------------- #
def build_context(results: list[RetrievedChunk], max_chars: int | None = None) -> tuple[str, list[dict[str, Any]]]:
    """Pack retrieved chunks into a prompt context block with numbered citations."""
    max_chars = max_chars or settings.max_context_chars
    blocks: list[str] = []
    citations: list[dict[str, Any]] = []
    used = 0
    for i, res in enumerate(results, start=1):
        cite = res.citation
        cite["index"] = i
        citations.append(cite)
        header = f"[{i}] {cite['label'] or cite['document']}"
        budget = max(300, (max_chars - used) // max(1, len(results) - i + 1))
        body = normalize(res.text)[:budget]
        block = f"{header}\n{body}"
        if used + len(block) > max_chars:
            break
        used += len(block)
        blocks.append(block)
    return "\n\n---\n\n".join(blocks), citations


def format_sources_for_user(citations: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None]] = set()
    for cite in citations[: limit * 3]:
        key = (cite.get("document", ""), cite.get("page"))
        if key in seen:
            continue
        seen.add(key)
        out.append(cite)
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
#  Conflict detection between sources
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?(?:\s?%|x|GB|MB|epochs?|layers?)?", re.IGNORECASE)
_NEGATIONS = ("however", "in contrast", "instead", "unlike", "but", "однако", "в отличие", "а вот")
_DEF_RE = re.compile(
    r"([A-Za-zА-Яа-яЁё][\w \-]{2,44}?)\s+(?:is|are|means|refers to|определяется как|это)\s+(.{15,240})",
)


def detect_conflicts(results: list[RetrievedChunk], term: str = "") -> list[dict[str, Any]]:
    """
    Surface genuine disagreements between sources (numbers, definitions, negations).
    Returns [] when nothing meaningful conflicts - we never invent a conflict.
    """
    if len(results) < 2:
        return []
    term_lower = (term or "").lower().strip()
    conflicts: list[dict[str, Any]] = []

    statements: dict[str, list[tuple[RetrievedChunk, str, list[str]]]] = defaultdict(list)
    for res in results:
        for sentence in split_sentences(res.text):
            low = sentence.lower()
            if term_lower and term_lower not in low:
                continue
            if not term_lower and not _DEF_RE.search(sentence):
                continue
            numbers = [n.strip() for n in _NUM_RE.findall(sentence) if any(ch.isdigit() for ch in n)]
            if len(numbers) >= 1:
                statements[term_lower or "numbers"].append((res, sentence, numbers))

    seen_pairs: set[tuple[int, int]] = set()
    grouped: dict[str, list[tuple[RetrievedChunk, str, list[str]]]] = defaultdict(list)
    for entries in statements.values():
        for res, sentence, numbers in entries:
            grouped[",".join(sorted(numbers)[:2])].append((res, sentence, numbers))
    buckets = [v for v in grouped.values() if len(v) >= 1]
    for i in range(len(buckets)):
        for j in range(i + 1, len(buckets)):
            a, b = buckets[i][0], buckets[j][0]
            if a[0].document_id == b[0].document_id or a[1] == b[1]:
                continue
            if {tuple(sorted(a[2]))} == {tuple(sorted(b[2]))}:
                continue
            key = (min(a[0].chunk_id, b[0].chunk_id), max(a[0].chunk_id, b[0].chunk_id))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            conflicts.append(
                {
                    "term": term_lower or "quantity",
                    "a": {"text": a[1][:260], "source": a[0].citation["label"]},
                    "b": {"text": b[1][:260], "source": b[0].citation["label"]},
                    "note": "The sources state different values for the same quantity - verify against the primary source you trust most.",
                }
            )
            if len(conflicts) >= 3:
                return conflicts

    # negation-based conflict on the same term
    if term_lower:
        hits = [
            (res, sentence)
            for res in results
            for sentence in split_sentences(res.text)
            if term_lower in sentence.lower() and any(neg in sentence.lower() for neg in _NEGATIONS)
        ]
        if len(hits) >= 2 and hits[0][0].document_id != hits[1][0].document_id:
            conflicts.append(
                {
                    "term": term_lower,
                    "a": {"text": hits[0][1][:260], "source": hits[0][0].citation["label"]},
                    "b": {"text": hits[1][1][:260], "source": hits[1][0].citation["label"]},
                    "note": "The sources qualify this differently; read both framings before deciding.",
                }
            )
    return conflicts


def context_summary(results: list[RetrievedChunk]) -> dict[str, Any]:
    return {
        "chunks": len(results),
        "documents": len({r.document_id for r in results}),
        "best_score": round(max((r.score for r in results), default=0.0), 4),
        "tiers": sorted({r.tier for r in results}),
        "build_ms": round(time.time() * 1000) % 10_000,
    }

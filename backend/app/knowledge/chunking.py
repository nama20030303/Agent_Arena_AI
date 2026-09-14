"""
Chunking: clean text -> semantically coherent chunks that carry page/chapter metadata.

Strategy (structure first, then size):
  1. rebuild a linear stream of blocks, remembering document position (page index)
  2. detect headings (markdown, ``@@H2 … @@`` markers emitted by the HTML parser,
     "Chapter 4: …" lines) and group blocks into sections
  3. pack sections into chunks of ~CHUNK_MAX_CHARS, never crossing a page more
     than needed, with a small sentence-level overlap
  4. each chunk keeps: document_id, chunk_index, chapter, section, heading path,
     page_start/page_end, topic guess, keywords
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.knowledge.extractors import ExtractedDocument
from app.knowledge.textutil import content_hash, keywords, normalize, split_sentences

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.{2,120})$")
_HTML_HEADING_RE = re.compile(r"@@H([1-6])\s(.{2,140}?)\s?@@")
_NUMBERED_HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\s+([A-ZА-ЯЁ][^.]{2,90})\.?\s*$")
_CHAPTER_LINE_RE = re.compile(
    r"^(Chapter|CHAPTER|Глава|Part|Appendix|Section)\s+([IVXLCDM\d]+(?:\.\d+)?)\s*[:.–-]?\s*(.{0,80})$"
)
_BULLET_RE = re.compile(r"^\s*([*\-•‣]|\d+[.)])\s+")


@dataclass
class ChunkDraft:
    text: str
    chunk_index: int
    chapter: str = ""
    section: str = ""
    heading_path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    keywords: list[str] = field(default_factory=list)
    char_count: int = 0
    token_count: int = 0
    quality: float = 1.0
    content_hash: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_db_kwargs(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "chapter": self.chapter,
            "section": self.section,
            "heading_path": self.heading_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "keywords": self.keywords,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "quality": self.quality,
            "meta": self.meta,
        }


@dataclass
class _Block:
    text: str
    page_index: int
    page_number: int | None
    chapter: str
    heading: tuple[int, str] | None = None
    is_code: bool = False


def _clean_block(raw: str) -> tuple[str, tuple[int, str] | None, bool]:
    """Return (text, heading(level,title), is_code)."""
    text = raw.rstrip()
    if not text:
        return "", None, False
    is_code = False
    m = _MD_HEADING_RE.match(text.strip())
    if m:
        return text.strip(), (len(m.group(1)), m.group(2).strip().rstrip("# ").strip()), False
    m = _HTML_HEADING_RE.search(text)
    if m:
        level, title = int(m.group(1)), m.group(2).strip()
        rest = _HTML_HEADING_RE.sub("", text).strip()
        return rest, (level, title), False
    m = _CHAPTER_LINE_RE.match(text.strip())
    if m and len(text.strip()) < 110:
        title = (m.group(3) or "").strip()
        label = f"{m.group(1).title()} {m.group(2)}"
        return text.strip(), (1, f"{label}: {title}".strip(": ")), False
    m = _NUMBERED_HEADING_RE.match(text.strip())
    if m and len(text.strip()) < 90:
        return text.strip(), (2, f"{m.group(1)} {m.group(2)}"), False
    if text.lstrip().startswith("```"):
        is_code = True
    return text, None, is_code


def _blocks_from_document(document: ExtractedDocument) -> list[_Block]:
    blocks: list[_Block] = []
    for page in document.pages:
        if not page.text:
            continue
        # split on blank lines first; keep code fences intact
        for segment in re.split(r"\n\s*\n", page.text):
            segment = normalize(segment)
            if not segment:
                continue
            text, heading, is_code = _clean_block(segment)
            blocks.append(
                _Block(
                    text=text,
                    page_index=page.index,
                    page_number=page.page_number,
                    chapter=page.chapter,
                    heading=heading,
                    is_code=is_code,
                )
            )
    return blocks


def _quality_score(text: str) -> float:
    """Heuristic junk detector: OCR garbage and boilerplate get down-weighted."""
    if not text:
        return 0.0
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    words = text.split()
    if not words:
        return 0.0
    avg_len = sum(len(w) for w in words) / len(words)
    ratio_alpha = letters / max(1, len(text))
    score = 1.0
    if avg_len < 2.2:
        score -= 0.35
    if ratio_alpha < 0.55:
        score -= 0.3
    if digits / max(1, len(text)) > 0.35:
        score -= 0.2
    if len(text) < 120:
        score -= 0.15
    # repeated single-character lines => broken extraction
    if text.count("  ") / max(1, len(text)) > 0.05:
        score -= 0.1
    return round(max(0.15, min(1.0, score)), 3)


def chunk_document(
    document: ExtractedDocument,
    *,
    max_chars: int | None = None,
    min_chars: int | None = None,
    overlap_chars: int | None = None,
) -> list[ChunkDraft]:
    max_chars = max_chars or settings.chunk_max_chars
    min_chars = min_chars or settings.chunk_min_chars
    overlap_chars = overlap_chars if overlap_chars is not None else settings.chunk_overlap_chars

    blocks = _blocks_from_document(document)
    if not blocks:
        return []

    chunks: list[ChunkDraft] = []
    heading_stack: list[tuple[int, str]] = []
    buf: list[_Block] = []
    buf_len = 0
    chapter = ""

    def current_section() -> str:
        return heading_stack[-1][1] if heading_stack else (chapter or document.title)

    def flush(force: bool = False) -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        text = "\n\n".join(b.text for b in buf if b.text).strip()
        if len(text) < min_chars and not force:
            # keep it: single short blocks (lists, formulas) are still useful knowledge
            if len(text) < 40:
                buf, buf_len = [], 0
                return
        pages = [b.page_number for b in buf if b.page_number is not None]
        # overlap: prepend tail sentences of previous chunk
        if chunks and overlap_chars > 0 and text:
            prev = chunks[-1]
            tail_sentences = split_sentences(prev.text)[-3:]
            tail = " ".join(tail_sentences)
            if tail and len(tail) <= overlap_chars * 2 and content_hash(tail) not in {c.meta.get("hash") for c in chunks[-1:]}:
                text = (tail + "\n\n" + text).strip()
        hashed = content_hash(text)
        chunks.append(
            ChunkDraft(
                text=text,
                chunk_index=len(chunks),
                chapter=chapter,
                section=current_section(),
                heading_path=[h[1] for h in heading_stack][-3:],
                page_start=pages[0] if pages else None,
                page_end=pages[-1] if pages else None,
                keywords=keywords(text, limit=12),
                char_count=len(text),
                token_count=max(1, int(len(text) / 4)),
                quality=_quality_score(text),
                content_hash=hashed,
                meta={"hash": hashed, "block_pages": pages[:12]},
            )
        )
        buf, buf_len = [], 0

    for block in blocks:
        if block.chapter and block.chapter != chapter:
            chapter = block.chapter
            flush()
            heading_stack = []
        if block.heading:
            level, title = block.heading
            if len(title) > 140:
                title = title[:137] + "..."
            # a heading ends the current chunk unless it is the very first one
            if buf_len > max_chars * 0.35:
                flush()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            if level <= 2:
                flush()
            continue
        buf.append(block)
        buf_len += len(block.text) + 2
        if buf_len >= max_chars:
            # prefer to break at a paragraph or sentence boundary
            if buf_len > max_chars * 1.45 and len(buf) > 1:
                last = buf.pop()
                flush()
                buf.append(last)
                buf_len = len(last.text)
            else:
                flush()
    flush(force=True)

    # drop exact duplicates within the document
    seen: set[str] = set()
    unique: list[ChunkDraft] = []
    for chunk in chunks:
        if chunk.content_hash in seen:
            continue
        seen.add(chunk.content_hash)
        unique.append(chunk)
    for i, chunk in enumerate(unique):
        chunk.chunk_index = i
    return unique


def guess_topic(text: str, vocabulary: dict[str, list[str]] | None = None) -> str:
    """Map chunk keywords to a topic code using the skill/topic keyword vocabulary."""
    if not vocabulary:
        return ""
    terms = set(keywords(text, limit=25))
    best, best_hits = "", 0
    for topic_code, topic_terms in vocabulary.items():
        hits = len(terms & set(topic_terms))
        if hits > best_hits:
            best, best_hits = topic_code, hits
    return best if best_hits >= 1 else ""

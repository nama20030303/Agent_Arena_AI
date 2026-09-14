"""
Parsers: File -> structured text with positional metadata (chapter/page/section).

Supported: PDF, EPUB, Markdown, HTML, TXT (and any text-ish file as a fallback).
Everything here is pure-python and works offline.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from app.knowledge.textutil import detect_chapter, normalize, strip_boilerplate

SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".txt", ".md", ".markdown", ".html", ".htm", ".rst", ".org", ".csv"}


@dataclass
class RawPage:
    index: int
    text: str
    page_number: int | None = None
    page_label: str = ""
    chapter: str = ""
    section: str = ""
    headings: list[str] = field(default_factory=list)


@dataclass
class ExtractedDocument:
    pages: list[RawPage]
    title: str = ""
    author: str = ""
    language: str = "en"
    doc_type: str = "txt"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)


# --------------------------------------------------------------------------- #
#  HTML / EPUB helpers
# --------------------------------------------------------------------------- #
class _TextExtractor(HTMLParser):
    """Stdlib HTML -> text, preserving block boundaries and heading structure."""

    SKIP = {"script", "style", "noscript", "svg", "canvas", "form", "nav", "footer", "iframe"}
    BLOCK = {
        "p", "div", "br", "li", "ul", "ol", "table", "tr", "td", "th", "pre", "blockquote",
        "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "header", "main", "figure",
        "figcaption", "dt", "dd", "code", "hr",
    }
    HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._current_heading: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self.HEADINGS:
            self._current_heading = self.HEADINGS[tag]
            self.parts.append(f"\n@@H{self._current_heading} ")
        elif tag in self.BLOCK:
            self.parts.append("\n" if tag != "code" else " ")
        if tag == "pre":
            self.parts.append("```python\n")
        if tag == "img":
            alt = dict(attrs).get("alt")
            if alt:
                self.parts.append(f" [image: {alt}] ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "pre":
            self.parts.append("\n```")
            return
        if tag in self.HEADINGS:
            if self._current_heading:
                self.parts.append(" @@\n")
                self._current_heading = None
            return
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self.parts.append(data.replace("\xa0", " "))

    @property
    def text(self) -> str:
        return "".join(self.parts)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed html must not kill ingestion
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<[^>]+>", "\n", html)
        return normalize(html)
    return parser.text


def strip_html_to_text(html: str) -> str:
    return normalize(re.sub(r"<[^>]+>", " ", html))


# --------------------------------------------------------------------------- #
#  PDF
# --------------------------------------------------------------------------- #
def extract_pdf(path: Path) -> ExtractedDocument:
    from pypdf import PdfReader  # imported lazily so the app boots without the dep

    reader = PdfReader(str(path))
    meta: dict[str, Any] = {}
    title = author = ""
    try:
        raw_meta = reader.metadata or {}
        meta = {str(k).lstrip("/"): str(v) for k, v in raw_meta.items()}
        title = meta.get("Title", "").strip()
        author = meta.get("Author", "").strip()
    except Exception:  # noqa: BLE001
        meta = {}

    outline_map: dict[int, str] = {}
    try:
        for item in reader.outline or []:
            if isinstance(item, list):  # nested outline level - flatten it
                stack = list(item)
            else:
                stack = [item]
            for entry in stack:
                try:
                    pg = reader.get_destination_page_number(entry)
                    label = str(getattr(entry, "title", "") or "").strip()
                    if pg is not None and label and pg not in outline_map:
                        outline_map[pg] = re.sub(r"\s+", " ", label)
                except Exception:  # noqa: BLE001 - broken bookmark must not stop ingestion
                    continue
    except Exception:  # noqa: BLE001
        outline_map = {}

    pages: list[RawPage] = []
    current_chapter = ""
    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - a broken page must not break the book
            raw = ""
        text = strip_boilerplate(normalize(raw))
        first_line = text.split("\n", 1)[0][:120] if text else ""
        detected = detect_chapter(first_line) if first_line else None
        if detected:
            current_chapter = f"{detected[0]}: {detected[1]}".strip(": ")
        elif i in outline_map:
            current_chapter = outline_map[i]
        elif first_line and len(first_line) < 70 and text.count("\n") < 4 and i < 8:
            # front-matter title page: keep it as the chapter label
            current_chapter = first_line
        pages.append(
            RawPage(
                index=i,
                text=text,
                page_number=i + 1,
                page_label=str(meta.get("Label", i + 1)),
                chapter=current_chapter,
                headings=[],
            )
        )
    return ExtractedDocument(
        pages=pages,
        title=title or path.stem,
        author=author,
        doc_type="pdf",
        language=_guess_language(" ".join(p.text[:400] for p in pages[:6])),
        metadata={"producer": meta.get("Producer", ""), "outline": outline_map},
    )


# --------------------------------------------------------------------------- #
#  EPUB
# --------------------------------------------------------------------------- #
def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def extract_epub(path: Path) -> ExtractedDocument:
    title, author, desc = path.stem, "", ""
    pages: list[RawPage] = []
    with zipfile.ZipFile(path) as zf:
        names = {n: n for n in zf.namelist()}
        try:
            container = zf.read("META-INF/container.xml").decode("utf-8", "ignore")
            opf_path = re.search(r'full-path="([^"]+)"', container).group(1)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            opf_path = next((v for k, v in names.items() if k.endswith(".opf")), "")
        manifest: dict[str, str] = {}
        spine: list[str] = []
        base_dir = str(Path(opf_path).parent) if opf_path else ""
        if opf_path and opf_path in names:
            try:
                root = ET.fromstring(zf.read(opf_path).decode("utf-8", "ignore"))
                for el in root.iter():
                    tag = _localname(el.tag)
                    if tag == "title" and (el.text or "").strip():
                        title = el.text.strip()
                    elif tag == "creator" and (el.text or "").strip():
                        author = author or el.text.strip()
                    elif tag == "description" and (el.text or "").strip():
                        desc = el.text.strip()
                    elif tag == "item":
                        manifest[el.get("id", "")] = el.get("href", "")
                    elif tag == "itemref":
                        spine.append(el.get("idref", ""))
            except ET.ParseError:
                pass
        hrefs = [manifest[i] for i in spine if i in manifest] or [
            v for k, v in names.items() if k.endswith((".xhtml", ".html", ".htm"))
        ]
        for idx, href in enumerate(hrefs):
            full = f"{base_dir}/{href}" if base_dir and not href.startswith("/") else href
            full = full.lstrip("/")
            if full not in names:
                cand = next((k for k in names if k.endswith(Path(href).name)), None)
                if not cand:
                    continue
                full = cand
            try:
                html = zf.read(full).decode("utf-8", "ignore")
            except KeyError:
                continue
            text = normalize(strip_boilerplate(html_to_text(html)))
            if len(text) < 40:
                continue
            heading_match = re.search(r"<h[12][^>]*>(.*?)</h[12]>", html, re.DOTALL | re.IGNORECASE)
            chapter = strip_html_to_text(heading_match.group(1))[:180] if heading_match else f"Part {idx + 1}"
            pages.append(
                RawPage(
                    index=idx,
                    text=text,
                    page_number=None,
                    page_label=f"EPUB file {idx + 1}",
                    chapter=chapter,
                )
            )
    return ExtractedDocument(
        pages=pages,
        title=title,
        author=author,
        doc_type="epub",
        language=_guess_language(" ".join(p.text[:400] for p in pages[:4])),
        metadata={"description": desc[:500]},
    )


# --------------------------------------------------------------------------- #
#  Markdown / text / html files
# --------------------------------------------------------------------------- #
def extract_markdown(path: Path) -> ExtractedDocument:
    text = normalize(path.read_text(encoding="utf-8", errors="ignore"))
    title = ""
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if m:
        title = m.group(1).strip()
    author = ""
    am = re.search(r"^\s*(?:author|by)\s*[:—-]\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    if am:
        author = am.group(1).strip()[:120]
    return ExtractedDocument(
        pages=[RawPage(index=0, text=text, page_number=1, page_label="doc", chapter="")],
        title=title or path.stem,
        author=author,
        doc_type="md",
        language=_guess_language(text[:800]),
    )


def extract_html_file(path: Path) -> ExtractedDocument:
    html = path.read_text(encoding="utf-8", errors="ignore")
    text = normalize(strip_boilerplate(html_to_text(html)))
    tm = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
    return ExtractedDocument(
        pages=[RawPage(index=0, text=text, page_number=1, page_label="page", chapter="")],
        title=strip_html_to_text(tm.group(1))[:300] if tm else path.stem,
        doc_type="html",
        language=_guess_language(text[:800]),
    )


def extract_plain(path: Path) -> ExtractedDocument:
    text = normalize(path.read_text(encoding="utf-8", errors="ignore"))
    return ExtractedDocument(
        pages=[RawPage(index=0, text=text, page_number=1, page_label="page")],
        title=path.stem,
        doc_type="txt",
        language=_guess_language(text[:800]),
    )


def extract_text_blob(text: str, *, title: str = "", doc_type: str = "md", url: str = "") -> ExtractedDocument:
    """Ingest an in-memory string (web pages, pasted notes)."""
    body = normalize(strip_boilerplate(html_to_text(text) if doc_type in {"html", "url"} else text))
    pages = [RawPage(index=0, text=part, page_number=i + 1, page_label=f"chunk-part {i + 1}")
             for i, part in enumerate(_split_long(body, 12000))]
    return ExtractedDocument(
        pages=pages or [RawPage(index=0, text="")],
        title=title or (url or "Untitled").rsplit("/", 1)[-1][:200],
        doc_type=doc_type,
        language=_guess_language(body[:800]),
        metadata={"url": url},
    )


def _split_long(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    out, buf, count = [], "", 0
    for para in text.split("\n\n"):
        if count + len(para) > size and buf:
            out.append(buf.strip())
            buf, count = "", 0
        buf += para + "\n\n"
        count += len(para) + 2
    if buf.strip():
        out.append(buf.strip())
    return out


def _guess_language(text: str) -> str:
    cyr = len(re.findall(r"[а-яА-ЯёЁ]", text or ""))
    lat = len(re.findall(r"[a-zA-Z]", text or ""))
    if cyr > lat * 0.5 and cyr > 20:
        return "ru"
    return "en"


# --------------------------------------------------------------------------- #
#  Dispatch
# --------------------------------------------------------------------------- #
def extract_file(path: Path) -> ExtractedDocument:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(path)
    if ext == ".epub":
        return extract_epub(path)
    if ext in {".md", ".markdown", ".rst", ".org"}:
        return extract_markdown(path)
    if ext in {".html", ".htm"}:
        return extract_html_file(path)
    if ext in {".txt", ".csv", ".json", ".py", ".ipynb", ""}:
        return extract_plain(path)
    # Unknown binary-ish extension: refuse loudly instead of silently producing garbage
    raise ValueError(f"Unsupported file type: {ext or 'unknown'}")

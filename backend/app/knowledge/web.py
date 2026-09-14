"""
Web sources: URL -> fetch -> extract -> (clean/chunk/embed/index handled by ingest).

Design constraints from the spec:
  * the web is NEVER queried automatically for every question;
  * it is only used when the user allowed it AND the local KB is insufficient;
  * every source gets a trust tier so ranking prefers the user's library first.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from typing import Any
from urllib.parse import urlparse

import httpx

from app.config import settings
from app.knowledge.extractors import html_to_text
from app.knowledge.textutil import normalize, split_sentences, tokenize

logger = logging.getLogger(__name__)

MAX_HTML_BYTES = 8_000_000

OFFICIAL_DOC_HOSTS = (
    "scikit-learn.org", "pytorch.org", "tensorflow.org", "numpy.org", "pandas.pydata.org",
    "docs.python.org", "huggingface.co", "pytorch3d.org", "mlflow.org", "kubeflow.org",
    "kserve.github.io", "ray.io", "spark.apache.org", "docs.qdrant.tech", "python.langchain.com",
    "openai.com", "docs.yandex.cloud", "yandex.cloud", "developers.google.com", "aws.amazon.com",
)
PAPER_HOSTS = ("arxiv.org", "openreview.net", "proceedings.mlr.press", "dl.acm.org", "ieeexplore.ieee.org", "aclanthology.org", "papers.nips.cc", "papers.neurips.cc")
EDU_HOSTS = ("wikipedia.org", "coursera.org", "edx.org", "fast.ai", "course.fast.ai", "d2l.ai", "explained.ai", "colah.github.io", "karpathy.ai", "sebastianraschka.com", "distill.pub", "half.ai", "fullstackdeeplearning.com", "madebyollin.github.io", "jakevdp.github.io", "allclose.ai")


class WebFetchError(RuntimeError):
    pass


def guess_tier_from_url(url: str) -> int:
    host = (urlparse(url).hostname or "").lower()
    if host.endswith("github.io") or "github.com" in host:
        return 2
    if any(host == d or host.endswith("." + d) for d in OFFICIAL_DOC_HOSTS):
        return 2
    if any(host == d or host.endswith("." + d) for d in PAPER_HOSTS):
        return 3
    if any(host == d or host.endswith("." + d) for d in EDU_HOSTS):
        return 4
    return 5


def _assert_public_host(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise WebFetchError(f"Only http/https URLs are supported (got {parsed.scheme or 'none'!r})")
    host = parsed.hostname
    if not host:
        raise WebFetchError("URL has no hostname")
    if settings.web_allow_private_hosts:
        return
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise WebFetchError(f"DNS resolution failed for {host}: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise WebFetchError(f"Refusing to fetch a non-public address ({ip}) - set WEB_ALLOW_PRIVATE_HOSTS=true to override")


def extract_title(html: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
    if m:
        return normalize(re.sub(r"<[^>]+>", "", m.group(1)))[:200]
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.DOTALL | re.IGNORECASE)
    if m:
        return normalize(re.sub(r"<[^>]+>", "", m.group(1)))[:200]
    return ""


def extract_meta(html: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for name in ("description", "article:published_time", "og:site_name", "author", "citation_title", "citation_author"):
        m = re.search(rf'<meta[^>]+(?:name|property)=["\']{re.escape(name)}["\'][^>]+content=["\'](.*?)["\']', html, re.IGNORECASE | re.DOTALL)
        if m:
            meta[name] = normalize(re.sub(r"<[^>]+>", "", m.group(1)))[:400]
    return meta


def _main_content(html: str) -> str:
    """Prefer <article>/<main>, else largest <div>; fall back to whole body."""
    for tag in ("article", "main"):
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", html, re.DOTALL | re.IGNORECASE)
        if m and len(m.group(1)) > 1200:
            return m.group(1)
    best, best_len = "", 0
    for m in re.finditer(r'<div[^>]*class="[^"]*(?:content|article|post|doc|body|prose)[^"]*"[^>]*>(.*?)</div>', html, re.DOTALL | re.IGNORECASE):
        if len(m.group(1)) > best_len:
            best, best_len = m.group(1), len(m.group(1))
    if best_len > 1500:
        return best
    body = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL | re.IGNORECASE)
    return body.group(1) if body else html


def clean_web_text(html: str) -> str:
    html = re.sub(r"<(script|style|noscript|template|svg|form|nav|header|footer|aside)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = html_to_text(html)
    text = normalize(text)
    kept: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if len(para) < 60 and not para.startswith("#") and "@@H" not in para:
            continue
        low = para.lower()
        if any(marker in low for marker in ("cookie", "subscribe to", "log in", "sign up", "all rights reserved", "javascript is required", "privacy policy", "terms of service")):
            continue
        kept.append(para)
    text = "\n\n".join(kept)
    return re.sub(r"@@H([1-6])\s(.*?)\s?@@", lambda m: "#" * int(m.group(1)) + " " + m.group(2), text)


def fetch_url(url: str, *, timeout: float | None = None, max_chars: int = 400_000) -> dict[str, Any]:
    """Fetch + extract readable text from a URL (sync; called from worker threads)."""
    if not settings.web_ingestion_enabled:
        raise WebFetchError("Web ingestion is disabled (WEB_INGESTION_ENABLED=false)")
    _assert_public_host(url)
    timeout = timeout or settings.web_fetch_timeout_seconds
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": settings.user_agent, "Accept": "text/html,application/xhtml+xml,text/plain,*/*"},
        ) as client:
            response = client.get(url)
            content_type = response.headers.get("content-type", "")
            if response.status_code >= 400:
                raise WebFetchError(f"HTTP {response.status_code} while fetching URL")
            raw = response.content[:MAX_HTML_BYTES]
    except httpx.HTTPError as exc:
        raise WebFetchError(f"Could not fetch {url}: {exc}") from exc

    body = raw.decode("utf-8", "ignore")
    if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
        return {"url": url, "final_url": str(response.url), "title": extract_title(body) or url.rsplit("/", 1)[-1], "html": "", "text": "", "is_pdf": True, "content_type": content_type}

    text = clean_web_text(body)
    title = extract_title(body)
    return {
        "url": url,
        "final_url": str(response.url),
        "title": (title or url)[:200],
        "html": body[:max_chars * 3],
        "text": text[:max_chars],
        "content_type": content_type,
        "meta": extract_meta(body),
        "tier": guess_tier_from_url(url),
        "chars": len(text),
    }


def fetch_to_document_blob(url: str) -> dict[str, Any]:
    data = fetch_url(url)
    if not data.get("text") or len(data["text"]) < 200:
        raise WebFetchError(
            "That page did not contain enough extractable text (it may be JavaScript-rendered). "
            "Try a static URL, documentation page, or upload the file directly."
        )
    return data


def relevant_sentences(text: str, query: str, *, limit: int = 8) -> list[str]:
    """Lightweight post-retrieval narrowing for long web pages."""
    query_terms = set(tokenize(query))
    scored: list[tuple[float, str]] = []
    for sentence in split_sentences(text):
        terms = set(tokenize(sentence))
        if not terms:
            continue
        overlap = len(query_terms & terms) / max(3.0, len(query_terms))
        length_bonus = min(1.0, len(sentence) / 180.0)
        scored.append((overlap * (0.6 + 0.4 * length_bonus), sentence))
    scored.sort(key=lambda kv: -kv[0])
    return [s for score, s in scored[:limit] if score > 0.02]

"""Text normalisation, tokenisation and light NLP used by ingestion + retrieval."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter

# --------------------------------------------------------------------------- #
#  Regexes
# --------------------------------------------------------------------------- #
_WS_RE = re.compile(r"[ \t\f\v]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？])[\s\n]+(?=[A-ZА-ЯЁ0-9\"'(\[])")
_TOKEN_RE = re.compile(r"[^\W\d_]{2,}|(?:[a-z_]\w*\.){1,}\w+|\d+(?:[.,]\d+)?", re.UNICODE)
_CODE_RE = re.compile(r"<code[^>]*>(.*?)</code>|```(?:\w+)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+")
_PAGE_MARK_RE = re.compile(r"^\s*\[?page[:\s]?(\d{1,5})\]?\s*$", re.IGNORECASE | re.MULTILINE)
_LINE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
_FOOTER_RE = re.compile(r"^\s*(page\s+\d+|\d{1,4})\s*(/\s*\d+\s*)?$", re.IGNORECASE | re.MULTILINE)
_CHAPTER_RE = re.compile(
    r"^(chapter|глава|part|section| appendix)\s*([ivxlcdm\d]+(?:\.\d+)*)(?::|\.\s|-|\s+)(.{0,90})$",
    re.IGNORECASE,
)

_STOP_GROUPS = (
    "the a an and or of to in for on with by is are was were be been being this that these those it its as at from",
    "then than into over under which while their there here about can could will would should must may might",
    "и в во не что он на я с со как а то все она так его но да ты к у же вы там если когда более очень самый есть нет",
    "который которая которые это для того чтобы при над под меж без сквозной",
)
STOPWORDS = {w for group in _STOP_GROUPS for w in group.split()}

# ML vocabulary that must survive as single tokens (used for topic/skill tagging)
DOMAIN_TERMS = {
    "gradient", "descent", "regularization", "overfitting", "underfitting", "bias", "variance",
    "loss", "cost", "logits", "softmax", "sigmoid", "relu", "dropout", "batchnorm", "normalization",
    "convolution", "pooling", "embedding", "attention", "transformer", "encoder", "decoder",
    "tokenizer", "tokenization", "perplexity", "fine-tuning", "finetuning", "lora", "quantization",
    "regression", "classification", "clustering", "kmeans", "svm", "random", "forest", "boosting",
    "bagging", "cross-validation", "auc", "roc", "precision", "recall", "f1", "rmse", "mae",
    "feature", "pipeline", "inference", "training", "validation", "test", "dataset", "label",
    "epoch", "iteration", "learning-rate", "momentum", "adam", "sgd", "backpropagation",
    "derivative", "derivative", "partial", "eigenvalue", "eigenvector", "matrix", "vector",
    "tensor", "norm", "hessian", "likelihood", "bayesian", "prior", "posterior", "entropy",
    "information-gain", "gini", "cart", "naive", "bayes", "pca", "svd", "lda", "umap", "tsne",
    "sharding", "latency", "throughput", "monitoring", "drift", "shadowing", "canary",
    "docker", "kubernetes", "airflow", "featurestore", "vector", "database", "retrieval",
    "rag", "prompt", "chain", "agent", "index", "chunk", "embedding", "sqlite", "postgresql",
}


def normalize(text: str) -> str:
    """Unicode-normalise + strip control chars + collapse whitespace runs."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "").replace("\ufeff", "")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def strip_boilerplate(text: str) -> str:
    """Remove running headers/footers/page numbers that PDF extraction leaves behind."""
    text = _LINE_NUM_RE.sub("", text)
    text = _FOOTER_RE.sub("", text)
    text = _URL_RE.sub(lambda m: m.group(0) if "arxiv" in m.group(0) else "", text)
    return normalize(text)


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    chunks = re.split(r"\n{2,}", text)
    out: list[str] = []
    for block in chunks:
        for sentence in _SENT_SPLIT_RE.split(block.strip()):
            sentence = sentence.strip()
            if len(sentence) >= 3:
                out.append(sentence)
    return out


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Language-agnostic tokeniser: words, dotted identifiers and numbers."""
    if not text:
        return []
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        tok = raw.strip("._-")
        if not tok or len(tok) < 2:
            continue
        if not keep_stopwords and tok in STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def content_hash(text: str) -> str:
    return hashlib.sha256(" ".join(tokenize(text, keep_stopwords=True)).encode("utf-8")).hexdigest()[:32]


def keywords(text: str, limit: int = 14) -> list[str]:
    """Cheap but effective key-term extraction (freq * domain boost)."""
    tokens = tokenize(text)
    if not tokens:
        return []
    counts = Counter(tokens)
    n = len(tokens)
    scored: list[tuple[float, str]] = []
    for term, freq in counts.items():
        score = freq / n * (2.6 if term in DOMAIN_TERMS else 1.0)
        if freq < 2 and len(term) < 7 and term not in DOMAIN_TERMS:
            continue
        scored.append((score, term))
    scored.sort(key=lambda kv: (-kv[0], kv[1]))
    return [term for _, term in scored[:limit]]


def detect_chapter(line: str) -> tuple[str, str] | None:
    """Return (chapter_label, chapter_title) if the line looks like a chapter heading."""
    m = _CHAPTER_RE.match(line.strip())
    if not m:
        return None
    label = f"{m.group(1).title()} {m.group(2)}"
    title = (m.group(3) or "").strip(" :.-")
    return label, title


def truncate(text: str, limit: int, suffix: str = " …") -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


def extract_headings(text: str) -> list[tuple[int, int, str]]:
    """Markdown/asciidoc style headings: [(line_no, level, title)]."""
    out: list[tuple[int, int, str]] = []
    for idx, line in enumerate(text.split("\n")):
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m:
            out.append((idx, len(m.group(1)), m.group(2).strip().rstrip("# ")))
    return out


def count_words(text: str) -> int:
    return len(_TOKEN_RE.findall(text or ""))


def estimate_tokens(text: str) -> int:
    """Rough token estimate for budget control (chars/3.6 keeps CJK+latin sane)."""
    return max(1, int(len(text or "") / 3.6)) if text else 0

"""
Embedding providers.

The default embedder is *offline, deterministic and dependency-free*:
``HashingEmbedder`` builds signed feature-hashing vectors with corpus IDF
weighting. It is not a neural model - it is a strong lexical-semantic hybrid
that makes RAG work out of the box without internet, GPU or API budget.

Real neural embeddings can be swapped in through configuration only:
  EMBEDDING_PROVIDER=yandex  -> Yandex Foundation Models embedding endpoint
  EMBEDDING_PROVIDER=openai  -> any OpenAI-compatible /v1/embeddings
"""

from __future__ import annotations

import hashlib
import logging
import re
import math
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

from app.config import settings
from app.knowledge.textutil import tokenize

logger = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{3,}")


class Embedder(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        ...

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @staticmethod
    def normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype("float32")


# --------------------------------------------------------------------------- #
#  Offline: signed feature hashing + IDF
# --------------------------------------------------------------------------- #
class HashingEmbedder(Embedder):
    name = "mlea-hashing-v1"

    def __init__(self, dim: int | None = None, idf: dict[str, float] | None = None) -> None:
        self.dim = dim or settings.embedding_dim
        self.idf: dict[str, float] = idf or {}

    # ---------------------------------------------------------------- idf io
    def set_idf(self, idf: dict[str, float]) -> None:
        self.idf = idf

    @staticmethod
    def compute_idf(docs: list[str]) -> dict[str, float]:
        n = max(1, len(docs))
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(set(tokenize(doc)))
        return {term: math.log(1.0 + n / (1.0 + freq)) for term, freq in df.items() if freq >= 1}

    # ------------------------------------------------------------- features
    def _features(self, text: str) -> Counter[str]:
        tokens = tokenize(text)
        if not tokens:
            return Counter()
        feats: Counter[str] = Counter()
        for pos, tok in enumerate(tokens):
            feats["w:" + tok] += 1.0
            if tok in {"model", "data", "loss", "train", "test"}:
                feats["w2:" + tok] += 0.4
            if pos + 1 < len(tokens):
                feats["b:" + tok + "|" + tokens[pos + 1]] += 0.75
            if len(tok) >= 8:  # morphological robustness (esp. Russian agglutination)
                for k in (3, 4, 5):
                    feats["s:" + tok[:k]] += 0.25
                    feats["e:" + tok[-k:]] += 0.2
        # identifier-ish runs (python code inside books/docs)
        for ident in set(
            h
            for h in _IDENT_RE.findall(text)
            if "_" in h or h[0].isupper() or "." in h
        ):
            feats["i:" + ident.lower()] += 1.6
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            feats = self._features(text)
            if not feats:
                continue
            total = sum(feats.values()) or 1.0
            for feat, tf in feats.items():
                digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                code = int.from_bytes(digest, "big")
                idx = code % self.dim
                sign = 1.0 if (code >> 63) & 1 == 0 else -1.0
                term = feat.split(":", 1)[1].split("|", 1)[0]
                weight = (1.0 + math.log(tf / total * len(feats) + 1.0))
                idf = self.idf.get(term)
                if idf is None and "|" in feat:
                    idf = 1.0
                weight *= idf if idf and idf > 0 else 1.0
                matrix[row, idx] += np.float32(sign * weight)
        return self.normalize(matrix)


# --------------------------------------------------------------------------- #
#  Remote providers (optional)
# --------------------------------------------------------------------------- #
class _ApiEmbedder(Embedder):
    def __init__(self) -> None:
        self.dim = settings.embedding_dim
        self._probed = False

    def _probe(self) -> None:  # pragma: no cover - network
        self._probed = True

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - network
        raise NotImplementedError


class YandexEmbedder(_ApiEmbedder):  # pragma: no cover - network dependent
    """Yandex Foundation Models embeddings (multilingual, 512/1024/2048 dims)."""

    name = "yandex-embedding"

    def __init__(self) -> None:
        super().__init__()
        from app.ai.yandex import YandexGPTProvider

        self._client = YandexGPTProvider()

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        vectors = self._client.embed(texts)
        return self.normalize(np.asarray(vectors, dtype="float32"))


class OpenAIEmbedder(_ApiEmbedder):  # pragma: no cover - network dependent
    name = "openai-embedding"

    def __init__(self) -> None:
        super().__init__()
        from app.ai.openai_compat import OpenAICompatProvider

        self._client = OpenAICompatProvider()

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        vectors = self._client.embed(texts)
        return self.normalize(np.asarray(vectors, dtype="float32"))


# --------------------------------------------------------------------------- #
_embedder: Embedder | None = None


def get_embedder(*, force_new: bool = False) -> Embedder:
    global _embedder
    if _embedder is not None and not force_new:
        return _embedder
    provider = settings.embedding_provider
    try:
        if provider == "yandex" and settings.has_yandex_credentials():
            _embedder = YandexEmbedder()
        elif provider == "openai" and settings.openai_api_key:
            _embedder = OpenAIEmbedder()
        else:
            _embedder = HashingEmbedder()
    except Exception as exc:  # noqa: BLE001 - never let embedding config break the app
        logger.warning("Embedder %s unavailable (%s); using offline hashing embedder", provider, exc)
        _embedder = HashingEmbedder()
    if isinstance(_embedder, HashingEmbedder):
        try:
            _embedder.set_idf(_load_idf_from_db())
        except Exception:  # noqa: BLE001 - empty DB at first boot
            pass
    return _embedder


def _load_idf_from_db() -> dict[str, float]:
    from app.db import session_scope
    from app.models import EmbeddingStat

    idf: dict[str, float] = {}
    with session_scope() as session:
        rows = session.query(EmbeddingStat).limit(80000).all()
        total = max(1, rows[0].total_docs if rows else 1)
        for row in rows:
            idf[row.term] = math.log(1.0 + total / (1.0 + row.doc_freq))
    return idf


def persist_idf(docs: list[str]) -> None:
    """Refresh corpus IDF statistics after (re)indexing a document set."""
    from app.db import session_scope
    from app.models import EmbeddingStat

    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(tokenize(doc)))
    if not df:
        return
    total_docs = max(1, len(docs))
    with session_scope() as session:
        session.query(EmbeddingStat).delete()
        session.bulk_insert_mappings(
            EmbeddingStat,
            [
                {"term": term, "doc_freq": freq, "total_docs": total_docs}
                for term, freq in df.most_common(60000)
            ],
        )
    embedder = get_embedder()
    if isinstance(embedder, HashingEmbedder):
        embedder.set_idf(
            {term: math.log(1.0 + total_docs / (1.0 + freq)) for term, freq in df.items()}
        )


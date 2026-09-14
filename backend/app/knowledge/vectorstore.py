"""
Pluggable vector stores.

* ``LocalVectorStore``  - numpy index persisted next to the SQLite DB. Zero deps,
                          works offline, handles 100k+ chunks fine for a personal KB.
* ``ChromaVectorStore`` - embedded Chroma (``VECTOR_BACKEND=chroma``, pip install chromadb)
* ``QdrantVectorStore`` - Qdrant server (``VECTOR_BACKEND=qdrant``, used by docker-compose)

All of them store the *authoritative* vector in ``document_chunks.embedding`` as well,
so any backend can be rebuilt from SQLite with `POST /api/library/reindex-all`.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class VectorHit:
    chunk_id: int
    score: float
    meta: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.meta = self.meta or {}


class VectorStore(ABC):
    name = "base"

    @abstractmethod
    def upsert(self, chunk_id: int, document_id: int, vector: np.ndarray, meta: dict) -> None: ...

    @abstractmethod
    def upsert_many(self, rows: list[tuple[int, int, np.ndarray, dict]]) -> None: ...

    @abstractmethod
    def delete_document(self, document_id: int) -> int: ...

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int = 10, *, document_ids: list[int] | None = None) -> list[VectorHit]: ...

    @abstractmethod
    def count(self) -> int: ...

    def health(self) -> dict:
        return {"backend": self.name, "ok": True, "vectors": self.count()}


# --------------------------------------------------------------------------- #
#  Local numpy index
# --------------------------------------------------------------------------- #
class LocalVectorStore(VectorStore):
    name = "local-numpy"

    def __init__(self, path: Path | None = None, dim: int | None = None) -> None:
        self.path = path or (settings.vector_dir / "index.npz")
        self.dim = dim or settings.embedding_dim
        self._lock = threading.RLock()
        self._ids: list[int] = []
        self._docs: list[int] = []
        self._matrix: np.ndarray = np.zeros((0, self.dim), dtype="float32")
        self._index: dict[int, int] = {}
        self._load()

    # ------------------------------------------------------------------ io
    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            with np.load(self.path, allow_pickle=False) as data:
                self._ids = [int(x) for x in data["ids"]]
                self._docs = [int(x) for x in data["docs"]]
                matrix = data["matrix"]
                if matrix.size:
                    self.dim = int(matrix.shape[1])
                    self._matrix = np.ascontiguousarray(matrix, dtype="float32")
                self._index = {cid: i for i, cid in enumerate(self._ids)}
        except Exception as exc:  # noqa: BLE001 - corrupt cache => rebuild
            logger.warning("Vector index unreadable (%s); starting empty", exc)
            self._ids, self._docs, self._index = [], [], {}
            self._matrix = np.zeros((0, self.dim), dtype="float32")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.npz")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, ids=np.asarray(self._ids, dtype="int64"), docs=np.asarray(self._docs, dtype="int64"), matrix=self._matrix)
        tmp.replace(self.path)

    def _ensure_dim(self, dim: int) -> None:
        if self.dim == dim or self._matrix.shape[0] == 0:
            self.dim = dim
            if self._matrix.shape[0] == 0:
                self._matrix = np.zeros((0, dim), dtype="float32")
            return
        raise ValueError(f"Vector dim mismatch: index={self.dim} new={dim}")

    # -------------------------------------------------------------- mutations
    def upsert(self, chunk_id: int, document_id: int, vector: np.ndarray, meta: dict) -> None:
        self.upsert_many([(chunk_id, document_id, vector, meta)])

    def upsert_many(self, rows: list[tuple[int, int, np.ndarray, dict]]) -> None:
        if not rows:
            return
        with self._lock:
            dim = int(np.asarray(rows[0][2]).shape[0])
            self._ensure_dim(dim)
            update_ids = [int(r[0]) for r in rows]
            positions = [self._index.get(cid) for cid in update_ids]
            fresh = [(i, r) for i, r in zip(positions, rows) if i is None]
            changed = [(i, r) for i, r in zip(positions, rows) if i is not None]
            for i, (_cid, _doc, vec, _meta) in changed:
                self._matrix[i] = np.asarray(vec, dtype="float32")
            if fresh:
                new_matrix = np.asarray([r[2] for _, r in fresh], dtype="float32")
                self._matrix = np.vstack([self._matrix, new_matrix]) if self._matrix.size else new_matrix
                for (_, r), pos in zip(fresh, range(self._matrix.shape[0] - len(fresh), self._matrix.shape[0])):
                    cid, doc_id, _vec, _meta = r
                    self._ids.append(int(cid))
                    self._docs.append(int(doc_id))
                    self._index[int(cid)] = pos
            self._save()

    def delete_document(self, document_id: int) -> int:
        with self._lock:
            if self._matrix.size == 0:
                return 0
            docs = np.asarray(self._docs, dtype="int64")
            keep = docs != int(document_id)
            removed = int((~keep).sum())
            if removed == 0:
                return 0
            self._matrix = self._matrix[keep]
            self._ids = [cid for cid, k in zip(self._ids, keep) if k]
            self._docs = [d for d, k in zip(self._docs, keep) if k]
            self._index = {cid: i for i, cid in enumerate(self._ids)}
            self._save()
            return removed

    def rebuild(self, rows: list[tuple[int, int, np.ndarray]]) -> None:
        with self._lock:
            self._ids, self._docs, self._index = [], [], {}
            if not rows:
                self._matrix = np.zeros((0, self.dim), dtype="float32")
                self._save()
                return
            self._matrix = np.asarray([r[2] for r in rows], dtype="float32")
            self.dim = int(self._matrix.shape[1])
            for i, (cid, doc_id, _vec) in enumerate(rows):
                self._ids.append(int(cid))
                self._docs.append(int(doc_id))
                self._index[int(cid)] = i
            self._save()

    # ---------------------------------------------------------------- search
    def search(self, query: np.ndarray, top_k: int = 10, *, document_ids: list[int] | None = None) -> list[VectorHit]:
        with self._lock:
            if self._matrix.size == 0:
                return []
            q = np.asarray(query, dtype="float32")
            if q.shape[0] != self.dim:  # embedder changed -> caller must reindex
                return []
            norm = float(np.linalg.norm(q)) or 1.0
            sims = (self._matrix @ q) / norm
            if document_ids:
                allowed = {int(d) for d in document_ids}
                mask = np.asarray([d in allowed for d in self._docs], dtype=bool)
                sims = np.where(mask, sims, -1.0)
            top_k = min(top_k, sims.shape[0])
            if top_k <= 0:
                return []
            idx = np.argpartition(-sims, kth=top_k - 1)[:top_k]
            idx = idx[np.argsort(-sims[idx])]
            return [VectorHit(chunk_id=int(self._ids[i]), score=float(max(0.0, sims[i]))) for i in idx if sims[i] > -0.5]

    def count(self) -> int:
        return len(self._ids)


# --------------------------------------------------------------------------- #
#  Qdrant (REST, no client dependency)
# --------------------------------------------------------------------------- #
class QdrantVectorStore(VectorStore):  # pragma: no cover - requires a server
    name = "qdrant"

    def __init__(self, url: str | None = None, collection: str | None = None) -> None:
        import httpx

        self.url = (url or settings.qdrant_url).rstrip("/")
        self.collection = collection or settings.qdrant_collection
        self.dim = settings.embedding_dim
        headers = {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}
        self._client = httpx.Client(base_url=self.url, timeout=20.0, headers=headers)
        self._ensure_collection()

    def _ensure_collection(self) -> None:  # pragma: no cover
        try:
            exists = self._client.get(f"/collections/{self.collection}")
            if exists.status_code == 200 and exists.json().get("result", {}).get("status"):
                return
            self._client.put(
                f"/collections/{self.collection}",
                json={"vectors": {"size": self.dim, "distance": "Cosine"}, "on_payload": {"document_id": "integer"}},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qdrant unavailable at %s: %s", self.url, exc)
            raise

    def upsert(self, chunk_id: int, document_id: int, vector: np.ndarray, meta: dict) -> None:
        self.upsert_many([(chunk_id, document_id, vector, meta)])

    def upsert_many(self, rows: list[tuple[int, int, np.ndarray, dict]]) -> None:  # pragma: no cover
        if not rows:
            return
        points = [
            {"id": int(cid), "vector": [float(x) for x in vec], "payload": {"document_id": int(doc), **(meta or {})}}
            for cid, doc, vec, meta in rows
        ]
        for i in range(0, len(points), 64):
            self._client.put(f"/collections/{self.collection}/points", json={"points": points[i : i + 64]})

    def delete_document(self, document_id: int) -> int:  # pragma: no cover
        res = self._client.post(
            f"/collections/{self.collection}/points/delete?wait=true",
            json={"filter": {"must": [{"key": "document_id", "match": {"value": int(document_id)}}]}},
        )
        return int(res.json().get("result", {}).get("deleted", 0)) if res.status_code == 200 else 0

    def search(self, query: np.ndarray, top_k: int = 10, *, document_ids: list[int] | None = None) -> list[VectorHit]:  # pragma: no cover
        body: dict = {"vector": [float(x) for x in query], "limit": max(1, top_k), "with_payload": True}
        if document_ids:
            body["filter"] = {"must": [{"key": "document_id", "match": {"any": [int(d) for d in document_ids]}}]}
        res = self._client.post(f"/collections/{self.collection}/points/search", json=body)
        hits = res.json().get("result", []) if res.status_code == 200 else []
        return [
            VectorHit(chunk_id=int(h["id"]), score=float(max(0.0, h.get("score", 0.0))), meta=h.get("payload") or {})
            for h in hits
        ]

    def count(self) -> int:  # pragma: no cover
        try:
            return int(self._client.get(f"/collections/{self.collection}").json()["result"]["points_count"])
        except Exception:  # noqa: BLE001
            return 0


# --------------------------------------------------------------------------- #
#  Chroma (optional embedded)
# --------------------------------------------------------------------------- #
class ChromaVectorStore(VectorStore):  # pragma: no cover - optional dependency
    name = "chroma"

    def __init__(self, path: Path | None = None, collection: str | None = None) -> None:
        import chromadb  # type: ignore
        from chromadb.config import Settings as ChromaSettings  # type: ignore

        self._client = chromadb.PersistentClient(
            path=str(path or (settings.data_dir / "chroma")),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection or settings.chroma_collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunk_id: int, document_id: int, vector: np.ndarray, meta: dict) -> None:
        self.upsert_many([(chunk_id, document_id, vector, meta)])

    def upsert_many(self, rows: list[tuple[int, int, np.ndarray, dict]]) -> None:  # pragma: no cover
        if not rows:
            return
        self._collection.upsert(
            ids=[str(r[0]) for r in rows],
            embeddings=[[float(x) for x in r[2]] for r in rows],
            metadatas=[{"document_id": int(r[1]), **(r[3] or {})} for r in rows],
        )

    def delete_document(self, document_id: int) -> int:  # pragma: no cover
        existing = self._collection.get(where={"document_id": int(document_id)}) or {}
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def search(self, query: np.ndarray, top_k: int = 10, *, document_ids: list[int] | None = None) -> list[VectorHit]:  # pragma: no cover
        kwargs: dict = {"query_embeddings": [[float(x) for x in query]], "n_results": max(1, top_k), "include": ["metadatas", "distances"]}
        if document_ids:
            kwargs["where"] = {"document_id": {"$in": [int(d) for d in document_ids]}}
        res = self._collection.query(**kwargs)
        out: list[VectorHit] = []
        for cid, dist, meta in zip(
            res["ids"][0], res["distances"][0], (res.get("metadatas") or [[]])[0] or [{}] * len(res["ids"][0])
        ):
            out.append(VectorHit(chunk_id=int(cid), score=float(max(0.0, 1.0 - dist)), meta=meta or {}))
        return out

    def count(self) -> int:  # pragma: no cover
        try:
            return int(self._collection.count())
        except Exception:  # noqa: BLE001
            return 0


_store: VectorStore | None = None
_store_lock = threading.Lock()


def get_vector_store(*, force_new: bool = False) -> VectorStore:
    global _store
    with _store_lock:
        if _store is not None and not force_new:
            return _store
        backend = settings.vector_backend
        try:
            if backend == "qdrant":
                _store = QdrantVectorStore()
            elif backend == "chroma":
                _store = ChromaVectorStore()
            else:
                _store = LocalVectorStore()
        except Exception as exc:  # noqa: BLE001 - external DB down => stay usable offline
            logger.warning("Vector backend %r failed (%s); falling back to local numpy index", backend, exc)
            _store = LocalVectorStore()
        return _store


def reset_vector_store_for_tests(store: VectorStore | None = None) -> None:
    global _store
    _store = store

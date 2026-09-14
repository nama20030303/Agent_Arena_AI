"""
Shared fixtures.

Unit tests and API tests deliberately share ONE database per test function: the app reads
its engine from app.db, so `reset_engine_for_tests` is the single switch that keeps HTTP
tests and service tests consistent (no session/function-scope mismatch).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


@pytest.fixture(scope="session")
def workspace() -> Iterator[Path]:
    root = Path(tempfile.mkdtemp(prefix="mlea-tests-"))
    os.environ.update(
        {
            "DATA_DIR": str(root),
            "DATABASE_URL": f"sqlite:///{root / 'test.db'}",
            "SEED_ON_STARTUP": "false",
            "AI_PROVIDER": "none",
            "YANDEX_API_KEY": "",
            "YANDEX_FOLDER_ID": "",
            "VECTOR_BACKEND": "local",
            "EMBEDDING_PROVIDER": "hashing",
            "CODE_EXECUTION_ENABLED": "true",
            "CODE_EXECUTION_TIMEOUT_SECONDS": "30",
            "LOG_LEVEL": "WARNING",
        }
    )
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def db(workspace):
    """Fresh schema + isolated vector/lexical caches, per test."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import app.models  # noqa: F401 - register mappers
    from app.config import get_settings
    from app.db import Base, reset_engine_for_tests
    from app.knowledge.embeddings import get_embedder
    from app.knowledge.retrieval import invalidate_lexical_index
    from app.knowledge.vectorstore import LocalVectorStore, reset_vector_store_for_tests

    settings = get_settings()
    settings.ensure_dirs()
    engine = create_engine(settings.resolved_db_url, connect_args={"check_same_thread": False}, future=True)
    reset_engine_for_tests(engine)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    index_path = settings.vector_dir / f"index-{abs(hash(id(engine))) % 10_000_000}.npz"
    if index_path.exists():
        index_path.unlink()
    reset_vector_store_for_tests(LocalVectorStore(path=index_path, dim=settings.embedding_dim))
    invalidate_lexical_index()
    try:
        get_embedder(force_new=True)
    except Exception:  # noqa: BLE001
        pass

    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        reset_vector_store_for_tests(None)
        invalidate_lexical_index()
        engine.dispose()


@pytest.fixture()
def seeded(db):
    """Full seed including the sample knowledge library (RAG works out of the box)."""
    from app.seed import seed_all

    seed_all(db, index_samples=True)
    return db


@pytest.fixture()
def fast_seeded(db):
    """Curriculum + content, no document ingestion (much faster for engine tests)."""
    from app.seed import seed_all

    seed_all(db, index_samples=False)
    return db


@pytest.fixture()
def user(fast_seeded):
    from app.models import User

    row = User(username="tester", display_name="Tester", daily_minutes=60, level_index=0)
    fast_seeded.add(row)
    fast_seeded.commit()
    fast_seeded.refresh(row)
    return row


@pytest.fixture()
def client(db, seeded):
    """HTTP client bound to the same per-test database, without app lifespan (seeded manually)."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture()
def auth_client(client, db):
    """Client + bearer token for a freshly registered profile."""
    response = client.post(
        "/api/auth/register",
        json={"username": "e2e-learner", "display_name": "E2E", "background": "career changer", "daily_minutes": 45},
    )
    assert response.status_code == 200, response.text
    token = response.json()["token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client

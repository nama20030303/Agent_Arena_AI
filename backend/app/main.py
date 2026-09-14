"""Application factory, lifespan (schema + seed + index recovery) and static frontend."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import router as api_router
from app.config import settings
from app.db import init_db, session_scope
from app.knowledge.vectorstore import get_vector_store
from app.services.graph import SkillGraph

logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO), format="%(asctime)s %(levelname)-6s %(name)s | %(message)s")
logger = logging.getLogger("mlea")

API_TAG_GROUPS = [
    ("profile", "Profiles, auth, settings"),
    ("diagnostic", "Initial diagnostic test"),
    ("library", "Document library, ingestion, web sources"),
    ("knowledge", "Knowledge base search & RAG"),
    ("teacher", "AI Teacher"),
    ("learning", "Roadmap, plan, skills, progress"),
    ("practice", "Questions, practice, review"),
    ("coding", "Coding lab"),
    ("projects", "Projects, mentor, evaluation"),
    ("assessments", "Exams and mock interviews"),
    ("notes", "Notes, bookmarks, journal, memory"),
    ("system", "Health, AI usage, cache, cost control"),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with session_scope() as session:
        from app.seed import seed_all, seed_summary

        stats = seed_all(session)
        if stats:
            logger.info("seed: %s", stats)
        graph = SkillGraph(session)
        logger.info("curriculum: %d skills, %d topics, %d dependency edges", len(graph.nodes), len(graph.topics), len(graph.requires))

        # recover documents interrupted by a restart
        from app.models import Document
        from app.knowledge.ingest import ingest_document

        stuck = [d.id for d in session.query(Document).filter(Document.status.in_([Document.STATUS_PROCESSING, Document.STATUS_UPLOADING])).all()]
        for doc_id in stuck:
            logger.info("resuming ingestion for document %s", doc_id)
            try:
                ingest_document(session, doc_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("resume failed for %s: %s", doc_id, exc)

        # keep the local vector index consistent with SQLite
        store = get_vector_store()
        if store.name == "local-numpy":
            from app.knowledge.ingest import rebuild_vector_index

            indexed = rebuild_vector_index(session)
            logger.info("vector index: %d vectors", indexed)
    logger.info("ML Engineer Academy API ready (AI provider: %s)", settings.selected_provider_name())
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="ML Engineer Academy",
        version=settings.app_version,
        description="Personal AI ML-engineering school: RAG over your own library + adaptive learning engine.",
        lifespan=lifespan,
    )
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")

    dist = settings.data_dir.parent / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/assets", StaticFiles(directory=str(dist / "assets")), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):  # pragma: no cover - static serving
            target = dist / path
            if path and target.is_file():
                return FileResponse(target)
            return FileResponse(dist / "index.html")

    @app.get("/", include_in_schema=False)
    def root():
        return JSONResponse(
            {
                "app": "ML Engineer Academy",
                "status": "backend is running",
                "docs": "/docs",
                "api_health": "/api/health",
                "hint": "Run the frontend with `npm run dev` in ./frontend (proxied to this API) or build it for static serving.",
            }
        )

    return app


app = create_app()

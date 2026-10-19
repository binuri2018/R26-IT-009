"""FastAPI app: AI STEM Ecosystem API under /api, frontend UI under /."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.common.config import settings
from backend.common.embeddings import (
    EmbeddingLoadError,
    configure_embeddings,
    warm_local_embedding_model,
)
from backend.common.index_manifest import resolve_runtime_manifest
from backend.common.vector_store import VectorStore
from backend.components.adaptive_quiz import adaptive_quiz_router
from backend.components.knowledge_maps import knowledge_maps_router
from backend.components.knowledge_maps.router import (
    set_store_getter as set_knowledge_maps_store_getter,
)
from backend.components.narrative_learning import narrative_learning_router
from backend.components.voice_tutor.router import (
    router as voice_tutor_router,
    set_store_getter,
)

logger = logging.getLogger(__name__)

_store: VectorStore | None = None


def get_store() -> VectorStore:
    if _store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector index not loaded. Run scripts/ingest.py after adding PDFs to Resource/.",
        )
    return _store


set_store_getter(get_store)
set_knowledge_maps_store_getter(get_store)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _store
    data_dir = settings.resolved_data_dir()
    _store = None
    configure_embeddings(None)
    try:
        loaded = VectorStore.load(data_dir)
    except FileNotFoundError:
        pass
    else:
        man = resolve_runtime_manifest(loaded.index.d, data_dir)
        configure_embeddings(man)
        _store = loaded
        if man.get("embedding_provider") != "openai":
            try:
                warm_local_embedding_model()
            except EmbeddingLoadError as exc:
                logger.warning("Local embeddings not ready: %s", exc)

        async def _warm_knowledge_maps_retriever(store: VectorStore) -> None:
            # Builds BM25 + loads the reranker so the first Verify doesn't pay ~10 s.
            try:
                from backend.components.knowledge_maps.hybrid_retrieval import get_hybrid_retriever

                await asyncio.to_thread(
                    lambda: get_hybrid_retriever(store).retrieve("photosynthesis", final_k=1)
                )
                logger.info("Knowledge Maps retriever warmed up")
            except Exception:  # noqa: BLE001
                logger.exception("Knowledge Maps retriever warmup failed")

        asyncio.create_task(_warm_knowledge_maps_retriever(loaded))

    # Adaptive Quiz component — isolated so a missing MONGO_URI or offline DB
    # never blocks the rest of the app.
    try:
        from backend.components.adaptive_quiz.db import init_quiz_db

        await init_quiz_db()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Adaptive Quiz DB not initialised: %s", exc)
    else:
        async def _warm_emotion_model() -> None:
            try:
                from backend.components.adaptive_quiz.emotion import get_model

                await asyncio.to_thread(get_model)
            except Exception:  # noqa: BLE001
                logger.exception("Adaptive Quiz emotion model warmup failed")

        asyncio.create_task(_warm_emotion_model())

    yield
    try:
        from backend.components.adaptive_quiz.db import close_quiz_db

        await close_quiz_db()
    except Exception:  # noqa: BLE001
        pass
    _store = None
    configure_embeddings(None)


app = FastAPI(
    title="Sri Lanka G10–11 Science AI STEM Ecosystem",
    lifespan=lifespan,
)


@app.middleware("http")
async def revalidate_frontend_assets(request: Request, call_next):
    """Prevent an open browser tab from reusing an outdated UI bundle."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    """Adaptive-quiz routes want validation failures in the {success, message} shape its
    ported frontend reads. Other components keep FastAPI's default 422 payload."""
    if not request.url.path.startswith("/api/adaptive-quiz"):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
    errors = exc.errors()
    msg = "Invalid request data"
    if errors:
        raw = errors[0].get("msg", msg)
        msg = raw.split("Value error, ", 1)[-1] if "Value error, " in raw else raw
    return JSONResponse(status_code=400, content={"success": False, "message": msg})


@app.get("/api/health")
def health() -> dict:
    data_dir = settings.resolved_data_dir()
    ok = _store is not None and (data_dir / settings.faiss_index_name).is_file()
    return {
        "status": "ok" if ok else "degraded",
        "index_loaded": ok,
        "openai_configured": bool(settings.openai_api_key and settings.openai_api_key.strip()),
    }


# Include component routers under /api
app.include_router(voice_tutor_router, prefix="/api")
app.include_router(voice_tutor_router, prefix="/api/voice-tutor")
# Legacy endpoint support
app.include_router(voice_tutor_router, prefix="", include_in_schema=False)

app.include_router(narrative_learning_router, prefix="/api")
app.include_router(adaptive_quiz_router, prefix="/api")
app.include_router(knowledge_maps_router, prefix="/api")

# Static frontend serving
_frontend_dir = settings.resolved_frontend_dir()


def _serve_frontend_page(subpath: str) -> FileResponse:
    target = _frontend_dir / subpath / "index.html"
    if not target.is_file():
        # Fallback to home or root index.html
        target = _frontend_dir / "home" / "index.html"
    if not target.is_file():
        target = _frontend_dir / "index.html"
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"Page not found: {subpath}")
    return FileResponse(target)


@app.get("/", include_in_schema=False)
def serve_root() -> FileResponse:
    return _serve_frontend_page("home")


@app.get("/home", include_in_schema=False)
@app.get("/home/", include_in_schema=False)
def serve_home() -> FileResponse:
    return _serve_frontend_page("home")


@app.get("/voice-tutor", include_in_schema=False)
@app.get("/voice-tutor/", include_in_schema=False)
def serve_voice() -> FileResponse:
    return _serve_frontend_page("voice_tutor")


@app.get("/narrative-learning", include_in_schema=False)
@app.get("/narrative-learning/", include_in_schema=False)
def serve_narrative() -> FileResponse:
    return _serve_frontend_page("narrative_learning")


_quiz_dist = _frontend_dir / "adaptive_quiz" / "dist"

# index.html must always be re-checked, or browsers keep showing an old build (it
# points at the old hashed bundles) until a hard refresh. Files under assets/ have
# a content hash in their name, so they can be cached forever.
_NO_CACHE = {"Cache-Control": "no-cache, must-revalidate"}
_IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


@app.get("/adaptive-quiz", include_in_schema=False)
@app.get("/adaptive-quiz/{resource_path:path}", include_in_schema=False)
def serve_quiz(resource_path: str = "") -> FileResponse:
    """Serve the built Vite/React SPA (frontend/adaptive_quiz/dist).

    Real build artifacts (``/adaptive-quiz/assets/...``) are returned directly;
    every other path falls back to ``index.html`` for client-side routing.
    """
    if resource_path:
        candidate = (_quiz_dist / resource_path).resolve()
        if _quiz_dist in candidate.parents and candidate.is_file():
            hashed = candidate.parent == (_quiz_dist / "assets").resolve()
            return FileResponse(candidate, headers=_IMMUTABLE if hashed else _NO_CACHE)
    index = _quiz_dist / "index.html"
    if index.is_file():
        return FileResponse(index, headers=_NO_CACHE)
    raise HTTPException(
        status_code=503,
        detail=(
            "Adaptive Quiz frontend is not built. Run: "
            "cd frontend/adaptive_quiz && npm install && npm run build"
        ),
    )


@app.get("/knowledge-maps", include_in_schema=False)
@app.get("/knowledge-maps/", include_in_schema=False)
def serve_maps() -> FileResponse:
    return _serve_frontend_page("knowledge_maps")


if _frontend_dir.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=str(_frontend_dir)),
        name="frontend-static",
    )

# Serve extracted textbook images from data/extracted_images/ at /images
_images_dir = settings.resolved_data_dir() / "extracted_images"
if _images_dir.is_dir():
    app.mount(
        "/images",
        StaticFiles(directory=str(_images_dir)),
        name="textbook-images",
    )

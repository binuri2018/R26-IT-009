"""Knowledge Maps component router — Member 4: hybrid RAG verification, note
repair, OCR, synthesis, and YouTube analysis tools, plus the mind-map UI.

Two internal routers are combined into a single export so that main.py's one
`app.include_router(knowledge_maps_router, prefix="/api")` call produces both:
  - /api/m4/*  (canonical, namespaced paths)
  - /api/*     (bare aliases the ported frontend's JS calls directly, e.g.
    /api/verify, /api/synthesize, /api/upload-ocr)
"""
from __future__ import annotations

from typing import Callable

import httpx
import openai
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from backend.common.config import settings
from backend.common.vector_store import VectorStore
from backend.components.knowledge_maps import llm_client
from backend.components.knowledge_maps.evidence import resolve_document
from backend.components.knowledge_maps.hybrid_retrieval import get_hybrid_retriever
from backend.components.knowledge_maps.ocr import extract_text as run_ocr
from backend.components.knowledge_maps.ocr_review import review_image as run_ocr_review
from backend.components.knowledge_maps.correct_note import run as run_correct_note
from backend.components.knowledge_maps.repair_note import run as run_repair_note
from backend.components.knowledge_maps.schemas import (
    CorrectNoteRequest,
    CorrectNoteResponse,
    M4HealthResponse,
    OcrResponse,
    OcrReviewResponse,
    RepairNoteRequest,
    RepairNoteResponse,
    SynthesizeRequest,
    VerifyRequest,
    VerifyResponse,
    VideoSummaryRequest,
    VideoSummaryResponse,
    YoutubeRequest,
    YoutubeResponse,
    YoutubeSuggestRequest,
    YoutubeSuggestResponse,
)
from backend.components.knowledge_maps.synthesis import synthesize as run_synthesize
from backend.components.knowledge_maps.verification import verify_text as run_verify
from backend.components.knowledge_maps.youtube import analyze_video as run_youtube
from backend.components.knowledge_maps.youtube import summarize_video as run_summarize_video
from backend.components.knowledge_maps.youtube_suggest import suggest_videos as run_suggest

_m4_router = APIRouter(prefix="/m4", tags=["Knowledge Maps"])
_legacy_router = APIRouter(tags=["Knowledge Maps (legacy aliases)"], include_in_schema=False)

_store_getter: Callable[[], VectorStore] | None = None


def set_store_getter(getter: Callable[[], VectorStore]) -> None:
    """Called once by main.py at startup (mirrors the voice_tutor component)."""
    global _store_getter
    _store_getter = getter


def _get_store() -> VectorStore:
    if _store_getter is None:
        raise HTTPException(status_code=503, detail="Store getter not initialized.")
    return _store_getter()


# ─── Health ─────────────────────────────────────────────────────────────────


@_m4_router.get("/health", response_model=M4HealthResponse)
def health() -> M4HealthResponse:
    """Member-4 connectivity check. Reports LLM, YouTube, and index status.

    Does not raise on provider errors — returns them in the response so the
    panel can see exactly what's misconfigured without reading server logs.
    """
    provider = settings.m4_llm_provider
    configured = llm_client.is_configured()
    model = (
        settings.m4_ollama_model
        if provider == "ollama"
        else settings.azure_openai_deployment
    )
    if configured:
        ping = llm_client.ping()
        llm_ok = bool(ping.get("ok"))
        latency = ping.get("latency_ms")
        error = ping.get("error")
        model = ping.get("model", model)
    else:
        llm_ok = False
        latency = None
        if provider == "ollama":
            error = "OLLAMA_BASE_URL or M4_OLLAMA_MODEL is not set"
        else:
            error = (
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, or "
                "AZURE_OPENAI_DEPLOYMENT is not set"
            )

    youtube_configured = bool(settings.youtube_api_key and settings.youtube_api_key.strip())

    try:
        store = _get_store()
        index_loaded = True
    except HTTPException:
        store = None
        index_loaded = False

    retrieval_status = {
        "retrieval_method": "hybrid_rrf",
        "retriever_signature": None,
        "reranker_model": settings.m4_reranker_model,
        "reranker_revision": None,
        "reranker_loaded": False,
        "reranker_error": None,
    }
    if index_loaded:
        try:
            retrieval_status.update(get_hybrid_retriever(store).status())
        except Exception as exc:
            retrieval_status["reranker_error"] = str(exc)

    return M4HealthResponse(
        llm_provider=provider,
        llm_configured=configured,
        llm_ok=llm_ok,
        llm_model=model,
        llm_latency_ms=latency,
        llm_error=error,
        youtube_api_configured=youtube_configured,
        index_loaded=index_loaded,
        **retrieval_status,
    )


@_m4_router.get("/documents/{document_id}", response_class=FileResponse)
def get_document(document_id: str) -> FileResponse:
    """Serve a curriculum PDF selected by an opaque, allowlisted document ID."""
    path = resolve_document(document_id, settings.resolved_resource_dir())
    if path is None:
        raise HTTPException(status_code=404, detail="Curriculum document not found.")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=path.name,
        content_disposition_type="inline",
        headers={
            "Cache-Control": "private, max-age=300",
            "X-Content-Type-Options": "nosniff",
        },
    )


# ─── Helper: map common errors to HTTP status codes consistently ──────────


def _map_error(exc: Exception) -> HTTPException:
    """Translate Member-4 internal errors into actionable HTTP responses.

    - RuntimeError from llm_client   → 501 (configuration/model issue)
    - SDK/httpx network errors       → 502 (provider unreachable)
    - ValueError                     → 400 (bad input, e.g. unknown mode)
    - anything else                  → 502 with redacted message
    """
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        if settings.m4_llm_provider == "ollama":
            return HTTPException(
                status_code=502,
                detail=(
                    f"Ollama not reachable at {settings.ollama_base_url}. "
                    f"Run `ollama serve` and `ollama pull {settings.m4_ollama_model}`."
                ),
            )
        return HTTPException(
            status_code=502,
            detail=(
                "Azure OpenAI is not reachable. Check AZURE_OPENAI_ENDPOINT in .env "
                "(copy the exact URL from Azure Portal → Keys and Endpoint). "
                f"Detail: {exc!s}"
            ),
        )
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=501, detail=str(exc))
    if isinstance(exc, httpx.HTTPError):
        return HTTPException(status_code=502, detail=f"Upstream request failed: {exc!s}")
    return HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc!s}")


# ─── Synthesize ────────────────────────────────────────────────────────────


@_m4_router.post("/synthesize")
@_legacy_router.post("/synthesize")
def synthesize_endpoint(body: SynthesizeRequest) -> dict:
    """Generate flashcards / mindmap / structured notes from text.

    Response shape varies by mode:
      - flashcards → {"flashcards": [{"front", "back"}, ...]}
      - mindmap    → {"nodes": [...], "edges": [...]}
      - notes      → {"sections": [{"heading", "bullets", "key_terms"}, ...]}
    """
    try:
        return run_synthesize(body.text, body.mode, body.language)
    except Exception as exc:
        raise _map_error(exc) from exc


# ─── OCR ───────────────────────────────────────────────────────────────────


@_m4_router.post("/upload-ocr", response_model=OcrResponse)
@_legacy_router.post("/upload-ocr", response_model=OcrResponse)
async def upload_ocr_endpoint(file: UploadFile = File(...)) -> OcrResponse:
    """Extract text from a handwritten/printed photo via the selected vision LLM.

    Frontend posts multipart form-data with field name 'file'.
    Returns {"text": "...", "signed_url": null}.
    """
    try:
        data = await file.read()
        text = await run_in_threadpool(run_ocr, data, file.content_type)
    except ValueError as exc:
        # bad input (size / mime / empty)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise _map_error(exc) from exc
    return OcrResponse(text=text, signed_url=None)


@_m4_router.post("/ocr-review", response_model=OcrReviewResponse)
async def ocr_review_endpoint(file: UploadFile = File(...)) -> OcrReviewResponse:
    """Extract OCR text plus reviewable uncertainty and recovery suggestions."""
    try:
        data = await file.read()
        return await run_in_threadpool(run_ocr_review, data, file.content_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise _map_error(exc) from exc


# ─── Verify ────────────────────────────────────────────────────────────────


@_m4_router.post("/verify", response_model=VerifyResponse)
@_legacy_router.post("/verify", response_model=VerifyResponse)
def verify_endpoint(body: VerifyRequest) -> VerifyResponse:
    """Verify each factual claim in the student text against the syllabus.

    Evidence-gated pipeline:
      1. Extract claims (structured LLM output).
      2. Retrieve syllabus context for each claim → structured verdict.
      3. Claims without decisive cited syllabus evidence use approved URL-backed web evidence.
      4. Undecidable and failed claims are excluded from the knowledge score.

    Response shape matches the ported frontend's renderVerificationResult:
      {"overall_score": 0..1|null, "score_summary": {...}, "claims": [...]}
    """
    store = _get_store()
    try:
        return run_verify(body.text, store, body.response_language)
    except Exception as exc:
        raise _map_error(exc) from exc


# ─── YouTube ───────────────────────────────────────────────────────────────


@_m4_router.post("/youtube", response_model=YoutubeResponse)
@_legacy_router.post("/youtube", response_model=YoutubeResponse)
def youtube_endpoint(body: YoutubeRequest) -> YoutubeResponse:
    """Analyse a YouTube video: fetch transcript → verify → synthesize.

    Response fields consumed by the frontend:
      transcript_excerpt  → yt-transcript-preview text
      verification        → renderVerificationResult (score + claim cards)
      notes                → renderStructuredNotes
      flashcards           → renderFlashcards
      mind_map             → renderMindMap

    Error codes:
      400 — bad / unrecognised URL
      422 — captions disabled or video unavailable
      501 — selected LLM provider not configured
      503 — vector index not loaded
    """
    store = _get_store()
    try:
        return run_youtube(body.url, body.language, store)
    except ValueError as exc:
        # Bad URL or transcript unavailable — user-facing, actionable message
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise _map_error(exc) from exc


# ─── YouTube Suggest ───────────────────────────────────────────────────────


@_m4_router.post("/youtube-suggest", response_model=YoutubeSuggestResponse)
def youtube_suggest_endpoint(body: YoutubeSuggestRequest) -> YoutubeSuggestResponse:
    """Search YouTube Data API v3 for videos matching a topic.

    Returns up to max_results video cards (title, url, thumbnail, channel).
    The frontend renders them as a clickable grid; clicking a card fills
    the URL input and auto-runs the full YouTube analysis.

    Error codes:
      400 — empty topic
      501 — YOUTUBE_API_KEY not set in .env
      429 — YouTube API quota exceeded
    """
    try:
        videos = run_suggest(body.topic, body.max_results)
        return YoutubeSuggestResponse(videos=videos)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        msg = str(exc)
        status = 429 if "quota" in msg.lower() else 501
        raise HTTPException(status_code=status, detail=msg) from exc
    except Exception as exc:
        raise _map_error(exc) from exc


# ─── Video Summary ─────────────────────────────────────────────────────────


@_m4_router.post("/video-summary", response_model=VideoSummaryResponse)
def video_summary_endpoint(body: VideoSummaryRequest) -> VideoSummaryResponse:
    """Fetch a YouTube transcript and return a concise study-note summary.

    Lighter than /youtube — no verification or synthesis passes.

    Error codes:
      422 — captions disabled, bad URL, video unavailable
      501 — selected LLM provider not configured
    """
    try:
        result = run_summarize_video(body.url, body.language)
        return VideoSummaryResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise _map_error(exc) from exc


# ─── Correct Note ──────────────────────────────────────────────────────────


@_m4_router.post("/correct-note", response_model=CorrectNoteResponse)
@_legacy_router.post("/correct-note", response_model=CorrectNoteResponse)
def correct_note_endpoint(body: CorrectNoteRequest) -> CorrectNoteResponse:
    """Orchestrate: verify claims → keep correct → rewrite as clean note.

    Flow:
      1. verify_text()  — extract + verify every claim (syllabus + web fallback)
      2. Split into kept (verdict=correct) and dropped (incorrect/incomplete)
      3. The selected LLM rewrites a clean note from kept claims only

    Error codes:
      501 — selected LLM provider not configured
      503 — vector index not loaded
    """
    store = _get_store()
    try:
        return run_correct_note(body.text, body.response_language, store)
    except Exception as exc:
        raise _map_error(exc) from exc


# ─── Structure-preserving repair note ─────────────────────────────────────


@_m4_router.post("/repair-note", response_model=RepairNoteResponse)
def repair_note_endpoint(body: RepairNoteRequest) -> RepairNoteResponse:
    """Repair cited mistakes, independently re-verify them, and preserve note blocks."""
    store = _get_store()
    try:
        return run_repair_note(
            body.text,
            body.response_language,
            store,
            preserve_structure=body.preserve_structure,
        )
    except Exception as exc:
        raise _map_error(exc) from exc


router = APIRouter(tags=["Knowledge Maps"])
router.include_router(_m4_router)
router.include_router(_legacy_router)

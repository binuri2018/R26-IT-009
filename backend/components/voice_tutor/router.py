"""Voice Tutor API routes (ask RAG, transcribe audio, synthesize speech)."""
from __future__ import annotations

from typing import Callable

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from backend.common.config import settings
from backend.common.embeddings import EmbeddingLoadError
from backend.common.image_retrieval import find_images_for_hits
from backend.common.retrieval import format_context_for_llm, retrieve
from backend.common.vector_store import VectorStore
from backend.common.visual_explainer import retrieve_visual_explanation
from backend.components.voice_tutor.audio_services import synthesize_speech, transcribe_bytes
from backend.components.voice_tutor.llm import REFUSAL_EN, REFUSAL_SI, answer_question
from backend.components.voice_tutor.schemas import (
    AskRequest,
    AskResponse,
    ImageItem,
    SourceItem,
    TranscribeResponse,
    TtsRequest,
    VisualItem,
)

router = APIRouter(tags=["Voice Tutor"])

_store_getter: Callable[[], VectorStore] | None = None


def set_store_getter(getter: Callable[[], VectorStore]) -> None:
    global _store_getter
    _store_getter = getter


def _get_store() -> VectorStore:
    if _store_getter is None:
        raise HTTPException(status_code=503, detail="Store getter not initialized.")
    return _store_getter()


def _llm_connection_detail(exc: httpx.HTTPError) -> str:
    raw = f"{type(exc).__name__}: {exc!s}"
    using_openai = bool(settings.openai_api_key and settings.openai_api_key.strip())
    if using_openai:
        return f"OpenAI request failed ({raw}). Check internet, VPN, firewall, and OPENAI_API_KEY."
    refused = (
        "10061" in raw
        or "actively refused" in raw.lower()
        or isinstance(exc, httpx.ConnectError)
    )
    if refused:
        m = settings.ollama_chat_model
        return (
            "No chat model reachable: Ollama is not accepting connections at "
            f"{settings.ollama_base_url}. Install/start Ollama, then run "
            f"`ollama pull {m}`, or set OPENAI_API_KEY in .env to use OpenAI instead of Ollama. "
            f"(Detail: {raw})"
        )
    if "404" in raw:
        m = settings.ollama_chat_model
        return (
            f"Ollama is running but model '{m}' was not found. "
            f"Run `ollama pull {m}` or set OLLAMA_CHAT_MODEL to an installed model "
            f"(this machine has llama3.1). (Detail: {raw})"
        )
    return f"LLM backend request failed: {raw}"


@router.post("/ask", response_model=AskResponse)
def ask_question_endpoint(body: AskRequest) -> AskResponse:
    store = _get_store()
    try:
        hits = retrieve(store, body.question.strip())
    except EmbeddingLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not hits:
        lang = (body.response_language or "auto").lower()
        msg = REFUSAL_SI if lang == "si" else REFUSAL_EN
        return AskResponse(answer=msg, sources=[])

    context = format_context_for_llm(hits)
    try:
        answer = answer_question(
            context,
            body.question.strip(),
            response_language=body.response_language,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=_llm_connection_detail(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM error: {exc!s}",
        ) from exc

    sources = [
        SourceItem(
            score=h.get("score"),
            grade=h.get("grade"),
            subject_area=h.get("subject_area"),
            topic=h.get("topic"),
            subtopic=h.get("subtopic"),
            document_type=h.get("document_type"),
            source_file=h.get("source_file"),
            page_start=h.get("page_start"),
            page_end=h.get("page_end"),
        )
        for h in hits
    ]

    # 1. Local Textbook Image retrieval — purely additive, never breaks answer generation
    images: list[ImageItem] = []
    try:
        raw_images = find_images_for_hits(hits, body.question.strip())
        images = [ImageItem(**img) for img in raw_images]
    except Exception:
        pass  # silently degrade — text answer is unaffected

    # 2. Dynamic Internet Visual Explanation Retrieval
    visual_required = False
    visual: VisualItem | None = None
    try:
        visual_required, raw_visual = retrieve_visual_explanation(body.question.strip())
        if raw_visual:
            visual = VisualItem(**raw_visual)
    except Exception:
        pass  # visual retrieval failures never affect text answers

    return AskResponse(
        answer=answer,
        sources=sources,
        images=images,
        visual_required=visual_required,
        visual=visual,
    )


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio_endpoint(file: UploadFile = File(...)) -> TranscribeResponse:
    try:
        data = await file.read()
        text = transcribe_bytes(data, file.filename or "audio.webm")
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Transcription failed: {exc!s}",
        ) from exc
    return TranscribeResponse(text=text)


@router.post("/tts")
async def tts_endpoint(body: TtsRequest) -> Response:
    try:
        audio = synthesize_speech(body.text, body.voice)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"TTS failed: {exc!s}",
        ) from exc
    return Response(content=audio, media_type="audio/mpeg")

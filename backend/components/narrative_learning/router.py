"""Narrative Learning API: persona theme + textbook RAG story."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.components.narrative_learning.classifier import classify, log_theme_feedback, survey_spec
from backend.components.narrative_learning.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    GenerateRequest,
    GenerateResponse,
    ThemeFeedbackRequest,
)
from backend.components.narrative_learning.syllabus_data import CHAPTER_MAP

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/narrative-learning", tags=["Narrative Learning"])

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from backend.components.narrative_learning.story_engine import StoryEngine

        _engine = StoryEngine()
    return _engine


@router.get("/status")
def status() -> dict:
    from backend.components.narrative_learning.model_config import VECTOR_DB_DIR

    spec = survey_spec()
    return {
        "status": "ready",
        "module": "Narrative Learning",
        "index_path": VECTOR_DB_DIR,
        "classifier_mode": spec["classifier_mode"],
        "output_label": spec["output_label"],
    }


@router.get("/survey")
def survey() -> dict:
    return survey_spec()


@router.post("/classify", response_model=ClassifyResponse)
def classify_student(body: ClassifyRequest) -> ClassifyResponse:
    result = classify(body.model_dump())
    return ClassifyResponse(**result)


@router.post("/theme-feedback")
def theme_feedback(body: ThemeFeedbackRequest) -> dict:
    log_theme_feedback(body.theme, body.matched, body.interest, body.aspiration)
    return {"ok": True}


@router.get("/chapters")
def chapters() -> dict:
    return {"books": CHAPTER_MAP}


@router.post("/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest) -> GenerateResponse:
    try:
        engine = _get_engine()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Narrative engine is not ready: {exc}",
        ) from exc

    if (body.theme or "").strip():
        theme = body.theme.strip()
        classified = {"method": "preclassified", "theme": theme}
    else:
        classified = classify({
            "grade": body.grade,
            "interest": body.interest,
            "aspiration": body.aspiration,
            "struggle_level": body.struggle_level,
            "story_style": body.story_style,
            "story_opening": body.story_opening,
        })
        theme = classified["theme"]
    try:
        data = engine.generate_chapter(
            student_theme=theme,
            topic=body.topic,
            diagnostic_query=body.diagnostic,
            interest=body.interest,
            aspiration=body.aspiration,
            struggle_level=body.struggle_level,
            book_name=body.book,
        )
    except Exception as exc:
        logger.exception("Narrative generate_chapter failed")
        raise HTTPException(
            status_code=502,
            detail=f"Story generation failed: {exc}",
        ) from exc
    if not data or not (data.get("story") or "").strip():
        raise HTTPException(status_code=502, detail="Story generation failed. Please try again.")

    return GenerateResponse(
        theme=theme,
        classifier_method=classified.get("method") or "",
        science_intro=data.get("science_intro") or {},
        story=data.get("story") or "",
        key_definitions=data.get("key_definitions") or [],
        key_equations=data.get("key_equations") or [],
        exam_bullets=data.get("exam_bullets") or [],
        quiz_topic=data.get("quiz_topic") or body.topic,
        sources=data.get("sources") or [],
    )

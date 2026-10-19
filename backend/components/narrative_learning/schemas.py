from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    grade: str = ""
    interest: str = Field(min_length=1)
    aspiration: str = Field(min_length=1)
    struggle_level: str = "Medium"
    story_style: str = ""
    story_opening: str = ""


class ClassifyResponse(BaseModel):
    theme: str
    output_label: str = "Story_Theme"
    method: str
    interest: str = ""
    aspiration: str = ""
    struggle_level: str = "Medium"
    grade: str = ""
    blurb: str = ""
    predicted_theme: str = ""
    model_confidence: float | None = None
    model_agrees: bool | None = None
    model_name: str = ""


class ThemeFeedbackRequest(BaseModel):
    theme: str = Field(min_length=1)
    matched: bool
    interest: str = ""
    aspiration: str = ""


class GenerateRequest(BaseModel):
    interest: str = Field(min_length=1)
    aspiration: str = Field(min_length=1)
    struggle_level: str = "Medium"
    book: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    diagnostic: str = "Explain the core concept in simple terms."
    grade: str = ""
    story_style: str = ""
    story_opening: str = ""
    theme: str = ""


class StoryFeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    event: Literal["rating", "check"]
    model: str = ""
    theme: str = ""
    book: str = ""
    topic: str = ""
    grade: str = ""
    struggle_level: str = ""
    rating: int | None = Field(default=None, ge=1, le=5)
    check_correct: bool | None = None
    check_source: str = ""


class GenerateResponse(BaseModel):
    theme: str
    model: str = ""
    classifier_method: str = ""
    science_intro: dict[str, Any] = Field(default_factory=dict)
    story: str = ""
    key_definitions: list[dict[str, Any]] = Field(default_factory=list)
    key_equations: list[dict[str, Any]] = Field(default_factory=list)
    exam_bullets: list[str] = Field(default_factory=list)
    check_question: dict[str, Any] = Field(default_factory=dict)
    quiz_topic: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)

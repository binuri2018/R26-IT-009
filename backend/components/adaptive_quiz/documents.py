"""Beanie document models for the Adaptive Quiz component.

Ported from the standalone Adaptive-quiz project; collection names are kept so an
existing ``stem_assessment`` database works unchanged.
"""
import re
from datetime import UTC, datetime
from typing import Annotated, Any

from beanie import Document, Indexed
from pydantic import BaseModel, Field
from pydantic.functional_validators import field_validator


class Lesson(Document):
    lessonId: Annotated[str, Indexed(unique=True)]
    title: str
    subject: str
    gradeLevel: str = "Year 1"
    arModuleId: str | None = None
    description: str = ""
    thumbnailUrl: str = ""
    estimatedDuration: int = 30
    isActive: bool = True
    # Textbook position (Grade 10/11 · Part I/II · chapter n) — None for ad-hoc lessons
    grade: int | None = None
    part: int | None = None
    chapterNo: int | None = None

    class Settings:
        name = "lessons"


class Question(Document):
    lessonId: Annotated[str, Indexed()]
    questionId: Annotated[str, Indexed(unique=True)]
    quizLevel: int
    conceptTag: str
    questionText: str
    options: list[str]
    correctAnswer: str
    shortTheoryExplanation: str
    hint: str
    order: int = 0
    source: str = "manual"
    # PDF-generated questions start as "pending" and reach students only once a
    # teacher approves them. Rows without the field (seed/manual) count as approved.
    status: str = "approved"

    @field_validator("options")
    @classmethod
    def options_len(cls, v: list[str]) -> list[str]:
        if not (2 <= len(v) <= 6):
            raise ValueError("A question must have 2-6 options")
        return v

    @field_validator("quizLevel")
    @classmethod
    def quiz_level(cls, v: int) -> int:
        if v not in (1, 2, 3):
            raise ValueError("quizLevel must be 1, 2, or 3")
        return v

    class Settings:
        name = "questions"


class Student(Document):
    studentId: Annotated[str, Indexed(unique=True)]
    name: str
    email: Annotated[str, Indexed(unique=True)]
    password: str
    gradeLevel: str = "Year 1"
    completedLessons: list[Any] = Field(default_factory=list)

    class Settings:
        name = "students"


class StudentResponse(Document):
    studentId: Annotated[str, Indexed()]
    lessonId: Annotated[str, Indexed()]
    questionId: str
    quizLevel: int
    conceptTag: str
    selectedAnswer: str = ""
    correctness: int  # 0 or 1
    responseTime: float
    answerChanges: float = 0
    attempts: float = 1
    detectedExpression: str = "neutral"
    learningState: str = "partial_understanding"
    cognitiveLoad: float = 0
    adaptiveLevel: int = 0
    hintUsed: bool = False
    sessionId: Annotated[str, Indexed()]

    @field_validator("correctness")
    @classmethod
    def corr(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("correctness must be 0 or 1")
        return v

    @field_validator("quizLevel")
    @classmethod
    def ql(cls, v: int) -> int:
        if v not in (1, 2, 3):
            raise ValueError("quizLevel must be 1, 2, or 3")
        return v

    class Settings:
        name = "studentresponses"


class AssessmentReport(Document):
    studentId: Annotated[str, Indexed()]
    lessonId: Annotated[str, Indexed()]
    sessionId: Annotated[str, Indexed(unique=True)]
    totalScore: float
    levelScores: dict[str, float] = Field(default_factory=dict)
    conceptPerformance: dict[str, float] = Field(default_factory=dict)
    mostCommonExpression: str = "neutral"
    expressionFrequency: dict[str, int] = Field(default_factory=dict)
    weakAreas: list[str] = Field(default_factory=list)
    progressSummary: str = ""
    hintUsageCount: int = 0
    learningStateDistribution: dict[str, int] = Field(default_factory=dict)
    totalTime: float = 0

    createdAt: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updatedAt: datetime = Field(default_factory=lambda: datetime.now(UTC))

    class Settings:
        name = "assessmentreports"


class StudyPlan(Document):
    """Student-customized weekly study timetable for a lesson — one per (student, lesson).

    ``schedule`` mirrors the frontend ``generateWeeklyTimetable()`` shape: a list of 7 day
    objects. Stored as opaque dicts since the shape is owned by the frontend.
    """

    studentId: Annotated[str, Indexed()]
    lessonId: Annotated[str, Indexed()]
    schedule: list[dict[str, Any]]
    intensity: str = "medium"
    weeklyMinutes: float = 0
    weekRangeLabel: str = ""
    updatedAt: datetime = Field(default_factory=lambda: datetime.now(UTC))

    class Settings:
        name = "studyplans"


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# Pydantic request bodies (not persisted)
class RegisterBody(BaseModel):
    studentId: str
    name: str
    email: str
    password: str
    gradeLevel: str | None = None

    @field_validator("studentId")
    @classmethod
    def validate_student_id(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Student ID must be at least 3 characters")
        if " " in v:
            raise ValueError("Student ID cannot contain spaces")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Name must be at least 2 characters")
        return v

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("Enter a valid email address")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class LoginBody(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("Enter a valid email address")
        return v


ALL_DOCUMENTS = [Student, Lesson, Question, StudentResponse, AssessmentReport, StudyPlan]

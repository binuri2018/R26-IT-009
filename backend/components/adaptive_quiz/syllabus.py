"""One quiz lesson per chapter of the Grade 10/11 science textbooks.

The chapter list comes from the Narrative Learning syllabus map, so every component
offers students exactly the same chapters. Lessons are created once (idempotent) and
start empty — a teacher fills each one by generating + approving questions from a PDF.
"""
from __future__ import annotations

import logging
import re

from backend.components.adaptive_quiz.documents import Lesson
from backend.components.narrative_learning.syllabus_data import CHAPTER_MAP

logger = logging.getLogger(__name__)

_BOOK_RE = re.compile(r"Grade\s+(\d+)\s*-\s*Part\s+(I+)", re.IGNORECASE)


def syllabus_lessons() -> list[dict]:
    lessons: list[dict] = []
    for book, chapters in CHAPTER_MAP.items():
        m = _BOOK_RE.search(book)
        if not m:
            continue
        grade, part = int(m.group(1)), len(m.group(2))   # "II" → 2
        for no, title in enumerate(chapters, start=1):
            lessons.append({
                "lessonId": f"g{grade}-p{part}-ch{no:02d}",
                "title": title,
                "subject": "Science",
                "gradeLevel": f"Grade {grade}",
                "description": f"{book} · Chapter {no}",
                "grade": grade,
                "part": part,
                "chapterNo": no,
            })
    return lessons


async def ensure_syllabus_lessons() -> int:
    """Insert any chapter lesson that doesn't exist yet. Never touches existing lessons."""
    existing = {l.lessonId for l in await Lesson.find_all().to_list()}
    missing = [Lesson(**d) for d in syllabus_lessons() if d["lessonId"] not in existing]
    if missing:
        await Lesson.insert_many(missing)
        logger.info("Adaptive Quiz: created %d textbook chapter lessons", len(missing))
    return len(missing)

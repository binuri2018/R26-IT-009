"""Generate pending Adaptive Quiz questions for every textbook chapter.

Splits each Grade 10/11 science textbook in ``Resource/`` into its chapters, runs the
T5 question generator on each chapter's pages only, and saves the questions as
"pending" on that chapter's lesson (``g11-p1-ch01`` …). Nothing reaches students
until a teacher approves the questions in the Teacher Dashboard.

Run from the project root:
  python scripts/generate_chapter_questions.py --dry-run          # show chapter page ranges only
  python scripts/generate_chapter_questions.py                    # generate all 35 chapters
  python scripts/generate_chapter_questions.py --only g11-p1-ch01 --replace

Re-running skips chapters that already have generated questions; ``--replace``
regenerates a chapter's *pending* questions (approved ones are never touched).
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.components.narrative_learning.syllabus_data import CHAPTER_MAP  # noqa: E402

BOOKS = {  # syllabus key → (grade, part, PDF file)
    "Grade 10 - Part I": (10, 1, "science Grade 10 Part 1.pdf"),
    "Grade 10 - Part II": (10, 2, "science Grade 10 Part 2.pdf"),
    "Grade 11 - Part I": (11, 1, "science Grade 11 Part 1.pdf"),
    "Grade 11 - Part II": (11, 2, "science Grade 11 Part 2.pdf"),
}
# Book chapter titles that differ from the syllabus map's wording
TITLE_ALIASES = {"Organizational levels and interactions of biosphere": "Biosphere"}
TOP_LINES = 4          # running header / chapter opener lives in a page's first lines


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _is_contents_page(lines: list[str]) -> bool:
    """A table-of-contents page: many lines that end in a page number."""
    return sum(bool(re.search(r"\s\d{1,3}\s*$", l)) for l in lines) >= 5


def find_chapter_ranges(pdf_path: Path, titles: list[str], first_no: int) -> list[dict]:
    """[{no, title, start, end}] with 0-based, end-exclusive PDF page indexes."""
    with pdfplumber.open(pdf_path) as pdf:
        tops = []
        for page in pdf.pages:
            lines = [l.strip() for l in (page.dedupe_chars().extract_text() or "").split("\n") if l.strip()]
            tops.append(None if _is_contents_page(lines) else _norm(" ".join(lines[:TOP_LINES])))
        n_pages = len(pdf.pages)

    starts: list[int] = []
    cursor = 0
    for i, title in enumerate(titles):
        # Openers often wrap the title over two lines ("Structure and functions of" / …),
        # so match its first 18 letters — enough to tell chapters apart.
        key = _norm(TITLE_ALIASES.get(title, title))[:18]
        no = first_no + i
        # First page whose opener / running header (top lines) carries the chapter title
        hit = next((p for p in range(cursor, n_pages) if tops[p] and key in tops[p]), None)
        if hit is None:
            raise SystemExit(f"Could not find chapter {no} '{title}' in {pdf_path.name}")
        starts.append(hit)
        cursor = hit + 1

    ranges = []
    for i, title in enumerate(titles):
        end = starts[i + 1] if i + 1 < len(starts) else n_pages
        ranges.append({"no": first_no + i, "title": title, "start": starts[i], "end": end})
    return ranges


def plan() -> list[dict]:
    chapters = []
    for book, titles in CHAPTER_MAP.items():
        grade, part, fname = BOOKS[book]
        first_no = 1 if part == 1 else len(CHAPTER_MAP[f"Grade {grade} - Part I"]) + 1
        for idx, r in enumerate(find_chapter_ranges(ROOT / "Resource" / fname, titles, first_no), start=1):
            chapters.append({
                **r,
                "lessonId": f"g{grade}-p{part}-ch{idx:02d}",
                "pdf": ROOT / "Resource" / fname,
                "book": book,
            })
    return chapters


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="only print the detected chapter page ranges")
    ap.add_argument("--per-chapter", type=int, default=18, help="questions per chapter (multiple of 3)")
    ap.add_argument("--only", nargs="*", help="lesson IDs to (re)generate, e.g. g11-p1-ch01")
    ap.add_argument("--replace", action="store_true", help="regenerate chapters that already have pending questions")
    args = ap.parse_args()

    chapters = plan()
    if args.only:
        chapters = [c for c in chapters if c["lessonId"] in set(args.only)]

    for c in chapters:
        print(f"{c['lessonId']}  ch{c['no']:>2}  PDF pages {c['start'] + 1:>3}-{c['end']:>3} "
              f"({c['end'] - c['start']:>2} p)  {c['title']}")
    if args.dry_run:
        return

    sys.path.insert(0, str(ROOT / "ml-service"))
    from modules.question_generator import QuestionGenerator  # noqa: E402 — heavy (torch/T5)

    from backend.components.adaptive_quiz.db import close_quiz_db, init_quiz_db
    from backend.components.adaptive_quiz.documents import Question

    await init_quiz_db()            # also creates the 35 chapter lessons if missing
    gen = QuestionGenerator()
    total = 0
    try:
        for c in chapters:
            lesson_id = c["lessonId"]
            generated = Question.find(Question.lessonId == lesson_id, Question.source == "pdf_generated")
            if await generated.count() and not args.replace:
                print(f"skip {lesson_id}: already has generated questions (use --replace)")
                continue
            if args.replace:
                await Question.find(Question.lessonId == lesson_id, Question.status == "pending").delete()

            t = time.perf_counter()
            try:
                out = gen.generate_from_pdf(str(c["pdf"]), args.per_chapter, "General",
                                            pages=list(range(c["start"], c["end"])))
            except ValueError as e:
                print(f"FAIL {lesson_id}: {e}")
                continue

            docs = []
            for level_key, qs in out.items():
                for order, q in enumerate(qs, start=1):
                    docs.append(Question(
                        questionId=f"{lesson_id}-{q['questionId']}",
                        lessonId=lesson_id,
                        quizLevel=int(q["quizLevel"]),
                        conceptTag=q["conceptTag"],
                        questionText=q["questionText"],
                        options=q["options"],
                        correctAnswer=q["correctAnswer"],
                        shortTheoryExplanation=q["shortTheoryExplanation"],
                        hint=q["hint"],
                        order=order,
                        source="pdf_generated",
                        status="pending",
                    ))
            if docs:
                await Question.insert_many(docs)
            total += len(docs)
            levels = "/".join(str(len(out[k])) for k in ("level1", "level2", "level3"))
            tags = sorted({d.conceptTag for d in docs})
            print(f"ok   {lesson_id}: {len(docs):>2} questions ({levels}) in {time.perf_counter() - t:4.1f}s  "
                  f"tags: {', '.join(tags)[:110]}")
    finally:
        await close_quiz_db()
    print(f"\nDone — {total} pending questions. Review them in the Teacher Dashboard.")


if __name__ == "__main__":
    asyncio.run(main())

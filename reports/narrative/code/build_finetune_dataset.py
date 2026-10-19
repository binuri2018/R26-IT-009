"""Build the QLoRA fine-tuning dataset for the Narrative Learning story writer.

Knowledge distillation: a large *teacher* model (default gemma4:cloud via Ollama,
or Groq) writes lessons through the real app pipeline - same retrieval, same
prompt - and the exact prompt the app sends is recorded with the teacher's
answer. Only answers that pass the quality gates are kept:

  * valid JSON with every required field
  * usable quick-check question
  * a 3-paragraph story (not the textbook fallback)
  * definition terms that really appear in the story

Topics are split train/test BY TOPIC, so the fine-tuned model is evaluated on
chapters it never saw during training.

Usage (repo root, Ollama running):

  python scripts/build_finetune_dataset.py --per-topic 4
  python scripts/build_finetune_dataset.py --teacher groq:qwen/qwen3.6-27b --per-topic 4
  # teachers review data/finetune/review.csv (approved = yes/no), then:
  python scripts/build_finetune_dataset.py --apply-review data/finetune/review.csv

Outputs in data/finetune/:
  train.jsonl / test.jsonl   chat format: {"messages": [{"role": "user"}, {"role": "assistant"}], "meta": {...}}
  review.csv                 one row per sample for teacher review
  test_topics.txt            'Book|Topic' lines for scripts/compare_story_models.py --topic-file
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "backend").is_dir())
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.runnables import RunnableLambda  # noqa: E402

import backend.components.narrative_learning.story_engine as se  # noqa: E402
from backend.components.narrative_learning.classifier import THEMES, survey_spec  # noqa: E402
from backend.components.narrative_learning.model_config import OLLAMA_BASE_URL  # noqa: E402
from backend.components.narrative_learning.syllabus_data import CHAPTER_MAP  # noqa: E402
from compare_story_models import (  # noqa: E402
    FALLBACK_PREFIX,
    check_is_valid,
    field_present,
    load_textbook_index,
)

OUT_DIR = _REPO / "data" / "finetune"
KEY_ORDER = ["science_intro", "key_definitions", "key_equations", "exam_bullets",
             "check_question", "quiz_topic", "story"]
DIAGNOSTICS = [
    "Explain the core concept in simple terms.",
    "I get confused by the equations.",
    "Help me remember the definitions for the exam.",
    "Show me where this happens in real life.",
]


def make_teacher(spec: str):
    provider, _, model = spec.partition(":")
    if provider == "groq":
        from langchain_groq import ChatGroq

        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            raise SystemExit("GROQ_API_KEY missing in .env")
        return ChatGroq(model=model, api_key=key, temperature=0.6, max_tokens=2200,
                        reasoning_format="hidden", reasoning_effort="none",
                        ).bind(response_format={"type": "json_object"})
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=0.6, format="json", base_url=OLLAMA_BASE_URL,
                          num_ctx=8192, num_predict=2200)
    raise SystemExit(f"Unknown teacher {spec!r}; use ollama:<model> or groq:<model>")


def quality_gate(data: dict) -> str:
    """Return '' if the sample is good enough to train on, else the reason."""
    if not data:
        return "no data"
    missing = [k for k in KEY_ORDER if not field_present(data, k)]
    if missing:
        return f"missing {missing}"
    story = data["story"]
    if story.startswith(FALLBACK_PREFIX):
        return "fallback"
    if story.count("\n\n") < 2 or len(story.split()) < 160:
        return "thin story"
    if not check_is_valid(data["check_question"]):
        return "bad check_question"
    terms = [d.get("term", "") for d in data["key_definitions"] if isinstance(d, dict)]
    if not terms or sum(t.lower() in story.lower() for t in terms) < max(1, len(terms) // 2):
        return "definition terms not in story"
    return ""


def apply_review(review_csv: Path) -> None:
    rejected = set()
    with open(review_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("approved") or "").strip().lower() in {"no", "n", "0", "false", "reject"}:
                rejected.add(row["id"])
    for split in ("train", "test"):
        src = OUT_DIR / f"{split}.jsonl"
        if not src.is_file():
            continue
        kept = [l for l in src.read_text(encoding="utf-8").splitlines()
                if l.strip() and json.loads(l)["meta"]["id"] not in rejected]
        dst = OUT_DIR / f"{split}_approved.jsonl"
        dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"{dst.name}: {len(kept)} samples ({len(rejected)} rejected ids in review)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="ollama:gemma4:cloud")
    ap.add_argument("--per-topic", type=int, default=4, help="Samples per chapter (different theme/profile)")
    ap.add_argument("--test-share", type=float, default=0.2, help="Share of chapters held out for testing")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N kept samples (0 = no limit)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply-review", type=Path)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.apply_review:
        apply_review(args.apply_review)
        return

    rng = random.Random(args.seed)
    spec = {q["id"]: [o["value"] for o in q["options"]] for q in survey_spec()["questions"]}
    chapters = [(book, t) for book, topics in CHAPTER_MAP.items() for t in topics]
    shuffled = chapters[:]
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * args.test_share))
    test_set = set(shuffled[:n_test])
    (OUT_DIR / "test_topics.txt").write_text(
        "\n".join(f"{b}|{t}" for b, t in chapters if (b, t) in test_set) + "\n", encoding="utf-8")

    done_ids = set()
    for split in ("train", "test"):
        p = OUT_DIR / f"{split}.jsonl"
        if p.is_file():
            done_ids |= {json.loads(l)["meta"]["id"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}

    load_textbook_index()
    engine = se.StoryEngine()
    teacher = make_teacher(args.teacher)
    prompts: list[str] = []

    def recording_teacher(prompt_value):
        prompts.append(prompt_value.to_string())
        return teacher.invoke(prompt_value)

    engine.llm = RunnableLambda(recording_teacher)

    review_path = OUT_DIR / "review.csv"
    review_new = not review_path.is_file()
    review_f = open(review_path, "a", newline="", encoding="utf-8")
    review = csv.DictWriter(review_f, fieldnames=[
        "id", "split", "book", "topic", "theme", "struggle_level", "story",
        "check_question", "check_options", "approved", "reviewer_note"])
    if review_new:
        review.writeheader()

    kept = skipped = 0
    for book, topic in chapters:
        split = "test" if (book, topic) in test_set else "train"
        themes = rng.sample(THEMES, k=min(args.per_topic, len(THEMES)))
        for j, theme in enumerate(themes):
            sid = f"{book}|{topic}|{j}"
            if sid in done_ids:
                continue
            profile = {
                "interest": rng.choice(spec["interest"]),
                "aspiration": rng.choice(spec["aspiration"]),
                "struggle_level": rng.choice(spec["struggle_level"]),
                "diagnostic": rng.choice(DIAGNOSTICS),
            }
            prompts.clear()
            t0 = time.perf_counter()
            try:
                data = engine.generate_chapter(
                    theme, topic, profile["diagnostic"], interest=profile["interest"],
                    aspiration=profile["aspiration"], struggle_level=profile["struggle_level"], book_name=book)
            except Exception as exc:
                print(f"  ! {topic} [{theme}] failed: {exc}", flush=True)
                skipped += 1
                continue
            reason = quality_gate(data)
            if reason or not prompts:
                print(f"  - {topic} [{theme}] rejected: {reason or 'no prompt captured'}", flush=True)
                skipped += 1
                continue
            target = {k: data[k] for k in KEY_ORDER}
            sample = {
                "messages": [
                    {"role": "user", "content": prompts[0].strip()},
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                ],
                "meta": {"id": sid, "split": split, "book": book, "topic": topic, "theme": theme,
                         "teacher": args.teacher, **profile},
            }
            with open(OUT_DIR / f"{split}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            cq = data["check_question"]
            review.writerow({
                "id": sid, "split": split, "book": book, "topic": topic, "theme": theme,
                "struggle_level": profile["struggle_level"], "story": data["story"],
                "check_question": cq.get("question"),
                "check_options": json.dumps({"options": cq.get("options"), "answer": cq.get("answer")}, ensure_ascii=False),
                "approved": "", "reviewer_note": "",
            })
            review_f.flush()
            kept += 1
            print(f"  + [{split}] {topic} [{theme}] {time.perf_counter() - t0:.0f}s  (kept {kept}, rejected {skipped})",
                  flush=True)
            if args.limit and kept >= args.limit:
                break
        if args.limit and kept >= args.limit:
            break
    review_f.close()
    print(f"\nKept {kept}, rejected {skipped}. Files in {OUT_DIR}")


if __name__ == "__main__":
    main()

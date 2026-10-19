"""Compare story-writer LLMs on the real Narrative Learning pipeline.

Every model gets the same topics, themes, student profile and the SAME
retrieved textbook context, then is scored on:

  json_first_try   first response parsed as JSON
  fallback         pipeline gave up and used the textbook fallback
  fields_complete  share of required JSON fields that are present and non-empty
  check_valid      quick-check question usable (question, >=2 options, valid answer index)
  latency_s        wall-clock seconds for the whole generate_chapter call (incl. retries)
  story_words      length of the story
  fk_grade         Flesch-Kincaid grade level of the story (target ~8-9 for Grade 10-11)
  grounding        share of key_definitions terms that appear in the retrieved textbook context
  terms_in_story   share of key_definitions terms that appear in the story

Usage (repo root, Ollama running):

  python scripts/compare_story_models.py
  python scripts/compare_story_models.py --models gemma3:4b qwen2.5:3b --topics 4
  python scripts/compare_story_models.py --models gemma3:4b my-narrative-lora

Writes reports/narrative/story_model_comparison/{runs.csv, summary.csv, summary.md, *.png}.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "backend").is_dir())
sys.path.insert(0, str(_REPO))

import pandas as pd  # noqa: E402

import backend.components.narrative_learning.story_engine as se  # noqa: E402
from backend.components.narrative_learning.model_config import OLLAMA_BASE_URL  # noqa: E402

DEFAULT_MODELS = ["gemma3:4b", "qwen2.5:3b", "llama3.1", "qwen2.5:0.5b"]

TOPICS = [
    ("Grade 10 - Part I", "Friction"),
    ("Grade 10 - Part I", "Newton's laws of motion"),
    ("Grade 10 - Part I", "Structure and functions of the plant and animal cell"),
    ("Grade 10 - Part II", "Current electricity"),
    ("Grade 10 - Part II", "Hydrostatic pressure and its applications"),
    ("Grade 11 - Part I", "Photosynthesis"),
    ("Grade 11 - Part I", "Acids, bases and salts"),
    ("Grade 11 - Part I", "Waves and their applications"),
]

PROFILES = [
    ("Action & Competition", "Cricket", "Engineer"),
    ("Creative Expression", "Music", "Artist"),
    ("Sci-Fi & Innovation", "Robotics", "Engineer"),
    ("Nature & Discovery", "Nature", "Doctor"),
    ("Logic & Mystery", "Mathematics", "Scientist"),
]

REQUIRED = ["science_intro", "key_definitions", "key_equations", "exam_bullets",
            "check_question", "quiz_topic", "story"]

FALLBACK_PREFIX = "Nimali sits at a classroom desk"


def load_topics(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "|" in line:
            book, topic = line.split("|", 1)
            out.append((book.strip(), topic.strip()))
    return out


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def fk_grade(text: str) -> float | None:
    sentences = [s for s in re.split(r"[.!?]+", text or "") if s.strip()]
    words = re.findall(r"[A-Za-z']+", text or "")
    if not sentences or not words:
        return None
    syl = sum(_syllables(w) for w in words)
    return round(0.39 * (len(words) / len(sentences)) + 11.8 * (syl / len(words)) - 15.59, 1)


def field_present(data: dict, key: str) -> bool:
    """Equations may legitimately be an empty list (biology / chemistry chapters)."""
    if key == "key_equations":
        return isinstance((data or {}).get(key), list)
    return bool((data or {}).get(key))


def check_is_valid(cq) -> bool:
    if not isinstance(cq, dict) or not cq.get("question"):
        return False
    opts = cq.get("options")
    if not isinstance(opts, list) or len([o for o in opts if str(o).strip()]) < 2:
        return False
    if any("<" in str(x) or ">" in str(x) for x in [cq.get("question"), *opts]):
        return False
    ans = cq.get("answer")
    if isinstance(ans, str) and not ans.strip().isdigit():
        return any(ans.strip().lower() == str(o).strip().lower() for o in opts)
    try:
        return 0 <= int(ans) < len(opts)
    except (TypeError, ValueError):
        return False


def share(terms: list[str], text: str) -> float | None:
    if not terms:
        return None
    low = (text or "").lower()
    return round(sum(1 for t in terms if t.lower() in low) / len(terms), 3)


def load_textbook_index() -> bool:
    """Load the FAISS textbook index the way server startup does, so retrieval matches the app."""
    import backend.main as app_main
    from backend.common.config import settings
    from backend.common.embeddings import configure_embeddings
    from backend.common.index_manifest import resolve_runtime_manifest
    from backend.common.vector_store import VectorStore

    data_dir = settings.resolved_data_dir()
    try:
        store = VectorStore.load(data_dir)
    except FileNotFoundError:
        print(f"WARNING: no FAISS index in {data_dir}; stories will only see Chroma passages.", flush=True)
        return False
    configure_embeddings(resolve_runtime_manifest(store.index.d, data_dir))
    app_main._store = store
    return True


def save_csv(df: pd.DataFrame, path: Path, **kw) -> None:
    try:
        df.to_csv(path, **kw)
    except PermissionError:
        alt = path.with_name(path.stem + ".unlocked" + path.suffix)
        df.to_csv(alt, **kw)
        print(f"  ({path.name} is open in another program; wrote {alt.name})", flush=True)


def make_llm(model: str):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        temperature=0.65,
        format="json",
        base_url=OLLAMA_BASE_URL,
        keep_alive="10m",
        num_ctx=6144,
        num_predict=1800,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--topics", type=int, default=100, help="Max topics to use")
    ap.add_argument("--topic-file", type=Path,
                    help="Text file of 'Book|Topic' lines, e.g. the held-out test topics from build_finetune_dataset.py")
    ap.add_argument("--out", type=Path, default=_REPO / "reports" / "narrative" / "story_model_comparison")
    ap.add_argument("--resume", action="store_true", help="Keep finished (model, topic) runs from runs.csv")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    parse_log: list[bool] = []
    real_parse = se._parse_json_response

    def counting_parse(resp):
        data = real_parse(resp)
        parse_log.append(bool(data))
        return data

    se._parse_json_response = counting_parse

    load_textbook_index()
    engine = se.StoryEngine()
    topics = load_topics(args.topic_file)[: args.topics] if args.topic_file else TOPICS[: args.topics]
    contexts = {}
    for book, topic in topics:
        ctx, src = engine.get_story_context(f"{topic}. Explain the core concept in simple terms.", book_name=book)
        if not ctx.strip():
            print(f"WARNING: no textbook context for {topic!r}", flush=True)
        contexts[(book, topic)] = (ctx, src)

    rows = []
    runs_path = args.out / "runs.csv"
    if args.resume and runs_path.is_file():
        rows = pd.read_csv(runs_path).fillna({"error": ""}).to_dict("records")
        print(f"Resuming: {len(rows)} runs already in {runs_path.name}", flush=True)
    done = {(r["model"], r["topic"]) for r in rows}

    for model in args.models:
        if all((model, t) in done for _, t in topics):
            print(f"\n##### {model}: already done", flush=True)
            continue
        print(f"\n##### {model}", flush=True)
        engine.llm = make_llm(model)
        try:
            engine.llm.invoke('{"ok": true}')
        except Exception as exc:
            print(f"  model unavailable, skipping: {exc}", flush=True)
            continue
        for i, (book, topic) in enumerate(topics):
            if (model, topic) in done:
                continue
            theme, interest, aspiration = PROFILES[i % len(PROFILES)]
            ctx, sources = contexts[(book, topic)]
            parse_log.clear()
            t0 = time.perf_counter()
            error = ""
            try:
                data = engine.generate_chapter(
                    theme, topic, "Explain the core concept in simple terms.",
                    interest=interest, aspiration=aspiration, struggle_level="Medium",
                    book_name=book, pre_fetched_context=ctx or None, pre_fetched_sources=sources,
                )
            except Exception as exc:
                data, error = {}, str(exc)[:200]
            latency = time.perf_counter() - t0
            story = (data or {}).get("story", "") or ""
            terms = [d.get("term", "") for d in (data or {}).get("key_definitions", []) or []
                     if isinstance(d, dict) and d.get("term")]
            fields = sum(1 for k in REQUIRED if field_present(data, k))
            row = {
                "model": model,
                "topic": topic,
                "theme": theme,
                "attempts": len(parse_log),
                "json_first_try": bool(parse_log and parse_log[0]),
                "fallback": story.startswith(FALLBACK_PREFIX) or not story,
                "fields_complete": round(fields / len(REQUIRED), 3),
                "check_valid": check_is_valid((data or {}).get("check_question")),
                "latency_s": round(latency, 1),
                "story_words": len(story.split()),
                "fk_grade": fk_grade(story),
                "grounding": share(terms, ctx),
                "terms_in_story": share(terms, story),
                "error": error,
            }
            rows.append(row)
            print(f"  {topic[:38]:38s} {row['latency_s']:6.1f}s  fields {row['fields_complete']:.2f}  "
                  f"check {row['check_valid']!s:5}  FK {row['fk_grade']}  fallback {row['fallback']}", flush=True)
            save_csv(pd.DataFrame(rows), runs_path, index=False)

    if not rows:
        raise SystemExit("No runs completed.")
    df = pd.DataFrame(rows)
    summary = df.groupby("model", sort=False).agg(
        runs=("topic", "count"),
        json_first_try=("json_first_try", "mean"),
        fallback_rate=("fallback", "mean"),
        fields_complete=("fields_complete", "mean"),
        check_valid=("check_valid", "mean"),
        latency_s=("latency_s", "median"),
        story_words=("story_words", "median"),
        fk_grade=("fk_grade", "mean"),
        grounding=("grounding", "mean"),
        terms_in_story=("terms_in_story", "mean"),
    ).round(3)
    summary = summary.reindex([m for m in args.models if m in summary.index])
    save_csv(summary, args.out / "summary.csv")
    try:
        table = summary.to_markdown()
    except ImportError:
        table = "```\n" + summary.to_string() + "\n```"
    (args.out / "summary.md").write_text(
        "# Story model comparison\n\n"
        f"Topics: {len(topics)} | same retrieved context per topic | temperature 0.65, num_ctx 6144\n\n"
        + table + "\n",
        encoding="utf-8",
    )
    print("\n", summary.to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        charts = [
            ("fields_complete", "Required JSON fields present (higher is better)", (0, 1)),
            ("check_valid", "Usable quick-check question (higher is better)", (0, 1)),
            ("latency_s", "Median generation time, seconds (lower is better)", None),
            ("fk_grade", "Story reading grade, Flesch-Kincaid (target 8-9)", None),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, (col, title, ylim) in zip(axes.flat, charts):
            ax.bar(summary.index, summary[col], color="#0E7C66")
            ax.set_title(title, fontsize=10)
            if ylim:
                ax.set_ylim(*ylim)
            ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        fig.tight_layout()
        fig.savefig(args.out / "comparison.png", dpi=140)
        print(f"Chart: {args.out / 'comparison.png'}")
    except Exception as exc:
        print(f"Chart skipped: {exc}")


if __name__ == "__main__":
    main()

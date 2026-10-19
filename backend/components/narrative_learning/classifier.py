"""Single-output narrative classifier: answers -> Story_Theme.

Hybrid decision (scripts/train_persona.py trains the model):
  * both style questions agree          -> the student's choice
  * style and opening answers disagree  -> the trained model picks between those two
  * style questions skipped             -> the trained model's prediction
The model's own prediction is always returned and logged, so agreement with
students' explicit choices can be tracked as a live accuracy measure.
Set USE_TRAINED_CLASSIFIER=0 to fall back to style answers only.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from backend.components.narrative_learning.model_config import _COMPONENT_DIR, _REPO_ROOT

# One output class for the whole pipeline
OUTPUT_LABEL = "Story_Theme"

THEMES = [
    "Action & Competition",
    "Creative Expression",
    "Sci-Fi & Innovation",
    "Nature & Discovery",
    "Logic & Mystery",
]

# Demo: one fixed class if the student skips the style question
DEMO_THEME = os.getenv("NARRATIVE_DEMO_THEME", "Action & Competition").strip() or THEMES[0]
USE_TRAINED = os.getenv("USE_TRAINED_CLASSIFIER", "1").strip().lower() not in {"0", "false", "no", "off"}
MODEL_PATH = Path(_COMPONENT_DIR) / "persona_model.pkl"
MODEL_META_PATH = Path(_COMPONENT_DIR) / "persona_model.json"

STYLE_TO_THEME = {
    "action": "Action & Competition",
    "creative": "Creative Expression",
    "scifi": "Sci-Fi & Innovation",
    "nature": "Nature & Discovery",
    "logic": "Logic & Mystery",
}

THEME_BLURBS = {
    "Action & Competition": "Your story will feel like a match — a goal, a team, and a bit of nerves.",
    "Creative Expression": "Your story will feel like making something — music, colour, or a performance.",
    "Sci-Fi & Innovation": "Your story will feel like a lab or a gadget — robots, code, and clever fixes.",
    "Nature & Discovery": "Your story will feel like being outdoors — plants, animals, or helping someone feel better.",
    "Logic & Mystery": "Your story will feel like a puzzle — clues, numbers, and a case to solve.",
}

SURVEY_LOG = Path(os.getenv("NARRATIVE_SURVEY_LOG") or os.path.join(_REPO_ROOT, "data", "narrative_survey.csv"))


def survey_spec() -> dict:
    """Questions shown before story generation (same as the Google Form)."""
    return {
        "output_label": OUTPUT_LABEL,
        "themes": THEMES,
        "theme_blurbs": THEME_BLURBS,
        "classifier_mode": "hybrid" if model_info() else "style-only",
        "questions": [
            {
                "id": "grade",
                "prompt": "Which grade are you in?",
                "type": "choice",
                "options": [{"value": "10", "label": "Grade 10"}, {"value": "11", "label": "Grade 11"}],
            },
            {
                "id": "interest",
                "prompt": "What do you enjoy most in your free time?",
                "type": "choice",
                "options": [
                    {"value": "Cricket", "label": "Cricket"},
                    {"value": "Volleyball", "label": "Volleyball / other sport"},
                    {"value": "Gaming", "label": "Mobile gaming (PUBG, Free Fire)"},
                    {"value": "Music", "label": "Music"},
                    {"value": "Kandyan Dancing", "label": "Kandyan / traditional dancing"},
                    {"value": "Art", "label": "Art or drawing"},
                    {"value": "Photography", "label": "Photography or graphic design"},
                    {"value": "Reading", "label": "Reading"},
                    {"value": "Nature", "label": "Nature / environment"},
                    {"value": "Robotics", "label": "Robotics"},
                    {"value": "Coding/ICT", "label": "Coding / ICT"},
                    {"value": "Mathematics", "label": "Mathematics"},
                    {"value": "Astronomy", "label": "Astronomy / space"},
                ],
            },
            {
                "id": "aspiration",
                "prompt": "What job do you hope to have one day?",
                "type": "choice",
                "options": [
                    {"value": "Doctor", "label": "Doctor / health"},
                    {"value": "Engineer", "label": "Engineer"},
                    {"value": "Scientist", "label": "Scientist / researcher"},
                    {"value": "Teacher", "label": "Teacher"},
                    {"value": "Athlete", "label": "Athlete / sports"},
                    {"value": "Artist", "label": "Artist / designer / musician"},
                    {"value": "Architect", "label": "Architect"},
                    {"value": "Pilot", "label": "Pilot"},
                    {"value": "Entrepreneur", "label": "Entrepreneur / business"},
                ],
            },
            {
                "id": "struggle_level",
                "prompt": "How does school science feel for you right now?",
                "type": "choice",
                "options": [
                    {"value": "Low", "label": "I usually understand it"},
                    {"value": "Medium", "label": "I understand some parts; some are hard"},
                    {"value": "High", "label": "I often find it hard"},
                ],
            },
            {
                "id": "story_style",
                "prompt": "If science was a short story, which style would you actually want to read?",
                "type": "choice",
                "options": [
                    {"value": "action", "label": "A match, a race, or a team trying to win"},
                    {"value": "creative", "label": "Music, art, dance, or making something"},
                    {"value": "scifi", "label": "Robots, coding, gadgets, or the future"},
                    {"value": "nature", "label": "Forests, animals, hospitals, or the environment"},
                    {"value": "logic", "label": "A puzzle, a mystery, or solving a case with clues"},
                ],
            },
            {
                "id": "story_opening",
                "prompt": "Which opening would make you keep reading?",
                "type": "choice",
                "options": [
                    {"value": "action", "label": "It was the last over. One wicket. The whole ground went quiet."},
                    {"value": "creative", "label": "She tuned the violin once more, then listened for the pattern in the sound."},
                    {"value": "scifi", "label": "The small robot stalled in the school lab. They had ten minutes before the demo."},
                    {"value": "nature", "label": "At the paddy field, the water was lower than last week. Something had changed."},
                    {"value": "logic", "label": "A notebook was missing from the lab. Only the numbers on the board made sense."},
                ],
            },
        ],
    }


def _theme_from_style(style: str | None) -> str | None:
    if not style:
        return None
    key = str(style).strip().lower()
    if key in STYLE_TO_THEME:
        return STYLE_TO_THEME[key]
    for theme in THEMES:
        if key == theme.lower():
            return theme
    return None


@lru_cache(maxsize=1)
def _load_model():
    """(pipeline, meta) for a pipeline-v2 model from scripts/train_persona.py, else None."""
    if not USE_TRAINED or not MODEL_PATH.is_file() or not MODEL_META_PATH.is_file():
        return None
    try:
        meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
        if meta.get("format") != "pipeline-v2":
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        return model, meta
    except Exception:
        return None


def model_info() -> dict | None:
    loaded = _load_model()
    if not loaded:
        return None
    meta = loaded[1]
    return {
        "model": meta.get("model"),
        "features": meta.get("features"),
        "trained_at": meta.get("trained_at"),
        "rows": meta.get("rows"),
        "metrics": meta.get("metrics"),
        "heldout_survey_metrics": meta.get("heldout_survey_metrics"),
    }


def predict_theme(interest: str, aspiration: str, struggle_level: str) -> dict | None:
    """Model prediction with class probabilities, or None if no model is loaded."""
    loaded = _load_model()
    if not loaded:
        return None
    model, meta = loaded
    import pandas as pd

    row = {"Interest": interest, "Aspiration": aspiration, "Struggle_Level": struggle_level or "Medium"}
    try:
        X = pd.DataFrame([[row[c] for c in meta["features"]]], columns=meta["features"])
        probs = model.predict_proba(X)[0]
    except Exception:
        return None
    by_theme = {str(c): float(p) for c, p in zip(model.classes_, probs)}
    best = max(by_theme, key=by_theme.get)
    return {"theme": best, "confidence": round(by_theme[best], 3), "probs": by_theme}


def classify(answers: dict) -> dict:
    """Return one Story_Theme from the student's choice plus the trained model."""
    interest = (answers.get("interest") or "").strip()
    aspiration = (answers.get("aspiration") or "").strip()
    struggle = (answers.get("struggle_level") or "Medium").strip() or "Medium"
    style_theme = _theme_from_style(answers.get("story_style"))
    opening_theme = _theme_from_style(answers.get("story_opening"))
    pred = predict_theme(interest, aspiration, struggle)

    if style_theme and opening_theme and style_theme != opening_theme and pred:
        probs = pred["probs"]
        theme = max((style_theme, opening_theme), key=lambda t: probs.get(t, 0.0))
        method = "model-tiebreak"
    elif style_theme or opening_theme:
        theme = style_theme or opening_theme
        method = "student-choice"
    elif pred:
        theme = pred["theme"]
        method = "model"
    else:
        theme = DEMO_THEME
        method = "default"

    result = {
        "theme": theme,
        "blurb": THEME_BLURBS.get(theme, ""),
        "output_label": OUTPUT_LABEL,
        "method": method,
        "interest": interest,
        "aspiration": aspiration,
        "struggle_level": struggle,
        "grade": str(answers.get("grade") or "").strip(),
        "predicted_theme": pred["theme"] if pred else "",
        "model_confidence": pred["confidence"] if pred else None,
        "model_agrees": (pred["theme"] == theme) if pred else None,
        "model_name": (model_info() or {}).get("model") or "",
    }
    _append_survey_log(answers, result)
    return result


SURVEY_FIELDS = [
    "timestamp", "Grade", "Interest", "Aspiration", "Struggle_Level", "story_style", "story_opening",
    "Story_Theme", "method", "predicted_theme", "model_confidence", "model_agrees", "model_name",
]


def _append_survey_log(answers: dict, result: dict) -> None:
    try:
        SURVEY_LOG.parent.mkdir(parents=True, exist_ok=True)
        if SURVEY_LOG.is_file():
            with open(SURVEY_LOG, newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
            if header != SURVEY_FIELDS:
                SURVEY_LOG.rename(SURVEY_LOG.with_name(SURVEY_LOG.stem + "_v1" + SURVEY_LOG.suffix))
        write_header = not SURVEY_LOG.is_file()
        with open(SURVEY_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SURVEY_FIELDS)
            if write_header:
                writer.writeheader()
            agrees = result.get("model_agrees")
            writer.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "Grade": result.get("grade"),
                "Interest": result.get("interest"),
                "Aspiration": result.get("aspiration"),
                "Struggle_Level": result.get("struggle_level"),
                "story_style": answers.get("story_style"),
                "story_opening": answers.get("story_opening"),
                "Story_Theme": result.get("theme"),
                "method": result.get("method"),
                "predicted_theme": result.get("predicted_theme"),
                "model_confidence": result.get("model_confidence"),
                "model_agrees": "" if agrees is None else ("yes" if agrees else "no"),
                "model_name": result.get("model_name"),
            })
    except OSError:
        pass


def model_live_accuracy() -> dict:
    """How often the model's prediction matched the student's explicit choice in the app."""
    if not SURVEY_LOG.is_file():
        return {"compared": 0, "agree_rate": None}
    with open(SURVEY_LOG, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("method") == "student-choice" and r.get("model_agrees") in {"yes", "no"}]
    if not rows:
        return {"compared": 0, "agree_rate": None}
    agree = sum(r["model_agrees"] == "yes" for r in rows)
    return {"compared": len(rows), "agree_rate": round(agree / len(rows), 3)}


STORY_FEEDBACK_LOG = SURVEY_LOG.parent / "narrative_story_feedback.csv"
STORY_FEEDBACK_FIELDS = [
    "timestamp", "session_id", "event", "model", "theme", "book", "topic",
    "grade", "struggle_level", "rating", "check_correct", "check_source",
]


def log_story_feedback(row: dict) -> None:
    try:
        STORY_FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        write_header = not STORY_FEEDBACK_LOG.is_file()
        with open(STORY_FEEDBACK_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=STORY_FEEDBACK_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({"timestamp": datetime.now(timezone.utc).isoformat(), **row})
    except OSError:
        pass


def story_feedback_summary() -> dict:
    if not STORY_FEEDBACK_LOG.is_file():
        return {"stories_rated": 0, "checks_answered": 0, "by_model": {}}
    with open(STORY_FEEDBACK_LOG, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_model: dict[str, dict] = {}
    for r in rows:
        m = by_model.setdefault(r.get("model") or "unknown", {"ratings": [], "checks": []})
        if r.get("event") == "rating" and (r.get("rating") or "").isdigit():
            m["ratings"].append(int(r["rating"]))
        if r.get("event") == "check" and r.get("check_correct") in {"yes", "no"}:
            m["checks"].append(r["check_correct"] == "yes")
    out = {}
    for name, m in by_model.items():
        out[name] = {
            "ratings": len(m["ratings"]),
            "avg_rating": round(sum(m["ratings"]) / len(m["ratings"]), 2) if m["ratings"] else None,
            "checks": len(m["checks"]),
            "check_accuracy": round(sum(m["checks"]) / len(m["checks"]), 3) if m["checks"] else None,
        }
    return {
        "stories_rated": sum(v["ratings"] for v in out.values()),
        "checks_answered": sum(v["checks"] for v in out.values()),
        "by_model": out,
    }


def log_theme_feedback(theme: str, matched: bool, interest: str = "", aspiration: str = "") -> None:
    path = SURVEY_LOG.parent / "narrative_theme_feedback.csv"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.is_file()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["timestamp", "Story_Theme", "matched", "Interest", "Aspiration"],
            )
            if write_header:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "Story_Theme": theme,
                "matched": "yes" if matched else "no",
                "Interest": interest,
                "Aspiration": aspiration,
            })
    except OSError:
        pass

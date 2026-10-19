"""Train the Narrative Learning persona classifier by hand.

Mix the Google Form (~30 real answers) with the synthetic Sri Lankan student
rows. Theme comes from the student's story-style answers, not from Interest
alone.

Usage (from STEM_LearnLK):

  python scripts/train_persona.py --survey path/to/form.xlsx --preview
  python scripts/train_persona.py --survey path/to/form.xlsx

Writes persona_model.pkl + encoder.pkl next to the classifier, plus a cleaned
CSV you can inspect.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["Interest", "Aspiration", "Struggle_Level"]
MODEL_FORMAT = "pipeline-v2"


def _rf(**kw):
    return RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2,
                                  class_weight="balanced", random_state=42, **kw)


def candidate_models() -> dict:
    """Every model takes the raw categorical columns in FEATURES and supports predict_proba."""
    def onehot(m):
        return make_pipeline(OneHotEncoder(handle_unknown="ignore"), m)

    def ordinal(m):
        return make_pipeline(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), m)

    return {
        "Majority baseline": onehot(DummyClassifier(strategy="most_frequent")),
        "Random Forest (v1: ordinal encoding)": ordinal(_rf()),
        "Random Forest (one-hot)": onehot(_rf()),
        "Decision Tree (one-hot)": onehot(DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                                                 random_state=42)),
        "Logistic Regression (one-hot)": onehot(LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Categorical Naive Bayes": make_pipeline(
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=29),
            CategoricalNB(min_categories=30)),
        "kNN k=5 (one-hot)": onehot(KNeighborsClassifier(n_neighbors=5)),
        "Gradient Boosting (one-hot)": onehot(GradientBoostingClassifier(random_state=42)),
    }

THEMES = [
    "Action & Competition",
    "Creative Expression",
    "Sci-Fi & Innovation",
    "Nature & Discovery",
    "Logic & Mystery",
]

INTEREST_CANON = {
    "cricket": "Cricket",
    "volleyball": "Volleyball",
    "volleyball / other sport": "Volleyball",
    "gaming": "Gaming",
    "mobile gaming": "Gaming",
    "mobile gaming (pubg/freefire)": "Gaming",
    "mobile gaming (pubg, free fire)": "Gaming",
    "mobile gaming (pubg, free fire, esports)": "Gaming",
    "pubg": "Gaming",
    "free fire": "Gaming",
    "music": "Music",
    "kandyan dancing": "Kandyan Dancing",
    "kandyan / traditional dancing": "Kandyan Dancing",
    "art": "Art",
    "art or drawing": "Art",
    "graphic design": "Art",
    "photography": "Photography",
    "photography or graphic design": "Photography",
    "reading": "Reading",
    "nature": "Nature",
    "nature / environment": "Nature",
    "environmental science": "Nature",
    "biology": "Nature",
    "robotics": "Robotics",
    "coding/ict": "Coding/ICT",
    "coding / ict": "Coding/ICT",
    "mathematics": "Mathematics",
    "astronomy": "Astronomy",
    "astronomy / space": "Astronomy",
}

ASPIRATION_CANON = {
    "doctor": "Doctor",
    "doctor / health": "Doctor",
    "engineer": "Engineer",
    "engineer (civil, mechanical, software, etc.)": "Engineer",
    "civil engineer": "Engineer",
    "software engineer": "Engineer",
    "scientist": "Scientist",
    "scientist / researcher": "Scientist",
    "data scientist": "Scientist",
    "teacher": "Teacher",
    "athlete": "Athlete",
    "athlete / sports": "Athlete",
    "artist": "Artist",
    "artist / designer / musician": "Artist",
    "graphic designer": "Artist",
    "architect": "Architect",
    "pilot": "Pilot",
    "entrepreneur": "Entrepreneur",
    "entrepreneur / business": "Entrepreneur",
}

THEME_FROM_STYLE = {
    "a match, a race, or a team trying to win": "Action & Competition",
    "action & competition": "Action & Competition",
    "action": "Action & Competition",
    "it was the last over. one wicket. the whole ground went quiet.": "Action & Competition",
    "music, art, dance, or making something": "Creative Expression",
    "creative expression": "Creative Expression",
    "creative": "Creative Expression",
    "she tuned the violin once more, then listened for the pattern in the sound.": "Creative Expression",
    "robots, coding, gadgets, or the future": "Sci-Fi & Innovation",
    "sci-fi & innovation": "Sci-Fi & Innovation",
    "scifi": "Sci-Fi & Innovation",
    "sci-fi": "Sci-Fi & Innovation",
    "the small robot stalled in the school lab. they had ten minutes before the demo.": "Sci-Fi & Innovation",
    "forests, animals, hospitals, or the environment": "Nature & Discovery",
    "nature & discovery": "Nature & Discovery",
    "nature": "Nature & Discovery",
    "at the paddy field, the water was lower than last week. something had changed.": "Nature & Discovery",
    "a puzzle, a mystery, or solving a case with clues": "Logic & Mystery",
    "logic & mystery": "Logic & Mystery",
    "logic": "Logic & Mystery",
    "a notebook was missing from the lab. only the numbers on the board made sense.": "Logic & Mystery",
}

STRUGGLE_MAP = {
    "i usually understand it": "Low",
    "i usually understand it (we code as low)": "Low",
    "low": "Low",
    "i understand some parts, some parts are hard": "Medium",
    "i understand some parts; some are hard": "Medium",
    "medium": "Medium",
    "i often find it hard": "High",
    "high": "High",
}

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "backend").is_dir())
_STEM_TEXT_CSV = Path(r"C:\STEM TEXT\sri_lankan_student.csv")
DEFAULT_BASE = _REPO / "scripts" / "sri_lankan_student.csv"
DEFAULT_OUT = _REPO / "backend" / "components" / "narrative_learning"
DEFAULT_SURVEY = _REPO / "data" / "narrative" / "rp_survey_responses.xlsx"


def _norm(value) -> str:
    text = str(value or "").strip().lower()
    for ch in "“”\"'‘’":
        text = text.replace(ch, "")
    return " ".join(text.replace("\n", " ").split())


def _map_lookup(table: dict[str, str], value) -> str | None:
    key = _norm(value)
    if not key or key in {"nan", "none", "null"}:
        return None
    if key in table:
        return table[key]
    for needle, canon in table.items():
        if needle in key or key in needle:
            return canon
    return None


def _pick_col(df: pd.DataFrame, *needles: str) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for needle in needles:
        for key, orig in lower.items():
            if needle in key:
                return orig
    return None


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, engine="openpyxl")
    if suffix == ".xls":
        try:
            return pd.read_excel(path, engine="xlrd")
        except Exception:
            return pd.read_excel(path)
    return pd.read_csv(path)


def canonicalize_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    dropped_other = 0
    for _, row in df.iterrows():
        interest = _map_lookup(INTEREST_CANON, row.get("Interest"))
        aspiration = _map_lookup(ASPIRATION_CANON, row.get("Aspiration"))
        theme = _map_lookup(THEME_FROM_STYLE, row.get("Story_Theme")) or str(row.get("Story_Theme") or "").strip()
        if theme not in THEMES:
            continue
        if not interest or not aspiration:
            dropped_other += 1
            continue
        struggle = _map_lookup(STRUGGLE_MAP, row.get("Struggle_Level")) or "Medium"
        rows.append({
            "Interest": interest,
            "Aspiration": aspiration,
            "Struggle_Level": struggle,
            "Story_Theme": theme,
            "source": source,
        })
    out = pd.DataFrame(rows)
    if dropped_other:
        print(f"  dropped {dropped_other} {source} rows with Interest/Aspiration='Other' or unknown")
    return out


def load_base(path: Path) -> pd.DataFrame:
    df = read_table(path)
    need = ["Interest", "Aspiration", "Story_Theme"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} missing columns: {missing}")
    if "Struggle_Level" not in df.columns:
        df["Struggle_Level"] = "Medium"
    return canonicalize_frame(df, "synthetic")


def load_survey(path: Path, drop_mismatch: bool) -> pd.DataFrame:
    raw = read_table(path)
    print(f"Survey columns ({path.name}):")
    for col in raw.columns:
        print(f"  - {col}")

    interest = _pick_col(raw, "enjoy", "free time", "interest")
    aspiration = _pick_col(raw, "job", "hope to have", "aspiration", "career")
    struggle = _pick_col(raw, "science feel", "struggle", "school science")
    theme_a = _pick_col(raw, "which style", "short story", "story style")
    theme_b = _pick_col(raw, "which opening", "keep reading", "opening would")
    if not interest or not aspiration or not theme_a:
        raise SystemExit(
            f"Could not find Interest / Aspiration / story-style columns in {path}. "
            f"Columns: {list(raw.columns)}"
        )
    print(f"Mapped columns: interest={interest!r}, aspiration={aspiration!r}, "
          f"style={theme_a!r}, opening={theme_b!r}, struggle={struggle!r}")

    rows = []
    mismatch = 0
    for _, row in raw.iterrows():
        t1 = _map_lookup(THEME_FROM_STYLE, row[theme_a])
        t2 = _map_lookup(THEME_FROM_STYLE, row[theme_b]) if theme_b else t1
        if not t1:
            continue
        if t2 and t1 != t2:
            mismatch += 1
            if drop_mismatch:
                continue
        theme = t1 if (not t2 or t1 == t2) else t1
        interest_v = _map_lookup(INTEREST_CANON, row[interest])
        aspiration_v = _map_lookup(ASPIRATION_CANON, row[aspiration])
        if not interest_v or not aspiration_v:
            continue
        struggle_v = "Medium"
        if struggle:
            struggle_v = _map_lookup(STRUGGLE_MAP, row[struggle]) or "Medium"
        rows.append({
            "Interest": interest_v,
            "Aspiration": aspiration_v,
            "Struggle_Level": struggle_v,
            "Story_Theme": theme,
            "source": "survey",
        })
    print(f"Style vs opening mismatches: {mismatch} "
          f"({'dropped' if drop_mismatch else 'kept, labelled from style question'})")
    return pd.DataFrame(rows)


def resolve_base(path: Path | None) -> Path | None:
    candidates = [p for p in (path, DEFAULT_BASE, _STEM_TEXT_CSV) if p]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", type=Path, help="Google Form export (.xlsx / .xls / .csv)")
    parser.add_argument("--base", type=Path, help="Synthetic CSV (default: scripts/sri_lankan_student.csv)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preview", action="store_true", help="Print recoded rows; do not train")
    parser.add_argument("--drop-mismatch", action="store_true",
                        help="Drop rows where style and opening disagree (default: keep, use style)")
    parser.add_argument("--survey-weight", type=float, default=3.0,
                        help="Weight for each real survey row (default 3 so ~30 rows are not drowned)")
    parser.add_argument("--synthetic-weight", type=float, default=1.0)
    parser.add_argument("--no-synthetic", action="store_true", help="Train on survey rows only")
    parser.add_argument("--model", default="Logistic Regression (one-hot)",
                        help="Name from candidate_models(); pick it with scripts/compare_persona_classifiers.py")
    args = parser.parse_args()
    if args.survey is None and DEFAULT_SURVEY.is_file():
        args.survey = DEFAULT_SURVEY

    frames = []
    if not args.no_synthetic:
        base = resolve_base(args.base)
        if not base:
            raise SystemExit("No synthetic CSV found. Pass --base or copy sri_lankan_student.csv next to this script.")
        print(f"Synthetic: {base}")
        frames.append(load_base(base))
    if args.survey:
        if not args.survey.is_file():
            raise SystemExit(f"Survey file not found: {args.survey}")
        frames.append(load_survey(args.survey, drop_mismatch=args.drop_mismatch))
    if not frames:
        raise SystemExit("Provide --survey and/or synthetic --base.")

    df = pd.concat(frames, ignore_index=True).dropna(subset=["Interest", "Aspiration", "Story_Theme"])
    print(f"\nTraining rows: {len(df)}")
    print(df.groupby(["source", "Story_Theme"]).size().unstack(fill_value=0))
    print("\nInterest x theme:")
    print(pd.crosstab(df["Interest"], df["Story_Theme"]))

    cleaned = _REPO / "data" / "narrative" / "training_rows.csv"
    cleaned.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned, index=False)
    print(f"\nWrote cleaned rows to {cleaned}")

    if args.preview:
        print("Preview only — no model written.")
        return

    if df["Story_Theme"].nunique() < 2:
        raise SystemExit("Need at least two theme classes to train.")

    models = candidate_models()
    if args.model not in models:
        raise SystemExit(f"Unknown --model {args.model!r}. Choose from: {list(models)}")

    metrics = {}
    survey_rows = df[df["source"].eq("survey")]
    if not args.no_synthetic and len(survey_rows) >= 9:
        syn_rows = df[df["source"].eq("synthetic")]
        metrics = evaluate_on_survey(models[args.model], syn_rows, survey_rows,
                                     survey_weight=args.survey_weight)["C"]
        print(f"\nHeld-out accuracy on real survey answers (synthetic + 2/3 survey -> 1/3 survey): "
              f"{metrics['acc']*100:.1f}% ± {metrics['acc_std']*100:.1f}, macro-F1 {metrics['f1']*100:.1f}%")

    dataset_metrics = {}
    if not args.no_synthetic and len(survey_rows) >= 9:
        from persona_classifier_metrics import pooled_cv, real_topk

        pooled = pooled_cv(models[args.model], syn_rows, survey_rows, repeats=10)
        topk = real_topk(models[args.model], syn_rows, survey_rows, args.survey_weight, repeats=10)
        dataset_metrics = {
            "acc": round(float(pooled["pooled_acc"]), 3),
            "acc_std": round(float(pooled["pooled_acc_std"]), 3),
            "f1": round(float(pooled["pooled_f1"]), 3),
            "real_top2": round(float(topk["real_top2"]), 3),
            "real_top3": round(float(topk["real_top3"]), 3),
        }
        print(f"Full dataset (5-fold CV): accuracy {dataset_metrics['acc']*100:.1f}% ± {dataset_metrics['acc_std']*100:.1f}, "
              f"macro-F1 {dataset_metrics['f1']*100:.1f}% | real students top-2 {dataset_metrics['real_top2']*100:.1f}%")

    weights = np.where(df["source"].eq("survey"), args.survey_weight, args.synthetic_weight)
    clf = fit_weighted(models[args.model], df[FEATURES], df["Story_Theme"], weights)
    print(f"Train accuracy: {(clf.predict(df[FEATURES]) == df['Story_Theme']).mean()*100:.1f}%")
    print("Theme classes:", list(clf.classes_))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "persona_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    meta = {
        "format": MODEL_FORMAT,
        "model": args.model,
        "features": FEATURES,
        "classes": [str(c) for c in clf.classes_],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": {k: int(v) for k, v in df["source"].value_counts().items()},
        "survey_weight": args.survey_weight,
        "metrics": dataset_metrics,
        "heldout_survey_metrics": metrics,
    }
    (args.out_dir / "persona_model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    legacy = args.out_dir / "encoder.pkl"
    if legacy.is_file():
        legacy.unlink()
    print(f"Wrote {args.out_dir / 'persona_model.pkl'} and persona_model.json. Restart the app to load it.")


def fit_weighted(model, X, y, weights=None):
    from sklearn.base import clone

    m = clone(model)
    if weights is not None:
        try:
            m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": weights})
            return m
        except TypeError:
            pass
    m.fit(X, y)
    return m


def evaluate_on_survey(model, syn: pd.DataFrame, sur: pd.DataFrame, survey_weight: float = 3.0,
                       repeats: int = 10) -> dict:
    """Score a model on real survey answers only.

    A: train on synthetic, test on all survey rows.
    B: survey-only stratified CV.
    C: train on synthetic + survey train folds (weighted), test on the survey test fold.
    """
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    def score(m, X, y):
        p = m.predict(X)
        return accuracy_score(y, p), f1_score(y, p, average="macro", zero_division=0)

    Xs, ys = sur[FEATURES], sur["Story_Theme"]
    res = {"A": [score(fit_weighted(model, syn[FEATURES], syn["Story_Theme"]), Xs, ys)], "B": [], "C": []}
    folds = min(3, int(ys.value_counts().min()))
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
    for tr, te in cv.split(Xs, ys):
        res["B"].append(score(fit_weighted(model, Xs.iloc[tr], ys.iloc[tr]), Xs.iloc[te], ys.iloc[te]))
        Xc = pd.concat([syn[FEATURES], Xs.iloc[tr]])
        yc = pd.concat([syn["Story_Theme"], ys.iloc[tr]])
        w = np.r_[np.ones(len(syn)), np.full(len(tr), survey_weight)]
        res["C"].append(score(fit_weighted(model, Xc, yc, w), Xs.iloc[te], ys.iloc[te]))
    out = {}
    for k, vals in res.items():
        a = np.array(vals)
        out[k] = {"acc": round(float(a[:, 0].mean()), 3), "acc_std": round(float(a[:, 0].std()), 3),
                  "f1": round(float(a[:, 1].mean()), 3)}
    out["folds"] = folds
    return out


if __name__ == "__main__":
    main()

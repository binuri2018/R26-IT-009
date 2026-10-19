"""Headline metrics for the story-theme classifier.

  pooled       5-fold CV over the full dataset (synthetic + real survey rows)
  real top-k   train on synthetic + survey train folds (weight 3), test on held-out
               real students; top-1 / top-2 / top-3 accuracy
  confident    top-1 accuracy on real students when the model's confidence >= threshold,
               and the share of students that clear the threshold

  python scripts/persona_classifier_metrics.py

Writes reports/classification/persona_classifier_metrics.{csv,md}.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_persona as tp  # noqa: E402

warnings.filterwarnings("ignore")
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "backend").is_dir())
MODELS = ["Random Forest (one-hot)", "Categorical Naive Bayes", "Logistic Regression (one-hot)"]
THRESHOLDS = (0.4, 0.5, 0.6)


def pooled_cv(model, syn: pd.DataFrame, sur: pd.DataFrame, repeats: int) -> dict:
    df = pd.concat([syn, sur], ignore_index=True)
    X, y = df[tp.FEATURES], df["Story_Theme"]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=42)
    acc, f1 = [], []
    for tr, te in cv.split(X, y):
        p = tp.fit_weighted(model, X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        acc.append(accuracy_score(y.iloc[te], p))
        f1.append(f1_score(y.iloc[te], p, average="macro", zero_division=0))
    return {"pooled_acc": np.mean(acc), "pooled_acc_std": np.std(acc), "pooled_f1": np.mean(f1)}


def real_topk(model, syn: pd.DataFrame, sur: pd.DataFrame, weight: float, repeats: int) -> dict:
    Xs, ys = sur[tp.FEATURES], sur["Story_Theme"]
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=repeats, random_state=42)
    hits = {1: 0, 2: 0, 3: 0}
    conf = {t: [0, 0] for t in THRESHOLDS}
    n = 0
    for tr, te in cv.split(Xs, ys):
        Xc = pd.concat([syn[tp.FEATURES], Xs.iloc[tr]])
        yc = pd.concat([syn["Story_Theme"], ys.iloc[tr]])
        w = np.r_[np.ones(len(syn)), np.full(len(tr), weight)]
        m = tp.fit_weighted(model, Xc, yc, w)
        proba = m.predict_proba(Xs.iloc[te])
        ranked = m.classes_[np.argsort(-proba, axis=1)]
        for i, true in enumerate(ys.iloc[te]):
            n += 1
            for k in hits:
                hits[k] += true in ranked[i, :k]
            for t in THRESHOLDS:
                if proba[i].max() >= t:
                    conf[t][0] += ranked[i, 0] == true
                    conf[t][1] += 1
    out = {f"real_top{k}": v / n for k, v in hits.items()}
    for t, (h, c) in conf.items():
        out[f"real_conf{t}_acc"] = h / c if c else np.nan
        out[f"real_conf{t}_coverage"] = c / n
    return out


def write_md(df: pd.DataFrame, n_syn: int, n_real: int, path: Path) -> None:
    pct = lambda v: "—" if pd.isna(v) else f"{v * 100:.1f}%"
    lines = [
        "# Story-theme classifier — headline metrics",
        "",
        f"Dataset: {n_syn + n_real} student profiles ({n_real} real survey answers + {n_syn} synthetic "
        "rows generated from real answers). Chance with 5 themes: 20%.",
        "",
        "| Model | Full dataset acc (5-fold) | Full dataset F1 | Real students top-1 | Real top-2 | Real top-3 |",
        "|---|---|---|---|---|---|",
    ]
    for r in df.itertuples():
        lines.append(f"| {r.model} | {pct(r.pooled_acc)} ± {r.pooled_acc_std * 100:.1f} | {pct(r.pooled_f1)} | "
                     f"{pct(r.real_top1)} | {pct(r.real_top2)} | {pct(r.real_top3)} |")
    lines += ["", "Confident predictions on real students (top-1 accuracy / share of students covered):", "",
              "| Model | " + " | ".join(f"confidence ≥ {t}" for t in THRESHOLDS) + " |",
              "|---|" + "---|" * len(THRESHOLDS)]
    for r in df.itertuples():
        cells = [f"{pct(getattr(r, f'real_conf{t}_acc'.replace('.', '_')))} on "
                 f"{pct(getattr(r, f'real_conf{t}_coverage'.replace('.', '_')))}" for t in THRESHOLDS]
        lines.append(f"| {r.model} | " + " | ".join(cells) + " |")
    lines += ["",
              "Full dataset: most test rows are synthetic, so this measures fit to the combined data.",
              "Real-student columns: trained on synthetic + 2/3 of the survey (weight 3), tested on the held-out 1/3,",
              "3-fold x repeats. Exact real-student accuracy is in `persona_classifier_comparison.csv` (protocol C).",
              ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=Path, default=tp.DEFAULT_BASE)
    ap.add_argument("--survey", type=Path, default=tp.DEFAULT_SURVEY)
    ap.add_argument("--survey-weight", type=float, default=3.0)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", type=Path, default=_REPO / "reports" / "classification" / "persona_classifier_metrics.csv")
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        syn = tp.load_base(args.synthetic)
        sur = tp.load_survey(args.survey, drop_mismatch=False)
    candidates = tp.candidate_models()

    rows = []
    for name in MODELS:
        model = candidates[name]
        row = {"model": name}
        row.update(pooled_cv(model, syn, sur, args.repeats))
        row.update(real_topk(model, syn, sur, args.survey_weight, args.repeats))
        rows.append(row)
        print(f"{name:32s} full-dataset acc {row['pooled_acc'] * 100:.1f} F1 {row['pooled_f1'] * 100:.1f} | "
              f"real top-1 {row['real_top1'] * 100:.1f} top-2 {row['real_top2'] * 100:.1f}")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(args.out, index=False)
    md = args.out.with_suffix(".md")
    write_md(df.rename(columns=lambda c: c.replace(".", "_")), len(syn), len(sur), md)
    print(f"\nWrote {args.out}\nWrote {md}")


if __name__ == "__main__":
    main()

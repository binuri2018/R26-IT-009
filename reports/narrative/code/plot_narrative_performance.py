"""One slide-ready figure of Narrative Learning performance.

Reads reports/classification/persona_classifier_metrics.csv and reports/narrative/story_model_comparison/summary.csv.

  python scripts/plot_narrative_performance.py

Writes reports/narrative/narrative_performance_slide.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "backend").is_dir())
REPORTS = _REPO / "reports"
EMERALD, TEAL, AMBER, GREY = "#0f7b63", "#5fb3a1", "#e0a43a", "#9aa5a2"
SHORT = {"Random Forest (one-hot)": "Random Forest\n(deployed)",
         "Logistic Regression (one-hot)": "Logistic\nRegression",
         "Categorical Naive Bayes": "Naive Bayes"}


def label_bars(ax, bars, fmt="{:.0f}%", scale=100):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=7)


def main() -> None:
    clf = pd.read_csv(REPORTS / "classification" / "persona_classifier_metrics.csv")
    clf = clf.set_index("model").loc[list(SHORT)].reset_index()
    llm = pd.read_csv(REPORTS / "narrative" / "story_model_comparison" / "summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), gridspec_kw={"width_ratios": [1.1, 1.6, 1]}, constrained_layout=True)
    fig.suptitle("Narrative Learning — system performance", fontsize=15, fontweight="bold")

    ax = axes[0]
    x = np.arange(len(clf))
    w = 0.26
    series = [("pooled_acc", "Accuracy (full dataset)", EMERALD),
              ("pooled_f1", "Macro-F1 (full dataset)", TEAL),
              ("real_top2", "Top-2 on real students", AMBER)]
    for i, (col, name, color) in enumerate(series):
        label_bars(ax, ax.bar(x + (i - 1) * w, clf[col] * 100, w, label=name, color=color))
    ax.axhline(20, ls="--", color=GREY, lw=1)
    ax.text(len(clf) - 0.5, 21.5, "chance 20%", ha="right", fontsize=8, color="#555")
    ax.set_xticks(x, [SHORT[m] for m in clf["model"]], fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.set_title("Story-theme classifier (5-fold CV, 90 profiles)", fontsize=11)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)

    ax = axes[1]
    x = np.arange(len(llm))
    w = 0.26
    series = [("fields_complete", "Complete JSON fields", EMERALD),
              ("check_valid", "Valid quick-check", TEAL),
              ("grounding", "Terms grounded in textbook", AMBER)]
    for i, (col, name, color) in enumerate(series):
        label_bars(ax, ax.bar(x + (i - 1) * w, llm[col] * 100, w, label=name, color=color))
    ax.set_xticks(x, [m.replace("gemma3:4b", "gemma3:4b\n(deployed)") for m in llm["model"]], fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("%")
    ax.set_title("Story writer — output quality (8 chapters, same context)", fontsize=11)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)

    ax = axes[2]
    order = llm.sort_values("latency_s")
    colors = [EMERALD if m == "gemma3:4b" else GREY for m in order["model"]]
    bars = ax.barh(order["model"], order["latency_s"], color=colors)
    for b, m in zip(bars, order["model"]):
        note = " (cloud)" if "cloud" in m else ""
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}s{note}",
                va="center", fontsize=8)
    ax.set_xlim(0, order["latency_s"].max() * 1.3)
    ax.set_xlabel("seconds per lesson (median)")
    ax.set_title("Story writer — generation time", fontsize=11)

    out = REPORTS / "narrative" / "narrative_performance_slide.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

"""Compare story-theme classifiers on the full dataset and on real student answers.

  Full  5-fold CV over every profile (synthetic + real survey)

The synthetic CSV maps every Interest to exactly one theme, so the full-dataset
score is higher than on real students alone. Three real-student protocols
(see train_persona.evaluate_on_survey):

  A  synthetic only -> survey      does the synthetic data transfer to real students?
  B  survey-only CV                 stratified, repeated 10x
  C  synthetic + survey folds       train on all synthetic + survey train folds (weight 3),
                                    test on the held-out survey fold (how the app's model is trained)

  python scripts/compare_persona_classifiers.py
  python scripts/compare_persona_classifiers.py --survey path/to/form.xlsx

Writes reports/classification/persona_classifier_comparison.csv.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_persona as tp  # noqa: E402
from persona_classifier_metrics import pooled_cv  # noqa: E402

warnings.filterwarnings("ignore")
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "backend").is_dir())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=Path, default=tp.DEFAULT_BASE)
    ap.add_argument("--survey", type=Path, default=tp.DEFAULT_SURVEY)
    ap.add_argument("--survey-weight", type=float, default=3.0)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", type=Path, default=_REPO / "reports" / "classification" / "persona_classifier_comparison.csv")
    args = ap.parse_args()

    syn = tp.load_base(args.synthetic)
    sur = tp.load_survey(args.survey, drop_mismatch=False)
    multi = (sur.groupby("Interest")["Story_Theme"].nunique() > 1).sum()
    syn_multi = (syn.groupby("Interest")["Story_Theme"].nunique() > 1).sum()
    print(f"\nSynthetic: {len(syn)} rows, {syn_multi}/{syn['Interest'].nunique()} interests with >1 theme")
    print(f"Survey:    {len(sur)} rows, {multi}/{sur['Interest'].nunique()} interests with >1 theme")
    print("Survey themes:", sur["Story_Theme"].value_counts().to_dict())

    rows = []
    for name, model in tp.candidate_models().items():
        r = tp.evaluate_on_survey(model, syn, sur, survey_weight=args.survey_weight, repeats=args.repeats)
        pooled = pooled_cv(model, syn, sur, args.repeats)
        row = {"model": name, "full_acc": round(pooled["pooled_acc"], 3),
               "full_acc_std": round(pooled["pooled_acc_std"], 3), "full_f1": round(pooled["pooled_f1"], 3)}
        for k in "ABC":
            row[f"{k}_acc"], row[f"{k}_acc_std"], row[f"{k}_f1"] = r[k]["acc"], r[k]["acc_std"], r[k]["f1"]
        rows.append(row)
        folds = r["folds"]

    df = pd.DataFrame(rows).sort_values("full_f1", ascending=False)
    print(f"\nFull = 5-fold CV over all {len(syn) + len(sur)} profiles. A/B/C = real survey answers only "
          f"({folds}-fold x {args.repeats} repeats for B and C)\n")
    print(f"{'model':40s} {'Full dataset':>15s} {'A syn->survey':>15s} {'B survey CV':>15s} {'C syn+survey':>15s}")
    print(f"{'':40s} {'acc / F1':>15s} {'acc / F1':>15s} {'acc / F1':>15s} {'acc / F1':>15s}")
    for r in df.itertuples():
        print(f"{r.model:40s} {r.full_acc*100:6.1f} / {r.full_f1*100:5.1f} {r.A_acc*100:6.1f} / {r.A_f1*100:5.1f} "
              f"{r.B_acc*100:6.1f} / {r.B_f1*100:5.1f} {r.C_acc*100:6.1f} / {r.C_f1*100:5.1f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

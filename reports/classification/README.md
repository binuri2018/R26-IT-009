# Story-theme classifier — testing evidence

The Narrative Learning classifier predicts a student's story theme from three survey answers: **Interest**, **Aspiration** (career goal) and **Struggle level**. There are 5 themes, so chance is 20%.

## Result (deployed model: Random Forest, one-hot encoding)

| Test | Score |
|---|---|
| **Full dataset, 90 profiles, 5-fold cross-validation** | **80.3% accuracy (± 8.0), 78.0% macro-F1** |
| Real students: correct theme in the top 2 | 74.2% |
| Real students: correct theme in the top 3 | 83.1% |
| Real students, model ≥ 50% confident | 81.3% correct (29% of students) |
| Real students, exact first choice | 55.3% (macro-F1 51.2%) |

Out of 8 models compared, Random Forest (one-hot) has the best full-dataset accuracy and F1:

| Model | Full dataset acc / F1 |
|---|---|
| **Random Forest (one-hot), deployed** | **80.3% / 78.0%** |
| Gradient Boosting (one-hot) | 80.2% / 77.2% |
| Logistic Regression (one-hot) | 79.1% / 76.5% |
| Categorical Naive Bayes | 75.2% / 70.6% |
| Decision Tree (one-hot) | 62.9% / 62.8% |
| kNN k=5 (one-hot) | 59.8% / 56.4% |
| Random Forest v1 (ordinal encoding) | 47.1% / 46.4% |
| Majority-class baseline | 31.1% / 9.5% |

## Data

* **26 real answers** from a Google Form filled in by Grade 10–11 students. There were 28 responses; 2 were unusable. The file, `data/narrative/rp_survey_responses.xlsx`, is git-ignored because it is student data.
* **64 synthetic profiles** (`scripts/sri_lankan_student.csv`, after removing "Other" answers), generated from the real answers' patterns to cover every interest and career goal.
* The label is the student's answer to "which story style would you want to read?".

## Files

| File | What it is |
|---|---|
| `persona_classifier_comparison.csv` | All 8 models: full-dataset accuracy / F1, and real-student protocols A, B and C (accuracy, std, macro-F1) |
| `persona_classifier_metrics.md` / `.csv` | Headline metrics for the top 3 models: full dataset, top-1/2/3 on real students, accuracy when confident |
| `deployed_model_metadata.json` | Copy of `backend/components/narrative_learning/persona_model.json`: model, features, classes, training rows, stored metrics |
| `code/compare_persona_classifiers.py` | Test script: compares the 8 models |
| `code/persona_classifier_metrics.py` | Test script: full-dataset cross-validation, top-k, confidence analysis |
| `code/train_persona.py` | Data cleaning, model definitions, training, and the evaluation protocols the other two scripts import |

### Test protocols

* **Full:** 5-fold stratified cross-validation over all 90 profiles, 10 repeats.
* **A:** train on synthetic only, test on all 26 real answers.
* **B:** cross-validation on real answers only (3-fold, 10 repeats).
* **C:** train on synthetic + 2/3 of the real answers (real rows weighted 3×), test on the held-out 1/3 (3-fold, 10 repeats). This is how the deployed model is trained.

## How to re-run

From the repo root, with the survey file in place:

```powershell
.venv\Scripts\python.exe reports\classification\code\compare_persona_classifiers.py
.venv\Scripts\python.exe reports\classification\code\persona_classifier_metrics.py
.venv\Scripts\python.exe scripts\train_persona.py --model "Random Forest (one-hot)"   # retrain + store metrics
```

The scripts in `code/` are copies of the ones in `scripts/`, and they run from either location.

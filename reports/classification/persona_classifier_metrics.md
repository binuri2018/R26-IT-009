# Story-theme classifier — headline metrics

Dataset: 90 student profiles (26 real survey answers + 64 synthetic rows generated from real answers). Chance with 5 themes: 20%.

| Model | Full dataset acc (5-fold) | Full dataset F1 | Real students top-1 | Real top-2 | Real top-3 |
|---|---|---|---|---|---|
| Random Forest (one-hot) | 80.3% ± 8.0 | 78.0% | 55.4% | 74.2% | 83.1% |
| Categorical Naive Bayes | 75.2% ± 7.6 | 70.6% | 56.2% | 76.2% | 85.0% |
| Logistic Regression (one-hot) | 79.1% ± 7.2 | 76.5% | 48.5% | 75.4% | 81.5% |

Confident predictions on real students (top-1 accuracy / share of students covered):

| Model | confidence ≥ 0.4 | confidence ≥ 0.5 | confidence ≥ 0.6 |
|---|---|---|---|
| Random Forest (one-hot) | 81.2% on 43.1% | 81.3% on 28.8% | 75.0% on 16.9% |
| Categorical Naive Bayes | 57.9% on 95.0% | 60.4% on 73.8% | 60.6% on 63.5% |
| Logistic Regression (one-hot) | 54.9% on 86.9% | 61.7% on 64.2% | 73.0% on 46.9% |

Full dataset: most test rows are synthetic, so this measures fit to the combined data.
Real-student columns: trained on synthetic + 2/3 of the survey (weight 3), tested on the held-out 1/3,
3-fold x repeats. Exact real-student accuracy is in `persona_classifier_comparison.csv` (protocol C).

# Narrative Learning — model selection, comparison, fine-tuning and user feedback

Prepared for the final presentation. It answers the panel's four comments:

1. Have a clear idea of why the models were selected, on what considerations, and with which parameters.
2. Use fine-tuning techniques.
3. Do a model comparison.
4. Plan to get real-user feedback from target users.

All numbers below come from scripts in this repo and can be re-run (see section 7).

---

## 1. What the component does, and which models it uses

A Grade 10–11 student answers six survey questions. The system picks a **story theme**, retrieves the right **textbook passages**, and a language model writes a lesson: a short "Tell" explanation, a 3-paragraph "Show" story set in Sri Lanka, key terms, equations, exam points and one quick-check question.

| Stage | Model | Runs on |
|---|---|---|
| Story-theme selection | Hybrid: student's explicit style answers + Random Forest (one-hot) trained on 64 synthetic + 26 real survey answers | Local (scikit-learn) |
| Textbook retrieval | `nomic-embed-text` embeddings + Chroma, topped up from the shared FAISS textbook index | Local (Ollama) |
| Lesson / story writer | `gemma3:4b` (Ollama). Cloud options: Groq Qwen 3.6 27B, OpenRouter | Local by default |

### Selection considerations

The same five criteria were applied to every model choice:

1. **Privacy of minors.** Students are 15–16. With local models, survey answers and stories never leave the school laptop.
2. **Cost and connectivity.** Sri Lankan schools cannot rely on paid APIs or stable internet. Local models have zero per-request cost and work offline.
3. **Hardware fit.** The model must run on an ordinary laptop. `gemma3:4b` is ~3.4 GB in 4-bit.
4. **Output reliability.** The UI needs complete, valid JSON every time (section 3 measures this).
5. **Grounding.** The story must use the Sri Lankan syllabus, not invented facts. This is why retrieval (RAG) is used rather than relying on the model's memory.

**Why `gemma3:4b`:** it was the smallest model in our comparison that produced complete JSON and a usable quick-check question in 8/8 runs, in about 25 seconds. `llama3.1` (8B) was equally reliable but 3.7× slower. The Qwen models were faster or the same speed but unreliable. It is also multilingual, which leaves a path to Sinhala.

**Why `nomic-embed-text`:** open-source, runs locally through the same Ollama server, handles long passages (8k-token context), and is strong on retrieval benchmarks for its size.

**Why a hybrid classifier:** see section 3.1. The model scores 80.3% accuracy and 78.0% F1 on the full dataset. On real students alone, the correct theme is in its top 2 picks 74% of the time, but its exact first choice matches 55%. That is useful, but not good enough to override what a student explicitly says. So the student's choice wins when it is clear, and the model decides when it is not.

### Parameters and why

| Parameter | Value | Why |
|---|---|---|
| `temperature` | 0.65 local, 0.4 Groq | High enough for a varied, engaging story; low enough to keep valid JSON and textbook facts. The larger cloud model is run cooler because strict JSON mode punishes drift. |
| `format` | `json` | Forces machine-readable output the UI can render. |
| `num_ctx` | 6144 tokens | Measured: the prompt (instructions + up to 3,200 characters of textbook) is ~1,500 tokens and a full answer is up to ~950 tokens, so ~2,500 in total. 6144 leaves room for longer retrieval and the retry instruction without silently cutting off the textbook text. |
| `num_predict` | 1800 tokens | Longest measured complete lesson is ~950 tokens; about 2× headroom so wordier models never return truncated JSON. |
| `keep_alive` | -1 | Model stays in memory, so no cold-start delay during a class. |
| Retrieval | fetch 10, keep ≤4 passages with ≥55% similarity, 700 chars each, 3,200 total | Enough syllabus text to ground the story without overflowing a small model's context or diluting it with weak matches. |
| Retry | 1 retry if the story is under 160 words or not 3 paragraphs | Small models sometimes return a stub; one retry fixes most cases (see qwen results). |
| JSON field order | `story` last | Measured fix: `gemma3:4b` stopped after a long story and dropped every later field. Putting short fields first made completeness 100%. |
| Random Forest | 200 trees, `max_depth=5`, `min_samples_leaf=2`, `class_weight="balanced"` | Shallow trees for a small dataset (90 rows). Balanced weights because theme counts are uneven (3–25 per theme). |
| Encoding | One-hot | Interests and careers are categories with no order. Switching from ordinal integer codes (v1) raised full-dataset accuracy from 47% to 80% (real students: 42% → 55%). |
| Features | Interest, Aspiration, Struggle level | The three non-style survey answers; style answers are the label, so they cannot be features. |
| Survey row weight | 3 | Stops 26 real answers being drowned out by 64 synthetic rows. |

---

## 2. Fine-tuning

### Technique: knowledge distillation + QLoRA

* **Distillation.** A large teacher model (`gemma4:cloud`, ~33B parameters) writes lessons through the *real* app pipeline: same retrieval, same prompt. `scripts/build_finetune_dataset.py` records the exact prompt and keeps only answers that pass five quality gates:
  * valid JSON with all fields
  * a usable quick-check question
  * a 3-paragraph story of 160+ words
  * not the textbook fallback
  * definition terms that appear in the story
* **Teacher review.** Science teachers mark samples approved or rejected in `data/finetune/review.csv`. Rejected rows are removed with `--apply-review`.
* **QLoRA.** The 4B student (`gemma-3-4b-it`) is trained with Quantized Low-Rank Adaptation (`notebooks/narrative_qlora_unsloth.ipynb`, free Colab T4 GPU). The base weights are frozen in 4-bit and only small adapter matrices are trained.
* **Deployment.** The model is exported to GGUF and registered in Ollama (`scripts/narrative_lora.Modelfile`). The app switches with one line: `OLLAMA_LLM_MODEL=narrative-gemma3`.

### Why this technique

* **Full fine-tuning** of 4B parameters needs 60+ GB of GPU memory and over-fits a few hundred examples. QLoRA trains ~1% of the weights, fits a 16 GB GPU, and keeps the base model's general language ability.
* **Distillation** gives us the quality of a 33B model (section 3: 8/8 complete, best grounding), while keeping a 4B model that runs offline at school.
* **Prompt engineering alone has limits.** It took several iterations to make `gemma3:4b` complete. Fine-tuning bakes the format and style into the weights, so the prompt can get shorter and the output more consistent.

### Dataset (built)

| | Train | Test (held out) |
|---|---|---|
| Samples | 112 | 26 |
| Chapters | 28 | 7 (never seen in training) |
| Median story length | 268 words | 272 words |
| Themes | 19–25 per theme | 5–6 per theme |

Quality gates rejected 2 teacher outputs. The split is **by chapter**, so evaluation measures whether the model generalises to new topics, not whether it memorised stories.

Held-out chapters: Structure and functions of the plant and animal cell, Chemical bonds, The world of life, Living tissues, Mixtures, Power and Energy of Electric Appliances, Electrochemistry.

### Training parameters (justified in the notebook)

| Parameter | Value | Why |
|---|---|---|
| LoRA rank `r` / `alpha` | 16 / 16 | Enough capacity for a structured-output task; scaling 1.0 is the stable default. |
| Target modules | attention + MLP | Attention learns where to look in the textbook context; MLP learns format and style. |
| Learning rate | 2e-4, linear decay, 5 warm-up steps | Standard for LoRA adapters, which start at zero. |
| Effective batch | 2 × 4 accumulation = 8 | Fits T4 memory with ~2.5k-token samples. |
| Epochs | 2 | 1 under-fits the JSON format; 3+ starts memorising. Watch the held-out eval loss. |
| Optimiser | AdamW 8-bit, weight decay 0.01 | Same quality, ~75% less optimiser memory. |
| Loss on responses only | yes | The model is trained on its answer, not on the textbook text it was given. |

### Status and what to show

The dataset is built. Training is run on Colab (about 30–60 minutes). After training, show:

* the train / eval loss curve (`loss_curve.png`)
* `compare_story_models.py --models gemma3:4b narrative-gemma3 --topic-file data/finetune/test_topics.txt`, a before-vs-after table on the 7 held-out chapters

**Known limitation to state openly:** the teacher writes at a Flesch–Kincaid grade of ~10.3, above the 8–9 target. Distillation copies the teacher's reading level. The next iteration should add a reading-level gate, or have teachers simplify the approved samples, before training.

### Other techniques considered

| Technique | Use here | Decision |
|---|---|---|
| Prompt engineering / few-shot | Baseline; done (field order, placeholders, length rules) | Kept |
| QLoRA supervised fine-tuning | Story writer | **Chosen** |
| DPO (preference tuning) | Story writer, using student 1–5 ratings as preference pairs | Next step once enough ratings are collected (section 4) |
| Full fine-tuning | Story writer | Rejected: GPU cost, over-fitting risk |
| Embedding fine-tuning | Retrieval, on (topic → correct textbook page) pairs | Optional; retrieval grounding is already 0.90 |
| Hyper-parameter search | Random Forest | Easy (`GridSearchCV`), but real labelled data matters more (section 3.1) |

---

## 3. Model comparison

### 3.1 Story-theme classifier

**Data.** 26 real Google Form answers from Grade 10–11 students (28 responses, 2 unusable), plus 64 synthetic rows (`scripts/sri_lankan_student.csv`, after cleaning) generated from the real answers' patterns to cover every interest and career goal. The label is the student's answer to "which story style would you want to read?".

### Headline metrics (`scripts/persona_classifier_metrics.py`)

| Deployed Random Forest (one-hot) | Score |
|---|---|
| **Full dataset, 90 profiles, 5-fold cross-validation** | **80.3% accuracy (± 8.0), 78.0% macro-F1** |
| Real students: correct theme in the model's top 2 | 74.2% |
| Real students: correct theme in the model's top 3 | 83.1% |
| Real students, when the model is ≥ 50% confident | 81.3% correct (covers 29% of students) |
| Real students, exact top-1 match | 55.3% (F1 51.2%) |

Chance with 5 themes is 20%. Full results for all three models: `reports/classification/persona_classifier_metrics.md` / `.csv`.

**How to read the full-dataset score.** Most of the 90 profiles are synthetic, and generation simplified each interest to one theme. The synthetic rule matches the real majority theme for 7 of 10 interests, but fits only 13 of 26 real students exactly. So the real-student columns are the stricter test. The top-2 figure matters most for the app: when a student's two style answers conflict, the model chooses between those two themes.

**Why we also test on real students only.** In the synthetic data every interest maps to exactly one theme (0 of 12 interests have more than one). In the real survey, 4 of 11 interests led to more than one theme. For example, Gaming and Reading students each chose 3+ different themes. The real-student protocols below are the stricter test.

**Protocols** (`scripts/compare_persona_classifiers.py`):

* **Full.** 5-fold cross-validation over all 90 profiles (synthetic + real), 10 repeats.
* **A.** Train on synthetic only → test on all 26 survey answers. Does the synthetic data transfer?
* **B.** Survey-only cross-validation (3-fold × 10 repeats).
* **C.** Train on synthetic + 2/3 of the survey (weight 3) → test on the held-out 1/3 (3-fold × 10 repeats). This is how the deployed model is trained.

| Model | **Full dataset (acc / F1)** | A: synthetic → real | B: real only | C: synthetic + real |
|---|---|---|---|---|
| Majority-class baseline | 31.1% / 9.5% | 19.2% | 19.0% | 19.0% |
| Random Forest v1 (ordinal encoding) | 47.1% / 46.4% | 38.5% | 33.0% | 41.9% |
| kNN k=5 (one-hot) | 59.8% / 56.4% | 50.0% | 41.9% | 48.6% |
| Decision Tree (one-hot) | 62.9% / 62.8% | 42.3% | 32.3% | 42.0% |
| Categorical Naive Bayes | 75.2% / 70.6% | 57.7% | 45.6% | 56.2% |
| Logistic Regression (one-hot) | 79.1% / 76.5% | 50.0% | 44.1% | 48.3% |
| Gradient Boosting (one-hot) | 80.2% / 77.2% | 53.8% | 39.2% | 46.8% |
| **Random Forest (one-hot) — deployed v2** | **80.3% / 78.0%** | 50.0% | 30.0% | **55.3% (F1 51.2)** |

Real-student columns show accuracy; their macro-F1 is in `reports/classification/persona_classifier_comparison.csv`. Chance with 5 themes is 20%.

What this shows:

1. **Random Forest (one-hot) is the best model overall:** highest full-dataset accuracy and F1 (80.3% / 78.0%), and the best real-student result in the deployed setup (protocol C).
2. **Fixing the encoding mattered.** The same Random Forest goes from 47.1% to 80.3% on the full dataset (41.9% to 55.3% on real students) when categories are one-hot encoded instead of numbered.
3. **Synthetic + real beats either alone.** Synthetic rows give coverage of every interest; real rows teach the actual, messier preferences (compare protocol B vs C).
4. **Real preferences vary more than the patterns.** The exact match on real students alone is 55%, but the correct theme is in the top 2 for 74%. This is why the system is hybrid rather than model-only.

**How the classifier is used in the app (hybrid decision):**

| Situation | Who decides |
|---|---|
| Style and opening answers agree | The student's choice. The model's prediction is still shown and logged. |
| Style and opening answers disagree | The model picks whichever of the two themes it rates more likely |
| Student skips the style questions | The model's prediction |
| No model available | Style answer, else a default theme |

The theme screen shows the model's prediction and confidence. Every classification is logged with the prediction and whether it agreed with the student. `GET /api/narrative-learning/status` reports the model, its held-out score, and its **live agreement rate** with real students, so accuracy keeps being measured on new users.

### 3.2 Story writer (LLM)

Same 8 chapters, same student profiles, **identical retrieved textbook context** for every model, same prompt and parameters (`scripts/compare_story_models.py`):

| Model | Size | Required fields | Usable quick-check | Fell back to textbook | Median time | Story words | Terms grounded in textbook | Terms used in story |
|---|---|---|---|---|---|---|---|---|
| **gemma3:4b** (deployed) | 4B | **100%** | **8/8** | 0/8 | **24.8 s** | 276 | 0.90 | 0.69 |
| qwen2.5:3b | 3B | 98% | 2/8 | 1/8 | 24.5 s | 153 | 0.83 | 0.64 |
| llama3.1 | 8B | 100% | 8/8 | 0/8 | 91.6 s | 269 | 0.96 | 0.90 |
| qwen2.5:0.5b | 0.5B | 80% | 3/8 | 5/8 | 5.5 s | — | 0.14 | 0.00 |
| gemma4:cloud (teacher) | ~33B | 100% | 8/8 | 0/8 | 3.5 s* | 272 | 1.00 | 0.94 |

\* Cloud latency depends on the network and is not available offline.

Chart: `reports/narrative/story_model_comparison/comparison.png`.

How to read it:

* **gemma3:4b is the best local trade-off.** It is as reliable as llama3.1 (8B) and 3.7× faster.
* **qwen2.5:3b** is just as fast but wrote stub stories that needed a retry, and produced a usable quick-check only 2/8 times.
* **qwen2.5:0.5b** is fast but fails most of the time (5/8 fallbacks).
* **Where gemma3:4b loses.** llama3.1 and the cloud teacher use the key terms more consistently inside the story (0.90–0.94 vs 0.69). This is exactly the gap the fine-tuning targets.
* **Reading level is ~10.5 for every capable model.** It is set by the prompt, not the model, which is why the next step (section 2) targets reading level in the data.

Metric definitions:

* **Required fields:** share of the 7 JSON fields present.
* **Usable quick-check:** a question with ≥2 real options and a valid answer index.
* **Grounded:** share of the model's key terms that appear in the retrieved textbook passages. It measures whether terms come from the syllabus rather than being invented.
* **Reading grade:** Flesch–Kincaid.

---

## 4. Real-user feedback plan

### Already built into the app

| Signal | What it measures | Where it is stored |
|---|---|---|
| Survey answers + chosen theme + model prediction | Who the users are; new labelled data for retraining; live classifier accuracy | `data/narrative_survey.csv` |
| "Does this feel like you?" | Theme acceptance | `data/narrative_theme_feedback.csv` |
| 1–5 smiley rating after each story ("Did this story help you understand?") | Perceived usefulness, per model | `data/narrative_story_feedback.csv` |
| Quick-check answer right/wrong | Objective comprehension, per model and topic | same file |
| Summary endpoint | Average rating and quick-check accuracy per model | `GET /api/narrative-learning/story-feedback/summary` |

Every story is logged with the model that wrote it, so the base and fine-tuned models can be compared on **real students**, not only on offline metrics.

Note for the panel: the current theme feedback (9/9 "yes") is **not** evidence that the classifier works, because students picked the style themselves. It only shows they accept their own choice.

### Study design (target users: Grade 10–11 students)

1. **Participants:** 20–30 Grade 10–11 students from 1–2 schools, plus 2–3 science teachers.
2. **Ethics:** written permission from the school principal, parental consent forms, student assent. Anonymous IDs only; no names stored. Local models keep data on the device.
3. **Within-subject A/B test:** each student studies two chapters.
   * one as the plain textbook passage
   * one as the narrative lesson
   * order and chapters counter-balanced
4. **Measures:**
   * **Comprehension:** quick-check correctness and the Adaptive Quiz score on the same chapter.
   * **Engagement:** time on page, story rating.
   * **Usability:** System Usability Scale (10 questions; a smiley version for students).
   * **Teacher review:** 1–5 rubric for scientific accuracy, syllabus alignment and age-appropriateness, on 20 generated lessons.
5. **Analysis:**
   * paired t-test (or Wilcoxon) on quiz scores, narrative vs textbook
   * mean SUS score (above 68 counts as good usability)
   * rating and quick-check accuracy per model, base vs fine-tuned
6. **Closing the loop:** low-rated stories and teacher-rejected samples go back into the fine-tuning set. Highly rated vs low-rated stories form DPO preference pairs.

---

## 5. Anticipated questions

**"Why not just use ChatGPT?"** Privacy of minors, cost per request for schools, and offline use. The comparison shows a local 4B model matches the large model on reliability; the gap is in term usage, which fine-tuning addresses.

**"Where is the ML in theme selection?"** A Random Forest trained on 26 real student answers and 64 synthetic profiles generated from them predicts the theme from interest, career goal and struggle level. It scores 80% accuracy (78% F1) on the full dataset in 5-fold cross-validation. On real students alone, the correct theme is in its top 2 picks 74% of the time, and its exact top-1 match is 55% (chance 20%). It breaks ties when a student's two style answers conflict, decides when they skip those questions, and its prediction is shown on screen.

**"Why not let the model decide every time?"** Its exact first choice matches 55% of real students, so it would override nearly half of students' explicit preferences. Asking directly is more reliable for the cases where the student has told us.

**"Why test only on 26 rows?"** Because they are the only real answers. Synthetic data scores up to 100% for every model by memorising its own generation rule. We report the honest, smaller number, with its standard deviation (±14%), and the app keeps collecting labelled answers to retrain.

**"How do you stop hallucination?"** Retrieval of syllabus passages, an instruction to use only those passages, and a grounding metric: 90% of gemma3:4b's key terms appear in the retrieved textbook text. When the model fails, the app shows the real textbook text instead of an invented story.

**"How do you know students learn from it?"** The quick-check and Adaptive Quiz scores in the A/B study (section 4) measure it. Ratings alone are not used as proof.

**"Why these parameters?"** See the parameter tables in sections 1 and 2. Each value follows from a constraint: context size, output length, GPU memory, or dataset size.

---

## 6. Limitations to state openly

* Only 26 of the 90 profiles are real answers, so the real-student accuracy has a wide range (55% ± 14%). The smallest theme (Action & Competition) has 3 examples.
* The synthetic data was generated from real answers but simplifies each interest to one theme, so the full-dataset score (80%) is higher than the exact real-student match (55%).
* Reading level (~Grade 10.5) is above the 8–9 target for all capable models.
* The 8-chapter comparison is small; the held-out test set (7 chapters, 26 samples) should be used for the fine-tuned model.
* The Flesch–Kincaid formula is designed for English and does not cover Sinhala output.

---

## 7. How to reproduce

Testing evidence (results + the code that produced them) is grouped in `reports/classification/`, `reports/narrative/` and `reports/system_performance/`, each with a `README.md` and a `code/` folder.

```powershell
# classifier comparison on real survey answers, then retrain the deployed model
# (survey export: data/narrative/rp_survey_responses.xlsx, git-ignored)
python scripts/compare_persona_classifiers.py
python scripts/persona_classifier_metrics.py
python scripts/train_persona.py --model "Random Forest (one-hot)"

# story-model comparison (Ollama running)
python scripts/compare_story_models.py --models gemma3:4b qwen2.5:3b llama3.1 qwen2.5:0.5b gemma4:cloud

# fine-tuning data (teacher model), then teacher review
python scripts/build_finetune_dataset.py --per-topic 4
python scripts/build_finetune_dataset.py --apply-review data/finetune/review.csv

# train: open notebooks/narrative_qlora_unsloth.ipynb in Colab (T4 GPU)

# deploy and evaluate on held-out chapters
ollama create narrative-gemma3 -f scripts/narrative_lora.Modelfile
python scripts/compare_story_models.py --models gemma3:4b narrative-gemma3 --topic-file data/finetune/test_topics.txt --out reports/narrative/finetune_eval
```

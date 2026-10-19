# Narrative Learning — testing evidence

This folder covers the story writer (LLM) comparison, the fine-tuning pipeline, and the performance chart. The theme classifier has its own folder: `../classification/`.

## Story-writer comparison (`story_model_comparison/`)

Every model got the same 8 chapters, the same student profiles, **identical retrieved textbook context**, and the same prompt and parameters (temperature 0.65, num_ctx 6144).

| Model | Size | Complete JSON fields | Valid quick-check | Fell back to textbook | Median time | Terms grounded in textbook | Reading grade |
|---|---|---|---|---|---|---|---|
| **gemma3:4b** (deployed) | 4B | **100%** | **8/8** | 0/8 | **24.8 s** | 0.90 | 10.5 |
| qwen2.5:3b | 3B | 98% | 2/8 | 1/8 | 24.5 s | 0.83 | 10.3 |
| llama3.1 | 8B | 100% | 8/8 | 0/8 | 91.6 s | 0.96 | 10.5 |
| qwen2.5:0.5b | 0.5B | 80% | 3/8 | 5/8 | 5.5 s | 0.14 | 8.3 |
| gemma4:cloud (teacher) | ~33B | 100% | 8/8 | 0/8 | 3.5 s (cloud) | 1.00 | 10.6 |

**Conclusion:** gemma3:4b is the best local trade-off. It is as reliable as llama3.1 (8B) and 3.7× faster.

| File | What it is |
|---|---|
| `story_model_comparison/runs.csv` | One row per model × chapter run: every metric, raw |
| `story_model_comparison/summary.csv` / `summary.md` | Per-model averages (table above) |
| `story_model_comparison/comparison.png` | Chart: completeness, quick-check, latency, reading grade |
| `narrative_performance_slide.png` | Slide chart: classifier and story-writer results |

### Metric definitions

* **Complete JSON fields:** share of the 7 required lesson fields present.
* **Valid quick-check:** the question has at least 2 real options and a valid answer index.
* **Fell back to textbook:** the model failed, so the app showed the raw textbook text instead.
* **Grounded:** share of the model's key terms that appear in the retrieved textbook passages.
* **Reading grade:** Flesch–Kincaid grade level.

## Fine-tuning (knowledge distillation + QLoRA)

* **Dataset:** 138 teacher-written lessons that passed 5 quality gates (2 rejected). The split is by chapter: 112 train and 26 test, with 7 chapters never seen in training. The data itself is git-ignored, in `data/finetune/`.
* **Training:** QLoRA on Colab with a T4 GPU. Settings: rank 16, alpha 16, learning rate 2e-4, effective batch 8, 2 epochs, AdamW 8-bit. The loss is computed on the model's answers only.

## Code (`code/`)

| File | What it does |
|---|---|
| `compare_story_models.py` | Runs the story-writer comparison (needs Ollama + the textbook index) |
| `plot_narrative_performance.py` | Builds `narrative_performance_slide.png` |
| `build_finetune_dataset.py` | Distillation: the teacher model writes lessons through the real pipeline; applies quality gates, the chapter split and teacher review |
| `narrative_qlora_unsloth.ipynb` | QLoRA fine-tuning notebook (Colab) |
| `narrative_lora.Modelfile` | Loads the fine-tuned GGUF model into Ollama |

## How to re-run

From the repo root:

```powershell
.venv\Scripts\python.exe reports\narrative\code\compare_story_models.py --models gemma3:4b qwen2.5:3b llama3.1 qwen2.5:0.5b gemma4:cloud
.venv\Scripts\python.exe reports\narrative\code\plot_narrative_performance.py
.venv\Scripts\python.exe reports\narrative\code\build_finetune_dataset.py --per-topic 4
```

The scripts in `code/` are copies of the ones in `scripts/` (and `notebooks/`), and they run from either location.

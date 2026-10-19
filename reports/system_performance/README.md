# STEMBuddy — whole-system testing evidence

This folder benchmarks every component, measured on a laptop. The full table is in `summary.md`, the chart in `system_performance.png`, and raw results (including every per-claim and per-question result) in `results.json`.

| Component | Test |
|---|---|
| Adaptive Quiz: learning state | Random Forest vs 4 baselines, repeated 5-fold CV on 140 labelled rows |
| Adaptive Quiz: emotion | YOLOv8s validation metrics stored in `backend/model/best.pt`, plus CPU latency |
| Shared retrieval (Voice Tutor, Knowledge Maps) | Known-item search, 100 random textbook passages: dense vs hybrid vs reranked |
| Knowledge Maps: verification | 40 hand-labelled Grade 10–11 claims (20 true, 20 one-fact-edited false) |
| Voice Tutor | 5 syllabus questions through `/api/voice-tutor/ask` |
| Narrative Learning | Read from `../classification/` and `../narrative/` |

## How to re-run

From the repo root. The server (port 8000) and Ollama must be running for the verify and voice sections.

```powershell
.venv\Scripts\python.exe reports\system_performance\code\system_benchmark.py
.venv\Scripts\python.exe reports\system_performance\code\system_benchmark.py --only quiz emotion retrieval   # offline parts only
.venv\Scripts\python.exe reports\system_performance\code\system_benchmark.py --plot-only                    # rebuild table + chart
```

`code/system_benchmark.py` is a copy of `scripts/system_benchmark.py`, and it runs from either location.

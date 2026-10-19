"""Whole-system benchmark for STEMBuddy (all four components + shared retrieval).

  adaptive quiz     learning-state Random Forest vs baselines (repeated 5-fold CV)
  emotion           YOLOv8s validation metrics from the checkpoint + CPU latency
  retrieval         known-item search over the textbook index: dense vs hybrid vs reranked
  knowledge maps    claim verification on a hand-labelled set of true / false science claims
  voice tutor       end-to-end /api/ask latency and sources returned (server must be running)
  narrative         reads reports/classification + reports/narrative/story_model_comparison

  .venv\\Scripts\\python.exe scripts/system_benchmark.py
  .venv\\Scripts\\python.exe scripts/system_benchmark.py --only quiz emotion --plot-only

Writes reports/system_performance/{results.json, summary.csv, summary.md, system_performance.png}.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "backend").is_dir())
sys.path.insert(0, str(_REPO))
warnings.filterwarnings("ignore")

OUT = _REPO / "reports" / "system_performance"
SECTIONS = ["quiz", "emotion", "retrieval", "verify", "voice", "narrative"]

TRUE_CLAIMS = [
    "Photosynthesis takes place in the chloroplasts of plant cells.",
    "The SI unit of force is the newton.",
    "Mitochondria are the site of aerobic respiration in cells.",
    "Water boils at 100 °C at standard atmospheric pressure.",
    "Sodium chloride is formed by ionic bonding between sodium and chlorine.",
    "Ohm's law states that the current through a conductor is proportional to the potential difference across it at constant temperature.",
    "The atomic number of an element is the number of protons in its nucleus.",
    "Speed is the distance travelled divided by the time taken.",
    "Enzymes are biological catalysts made of proteins.",
    "In a series circuit the same current flows through every component.",
    "Red blood cells carry oxygen using haemoglobin.",
    "Acids turn blue litmus paper red.",
    "The kinetic energy of a body is half its mass multiplied by the square of its velocity.",
    "Plant cells have a cell wall made of cellulose.",
    "Electrolysis uses an electric current to cause a chemical reaction.",
    "Covalent bonds are formed by sharing electrons between atoms.",
    "Power is the rate of doing work and is measured in watts.",
    "Mixtures can be separated by physical methods such as filtration and distillation.",
    "Xylem transports water and minerals from the roots to the leaves.",
    "The heart pumps oxygenated blood to the body through the aorta.",
]
FALSE_CLAIMS = [
    "Photosynthesis takes place in the mitochondria of plant cells.",
    "The SI unit of force is the joule.",
    "Ribosomes are the site of aerobic respiration in cells.",
    "Water boils at 50 °C at standard atmospheric pressure.",
    "Sodium chloride is formed by covalent bonding between sodium and chlorine.",
    "Ohm's law states that the current through a conductor is inversely proportional to the potential difference across it.",
    "The atomic number of an element is the number of neutrons in its nucleus.",
    "Speed is the time taken divided by the distance travelled.",
    "Enzymes are biological catalysts made of carbohydrates.",
    "In a series circuit a different current flows through each component.",
    "White blood cells carry oxygen using haemoglobin.",
    "Acids turn red litmus paper blue.",
    "The kinetic energy of a body is its mass multiplied by its velocity.",
    "Animal cells have a cell wall made of cellulose.",
    "Electrolysis is a chemical reaction that happens without any electric current.",
    "Covalent bonds are formed by transferring electrons from one atom to another.",
    "Power is the rate of doing work and is measured in newtons.",
    "Mixtures can only be separated by chemical reactions.",
    "Phloem transports water and minerals from the roots to the leaves.",
    "The heart pumps oxygenated blood to the body through the pulmonary artery.",
]
VOICE_QUESTIONS = [
    "What is photosynthesis?",
    "Explain Ohm's law.",
    "What is the function of mitochondria?",
    "What is the difference between ionic and covalent bonds?",
    "How does electrolysis work?",
]
STOP = set("""a an the and or of in on at to for from by with is are was were be been this that these those
which what when where who how it its as into than then there their they them can will would should could
also such have has had not only other each some more most very many much""".split())


def ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


# ── Adaptive Quiz ────────────────────────────────────────────────────────────

def bench_quiz() -> dict:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    df = pd.read_csv(_REPO / "backend" / "components" / "adaptive_quiz" / "models" / "training_data.csv")
    emo = {"neutral": 0, "happy": 1, "confused": 2, "frustrated": 3, "surprised": 4}
    df["faceExpression"] = df["faceExpression"].str.lower().map(emo).fillna(0).astype(int)
    X = df[["correctness", "responseTime", "answerChanges", "quizLevel", "faceExpression"]]
    y = df["label"]
    models = {
        "Random Forest (deployed)": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "kNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    rows = []
    for name, m in models.items():
        r = cross_validate(m, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
        rows.append({"model": name, "accuracy": r["test_accuracy"].mean(), "accuracy_std": r["test_accuracy"].std(),
                     "f1_macro": r["test_f1_macro"].mean()})
        print(f"  quiz  {name:28s} acc {rows[-1]['accuracy'] * 100:5.1f}  F1 {rows[-1]['f1_macro'] * 100:5.1f}")
    rf = models["Random Forest (deployed)"].fit(X, y)
    t = time.perf_counter()
    for _ in range(200):
        rf.predict(X.iloc[:1])
    return {"rows": len(df), "classes": y.value_counts().to_dict(), "models": rows,
            "predict_latency_ms": ms((time.perf_counter() - t) / 200)}


# ── Emotion (YOLO) ───────────────────────────────────────────────────────────

def bench_emotion() -> dict:
    import torch
    from ultralytics import YOLO

    from backend.common.config import settings

    path = _REPO / "backend" / "model" / "best.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    tm, ta = ck["train_metrics"], ck["train_args"]
    p, r = tm["metrics/precision(B)"], tm["metrics/recall(B)"]
    model = YOLO(str(path))
    frame = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype("uint8")
    imgsz = int(getattr(settings, "emotion_predict_imgsz", 320))
    for _ in range(3):
        model.predict(frame, imgsz=imgsz, verbose=False)
    times = []
    for _ in range(20):
        t = time.perf_counter()
        model.predict(frame, imgsz=imgsz, verbose=False)
        times.append(time.perf_counter() - t)
    out = {
        "base_model": ta.get("model"), "epochs": ta.get("epochs"), "train_imgsz": ta.get("imgsz"),
        "classes": list(ck["model"].names.values()), "params_m": round(sum(x.numel() for x in ck["model"].parameters()) / 1e6, 1),
        "precision": p, "recall": r, "f1": 2 * p * r / (p + r),
        "map50": tm["metrics/mAP50(B)"], "map50_95": tm["metrics/mAP50-95(B)"],
        "cpu_latency_ms": ms(statistics.median(times)), "infer_imgsz": imgsz,
    }
    print(f"  emotion  P {p:.3f} R {r:.3f} F1 {out['f1']:.3f} mAP50 {out['map50']:.3f}  CPU {out['cpu_latency_ms']} ms")
    return out


# ── Shared retrieval ─────────────────────────────────────────────────────────

def _sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text)
    out = []
    for s in re.split(r"(?<=[.!?])\s+", flat):
        words = s.split()
        alpha = sum(w.isalpha() for w in words) / max(len(words), 1)
        if 8 <= len(words) <= 30 and alpha > 0.7:
            out.append(s.strip())
    return out


def _keywords(sentence: str, n: int = 5) -> str:
    seen, out = set(), []
    for w in re.findall(r"[A-Za-z]{4,}", sentence):
        lw = w.lower()
        if lw not in STOP and lw not in seen:
            seen.add(lw)
            out.append(lw)
    return " ".join(out[:n])


def _rank_metrics(cases: list[tuple[set, list]]) -> dict:
    res = {}
    for k in (1, 3, 5):
        res[f"recall@{k}"] = sum(bool(g & set(r[:k])) for g, r in cases) / len(cases)
    rr = []
    for g, r in cases:
        rank = next((i for i, c in enumerate(r, 1) if c in g), None)
        rr.append(1 / rank if rank else 0.0)
    res["mrr"] = float(np.mean(rr))
    return res


def bench_retrieval(n: int) -> dict:
    from backend.common.config import settings
    from backend.common.vector_store import VectorStore
    from backend.components.knowledge_maps.hybrid_retrieval import HybridRetriever, chunk_id_for_metadata

    store = VectorStore.load(settings.resolved_data_dir())
    retriever = HybridRetriever(store)
    by_page: dict[tuple, set] = {}
    for meta in store.metadatas:
        by_page.setdefault((meta.get("source_file"), meta.get("page_start")), set()).add(chunk_id_for_metadata(meta))
    rng = random.Random(7)
    pool = [m for m in store.metadatas
            if m.get("document_type") and "research_paper" not in str(m.get("source_file")) and _sentences(m.get("text", ""))]
    sample = rng.sample(pool, min(n, len(pool)))
    queries = []
    for m in sample:
        s = rng.choice(_sentences(m["text"]))
        queries.append((s, _keywords(s), by_page[(m.get("source_file"), m.get("page_start"))]))

    methods = {
        "Dense (FAISS, MiniLM)": lambda q: [h["chunk_id"] for h in retriever._dense(q, 10)],
        "Hybrid (dense + BM25, RRF)": lambda q: [h["chunk_id"] for h in retriever.retrieve(
            q, fused_k=10, final_k=10, use_reranker=False, use_expansion=False)],
        "Hybrid + cross-encoder rerank": lambda q: [h["chunk_id"] for h in retriever.retrieve(
            q, use_reranker=True, use_expansion=False)],
    }
    out = {"queries": len(queries), "corpus_chunks": len(store.metadatas), "methods": []}
    for qtype, idx in (("sentence", 0), ("keywords", 1)):
        for name, fn in methods.items():
            cases, times = [], []
            for q in queries:
                t = time.perf_counter()
                ranking = fn(q[idx])
                times.append(time.perf_counter() - t)
                cases.append((q[2], ranking))
            row = {"method": name, "query_type": qtype, **_rank_metrics(cases), "latency_ms": ms(statistics.median(times))}
            out["methods"].append(row)
            print(f"  retrieval  {qtype:9s} {name:32s} R@1 {row['recall@1']:.2f} R@5 {row['recall@5']:.2f} "
                  f"MRR {row['mrr']:.2f}  {row['latency_ms']} ms")
    return out


# ── Knowledge Maps verification ──────────────────────────────────────────────

def bench_verify(workers: int) -> dict:
    from backend.common.config import settings
    from backend.common.vector_store import VectorStore
    from backend.components.knowledge_maps.verification import _verify_claim

    store = VectorStore.load(settings.resolved_data_dir())
    items = [(c, True) for c in TRUE_CLAIMS] + [(c, False) for c in FALSE_CLAIMS]
    fast = bool(settings.m4_verify_fast)

    def run(item):
        t = time.perf_counter()
        claim = _verify_claim(item[0], store, "en", fast=fast)
        return item, claim.verdict, claim.source, time.perf_counter() - t

    _verify_claim(TRUE_CLAIMS[0], store, "en", fast=fast)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(run, items))
    wall = time.perf_counter() - t0

    rows, decisive = [], []
    for (claim, truth), verdict, source, secs in results:
        pred = {"correct": True, "incorrect": False, "incomplete": False}.get(verdict)
        rows.append({"claim": claim, "truth": truth, "verdict": verdict, "source": source, "seconds": round(secs, 2)})
        if pred is not None:
            decisive.append((truth, pred))
    tp = sum(1 for t, p in decisive if not t and not p)
    fp = sum(1 for t, p in decisive if t and not p)
    fn = sum(1 for (c, t), v, *_ in results if not t and v == "correct")
    acc = sum(t == p for t, p in decisive) / len(decisive) if decisive else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec_all = tp / len(FALSE_CLAIMS)
    out = {
        "claims": len(items), "true_claims": len(TRUE_CLAIMS), "false_claims": len(FALSE_CLAIMS),
        "mode": "fast" if fast else "full",
        "verdict_model": (settings.m4_verify_model or settings.m4_ollama_model) if fast else settings.m4_ollama_model,
        "coverage": len(decisive) / len(items),
        "accuracy_decisive": acc,
        "accuracy_overall": sum(t == p for t, p in decisive) / len(items),
        "error_detection_precision": prec,
        "error_detection_recall": rec_all,
        "error_detection_f1": 2 * prec * rec_all / (prec + rec_all) if prec + rec_all else 0.0,
        "false_accepted": fn,
        "median_claim_seconds": round(statistics.median(r["seconds"] for r in rows), 2),
        "wall_seconds": round(wall, 1), "workers": workers, "per_claim": rows,
    }
    print(f"  verify  coverage {out['coverage']:.2f}  acc(decisive) {acc:.2f}  error-F1 {out['error_detection_f1']:.2f}  "
          f"median {out['median_claim_seconds']} s/claim")
    return out


# ── Voice Tutor ──────────────────────────────────────────────────────────────

def bench_voice(base: str) -> dict:
    import httpx

    from backend.common.config import settings

    rows = []
    with httpx.Client(timeout=600) as client:
        client.post(f"{base}/api/voice-tutor/ask", json={"question": "What is a cell?"})
        for q in VOICE_QUESTIONS:
            t = time.perf_counter()
            r = client.post(f"{base}/api/voice-tutor/ask", json={"question": q})
            secs = time.perf_counter() - t
            body = r.json() if r.status_code == 200 else {}
            rows.append({"question": q, "status": r.status_code, "seconds": round(secs, 2),
                         "sources": len(body.get("sources") or []), "answer_words": len(str(body.get("answer", "")).split())})
            print(f"  voice  {q:58s} {r.status_code} {secs:5.1f}s sources {rows[-1]['sources']}")
    ok = [r for r in rows if r["status"] == 200]
    return {"chat_model": settings.ollama_chat_model, "questions": len(rows), "success_rate": len(ok) / len(rows),
            "median_seconds": round(statistics.median(r["seconds"] for r in ok), 2) if ok else None,
            "grounded_rate": sum(r["sources"] > 0 for r in ok) / len(ok) if ok else 0.0,
            "stt": "OpenAI whisper-1", "tts": settings.openai_tts_model, "per_question": rows}


# ── Narrative Learning (existing reports) ────────────────────────────────────

def bench_narrative() -> dict:
    clf = pd.read_csv(_REPO / "reports" / "classification" / "persona_classifier_metrics.csv").set_index("model")
    llm = pd.read_csv(_REPO / "reports" / "narrative" / "story_model_comparison" / "summary.csv").set_index("model")
    rf, g = clf.loc["Random Forest (one-hot)"], llm.loc["gemma3:4b"]
    return {"classifier_accuracy": rf["pooled_acc"], "classifier_f1": rf["pooled_f1"], "classifier_top2_real": rf["real_top2"],
            "story_fields_complete": g["fields_complete"], "story_check_valid": g["check_valid"],
            "story_grounding": g["grounding"], "story_latency_s": g["latency_s"], "story_fk_grade": g["fk_grade"],
            "story_models": llm.reset_index()[["model", "fields_complete", "check_valid", "latency_s", "grounding"]].to_dict("records")}


# ── Report ───────────────────────────────────────────────────────────────────

def summary_rows(R: dict) -> list[dict]:
    rows = []
    add = lambda comp, model, metric, value, bench="": rows.append(
        {"component": comp, "model": model, "metric": metric, "value": value, "benchmark": bench})
    pct = lambda v: f"{v * 100:.1f}%"
    if q := R.get("quiz"):
        m = {r["model"]: r for r in q["models"]}
        rf = m["Random Forest (deployed)"]
        others = ", ".join(f"{k} {v['f1_macro'] * 100:.1f}%" for k, v in m.items() if k != "Random Forest (deployed)")
        add("Adaptive Quiz – learning state", "Random Forest", "Accuracy (5-fold CV)", pct(rf["accuracy"]),
            f"majority baseline {pct(m['Majority baseline']['accuracy'])}")
        add("Adaptive Quiz – learning state", "Random Forest", "Macro-F1", pct(rf["f1_macro"]), others)
        add("Adaptive Quiz – learning state", "Random Forest", "Prediction latency", f"{q['predict_latency_ms']} ms")
    if e := R.get("emotion"):
        add("Adaptive Quiz – emotion", "YOLOv8s (3 classes)", "Precision / Recall", f"{pct(e['precision'])} / {pct(e['recall'])}", "validation set")
        add("Adaptive Quiz – emotion", "YOLOv8s (3 classes)", "F1", pct(e["f1"]))
        add("Adaptive Quiz – emotion", "YOLOv8s (3 classes)", "mAP@50 / mAP@50-95", f"{pct(e['map50'])} / {pct(e['map50_95'])}")
        add("Adaptive Quiz – emotion", "YOLOv8s (3 classes)", "CPU latency per frame", f"{e['cpu_latency_ms']} ms", f"imgsz {e['infer_imgsz']}")
    if r := R.get("retrieval"):
        k = {(x["method"], x["query_type"]): x for x in r["methods"]}
        best = k[("Hybrid + cross-encoder rerank", "keywords")]
        dense = k[("Dense (FAISS, MiniLM)", "keywords")]
        add("Shared retrieval (Voice Tutor, Knowledge Maps)", "Hybrid + reranker", "Recall@5 (keyword queries)",
            pct(best["recall@5"]), f"dense only {pct(dense['recall@5'])}")
        add("Shared retrieval (Voice Tutor, Knowledge Maps)", "Hybrid + reranker", "MRR (keyword queries)",
            f"{best['mrr']:.3f}", f"dense only {dense['mrr']:.3f}")
        add("Shared retrieval (Voice Tutor, Knowledge Maps)", "Hybrid + reranker", "Latency per query", f"{best['latency_ms']} ms",
            f"dense only {dense['latency_ms']} ms")
    if v := R.get("verify"):
        add("Knowledge Maps – verification", v["verdict_model"], "Accuracy (decided claims)", pct(v["accuracy_decisive"]),
            f"{v['claims']} hand-labelled claims")
        add("Knowledge Maps – verification", v["verdict_model"], "Error-detection F1", pct(v["error_detection_f1"]),
            f"P {pct(v['error_detection_precision'])}, R {pct(v['error_detection_recall'])}")
        add("Knowledge Maps – verification", v["verdict_model"], "Coverage (claims decided)", pct(v["coverage"]),
            "rest = insufficient evidence")
        add("Knowledge Maps – verification", v["verdict_model"], "Time per claim (throughput)",
            f"{v['wall_seconds'] / v['claims']:.1f} s", f"{v['workers']} claims in parallel; {v['median_claim_seconds']} s each")
    if n := R.get("narrative"):
        add("Narrative Learning – theme", "Random Forest (one-hot)", "Accuracy / Macro-F1 (5-fold)",
            f"{pct(n['classifier_accuracy'])} / {pct(n['classifier_f1'])}", "chance 20%")
        add("Narrative Learning – story", "gemma3:4b", "Complete output / valid quick-check",
            f"{pct(n['story_fields_complete'])} / {pct(n['story_check_valid'])}", "qwen2.5:3b 98% / 25%")
        add("Narrative Learning – story", "gemma3:4b", "Grounding in textbook", pct(n["story_grounding"]), "33B teacher 100%")
        add("Narrative Learning – story", "gemma3:4b", "Median time per lesson", f"{n['story_latency_s']} s", "llama3.1 91.6 s")
    if vo := R.get("voice"):
        add("Voice Tutor", vo["chat_model"], "Answered / grounded with sources",
            f"{pct(vo['success_rate'])} / {pct(vo['grounded_rate'])}", f"{vo['questions']} questions")
        add("Voice Tutor", vo["chat_model"], "Median answer time", f"{vo['median_seconds']} s")
    return rows


def write_md(R: dict, rows: list[dict]) -> None:
    lines = ["# STEMBuddy — overall system performance", "",
             "Generated by `scripts/system_benchmark.py`. All numbers are measured on this machine.", "",
             "| Component | Model | Metric | Score | Benchmark / comparison |", "|---|---|---|---|---|"]
    lines += [f"| {r['component']} | {r['model']} | {r['metric']} | **{r['value']}** | {r['benchmark']} |" for r in rows]
    lines += ["", "## How each number was measured", ""]
    if q := R.get("quiz"):
        lines.append(f"* **Adaptive Quiz learning state:** {q['rows']} labelled rows "
                     f"(4 states), repeated stratified 5-fold CV (5 repeats) on the deployed Random Forest and four baselines.")
    if e := R.get("emotion"):
        lines.append(f"* **Emotion:** {e['base_model']} fine-tuned for {e['epochs']} epochs on a 3-class "
                     f"({', '.join(e['classes'])}) face dataset; metrics are the checkpoint's validation scores. "
                     f"Latency is CPU inference on a 640×480 frame at imgsz {e['infer_imgsz']}.")
    if r := R.get("retrieval"):
        lines.append(f"* **Retrieval:** known-item search. {r['queries']} random textbook chunks (of {r['corpus_chunks']}); "
                     "the query is one sentence from the chunk (\"sentence\") or its first 5 content words (\"keywords\", "
                     "like a short student query). A hit is any chunk from the same textbook page.")
    if v := R.get("verify"):
        lines.append(f"* **Knowledge Maps verification:** {v['true_claims']} true and {v['false_claims']} false Grade 10–11 "
                     f"science claims (each false claim is a one-fact edit of a true one), {v['mode']} mode, verdicts by "
                     f"`{v['verdict_model']}`. Error detection = flagging false claims as incorrect/incomplete; "
                     f"{v['false_accepted']} false claims were accepted as correct.")
    if vo := R.get("voice"):
        lines.append(f"* **Voice Tutor:** {vo['questions']} syllabus questions through `/api/voice-tutor/ask` "
                     f"(retrieval + `{vo['chat_model']}`); speech uses {vo['stt']} and {vo['tts']} (cloud, not timed).")
    if R.get("narrative"):
        lines.append("* **Narrative Learning:** from `reports/classification/persona_classifier_metrics.md` and "
                     "`reports/narrative/story_model_comparison/summary.md`.")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(R: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    EM, TL, AM, GR = "#0f7b63", "#5fb3a1", "#e0a43a", "#9aa5a2"
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle("STEMBuddy — overall system performance", fontsize=16, fontweight="bold")

    def bars(ax, labels, series, title, ylim=105):
        x = np.arange(len(labels))
        w = 0.8 / len(series)
        for i, (name, vals, color) in enumerate(series):
            bs = ax.bar(x + (i - (len(series) - 1) / 2) * w, [v * 100 for v in vals], w, label=name, color=color)
            for b in bs:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}", ha="center", fontsize=7)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(0, ylim)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(series), frameon=False)

    ax = axes[0, 0]
    if q := R.get("quiz"):
        ms_ = q["models"]
        bars(ax, [m["model"].replace(" (deployed)", "\n(deployed)").replace("Logistic Regression", "Logistic\nRegression")
                  .replace("Majority baseline", "Majority\nbaseline") for m in ms_],
             [("Accuracy", [m["accuracy"] for m in ms_], EM), ("Macro-F1", [m["f1_macro"] for m in ms_], TL)],
             "Adaptive Quiz — learning-state model (5-fold CV)")

    ax = axes[0, 1]
    if e := R.get("emotion"):
        names = ["Precision", "Recall", "F1", "mAP@50", "mAP@50-95"]
        vals = [e["precision"], e["recall"], e["f1"], e["map50"], e["map50_95"]]
        bs = ax.bar(names, [v * 100 for v in vals], color=[EM, EM, AM, TL, TL])
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}", ha="center", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(f"Emotion detection — YOLOv8s ({e['cpu_latency_ms']:.0f} ms/frame CPU)", fontsize=11)

    ax = axes[0, 2]
    if r := R.get("retrieval"):
        k = {(x["method"], x["query_type"]): x for x in r["methods"]}
        meths = ["Dense (FAISS, MiniLM)", "Hybrid (dense + BM25, RRF)", "Hybrid + cross-encoder rerank"]
        bars(ax, ["Dense", "Hybrid\n(BM25 + RRF)", "Hybrid +\nreranker"],
             [("Recall@1 keywords", [k[(m, "keywords")]["recall@1"] for m in meths], TL),
              ("Recall@5 keywords", [k[(m, "keywords")]["recall@5"] for m in meths], EM),
              ("Recall@5 sentence", [k[(m, "sentence")]["recall@5"] for m in meths], AM)],
             f"Shared textbook retrieval ({r['queries']} queries)")

    ax = axes[1, 0]
    if v := R.get("verify"):
        names = ["Accuracy\n(decided)", "Error-detection\nprecision", "Error-detection\nrecall", "Error-detection\nF1", "Coverage"]
        vals = [v["accuracy_decisive"], v["error_detection_precision"], v["error_detection_recall"], v["error_detection_f1"], v["coverage"]]
        bs = ax.bar(names, [x * 100 for x in vals], color=[EM, TL, TL, AM, GR])
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}", ha="center", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(f"Knowledge Maps — claim verification ({v['claims']} claims, {v['verdict_model']})", fontsize=11)

    ax = axes[1, 1]
    if n := R.get("narrative"):
        names = ["Theme\naccuracy", "Theme\nmacro-F1", "Story complete\noutput", "Valid\nquick-check", "Textbook\ngrounding"]
        vals = [n["classifier_accuracy"], n["classifier_f1"], n["story_fields_complete"], n["story_check_valid"], n["story_grounding"]]
        bs = ax.bar(names, [x * 100 for x in vals], color=[EM, TL, EM, TL, AM])
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}", ha="center", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylim(0, 110)
        ax.set_title("Narrative Learning — Random Forest + gemma3:4b", fontsize=11)

    ax = axes[1, 2]
    lat = []
    if q := R.get("quiz"):
        lat.append(("Learning-state prediction", q["predict_latency_ms"] / 1000))
    if e := R.get("emotion"):
        lat.append(("Emotion (per frame)", e["cpu_latency_ms"] / 1000))
    if r := R.get("retrieval"):
        k = {(x["method"], x["query_type"]): x for x in r["methods"]}
        lat.append(("Retrieval + rerank", k[("Hybrid + cross-encoder rerank", "keywords")]["latency_ms"] / 1000))
    if v := R.get("verify"):
        lat.append(("Verify one claim (throughput)", v["wall_seconds"] / v["claims"]))
    if vo := R.get("voice"):
        if vo.get("median_seconds"):
            lat.append(("Voice Tutor answer", vo["median_seconds"]))
    if n := R.get("narrative"):
        lat.append(("Narrative lesson", n["story_latency_s"]))
    if lat:
        names, secs = zip(*lat)
        bs = ax.barh(names, secs, color=EM)
        ax.set_xscale("log")
        for b, s in zip(bs, secs):
            label = f"{s * 1000:.0f} ms" if s < 1 else f"{s:.1f} s"
            ax.text(b.get_width() * 1.15, b.get_y() + b.get_height() / 2, label, va="center", fontsize=8)
        ax.set_xlim(min(secs) / 3, max(secs) * 8)
        ax.invert_yaxis()
        ax.set_xlabel("seconds (log scale), measured on a laptop")
        ax.set_title("Response time per operation", fontsize=11)

    for ax in axes.flat:
        if not ax.has_data():
            ax.axis("off")
    fig.savefig(OUT / "system_performance.png", dpi=170)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=SECTIONS, default=SECTIONS)
    ap.add_argument("--retrieval-queries", type=int, default=100)
    ap.add_argument("--verify-workers", type=int, default=4)
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--plot-only", action="store_true", help="Rebuild report and chart from results.json")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    R = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not args.plot_only:
        runners = {"quiz": bench_quiz, "emotion": bench_emotion,
                   "retrieval": lambda: bench_retrieval(args.retrieval_queries),
                   "verify": lambda: bench_verify(args.verify_workers),
                   "voice": lambda: bench_voice(args.server), "narrative": bench_narrative}
        for name in args.only:
            print(f"[{name}]")
            try:
                R[name] = runners[name]()
            except Exception as exc:
                print(f"  {name} failed: {exc!r}")
            path.write_text(json.dumps(R, indent=2, default=float), encoding="utf-8")
    rows = summary_rows(R)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    write_md(R, rows)
    plot(R)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

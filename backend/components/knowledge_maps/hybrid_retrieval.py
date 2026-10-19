"""Member 4 hybrid dense/BM25 retrieval with optional local reranking."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import unicodedata
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from rank_bm25 import BM25Okapi

from backend.common.config import settings
from backend.common.embeddings import embed_query
from backend.common.vector_store import VectorStore

logger = logging.getLogger(__name__)


def normalise_query(value: str) -> str:
    """Normalize matching text without discarding Sinhala or science operators."""
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(normalized.split())


def tokenise(value: str) -> list[str]:
    """Unicode-aware tokenizer that retains formulas and operator tokens."""
    text = normalise_query(value)
    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    for char in text:
        category = unicodedata.category(char)
        if category[0] in {"L", "N", "M"} or char == "_":
            current.append(char)
        elif char in {"+", "-", "=", "/"}:
            flush()
            tokens.append(char)
        else:
            flush()
    flush()
    return tokens


def chunk_id_for_metadata(meta: dict) -> str:
    source = str(meta.get("source_file") or "").replace("\\", "/")
    payload = "\x1f".join(
        (source, str(meta.get("page_start") or ""), str(meta.get("page_end") or ""), str(meta.get("text") or ""))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class QueryGlossary:
    def __init__(self, entries: dict[str, list[str]] | None = None) -> None:
        self.entries = {
            tuple(tokenise(key)): [normalise_query(term) for term in values if normalise_query(term)]
            for key, values in (entries or {}).items()
            if tokenise(key)
        }

    @classmethod
    def load(cls, path: Path | None) -> "QueryGlossary":
        if path is None or not Path(path).is_file():
            return cls()
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("glossary root must be an object")
            entries = {
                str(key): [str(item) for item in value]
                for key, value in raw.items()
                if isinstance(value, list)
            }
            return cls(entries)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid Member 4 query glossary %s: %s", path, exc)
            return cls()

    def expand(self, query: str, limit: int = 5) -> list[str]:
        query_tokens = tokenise(query)
        output: list[str] = []
        for phrase, expansions in self.entries.items():
            width = len(phrase)
            matched = any(tuple(query_tokens[i : i + width]) == phrase for i in range(len(query_tokens) - width + 1))
            if not matched:
                continue
            for expansion in expansions:
                if expansion not in output:
                    output.append(expansion)
                if len(output) >= limit:
                    return output
        return output


def reciprocal_rank_fusion(
    dense_hits: list[dict], keyword_hits: list[dict], rrf_k: int = 60
) -> list[dict]:
    merged: dict[str, dict] = {}
    for ranking in (dense_hits, keyword_hits):
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = str(hit["chunk_id"])
            row = merged.setdefault(chunk_id, {"chunk_id": chunk_id, "fusion_score": 0.0})
            row.update({key: value for key, value in hit.items() if value is not None})
            row["fusion_score"] += 1.0 / (rrf_k + rank)
    return sorted(merged.values(), key=lambda row: (-float(row["fusion_score"]), row["chunk_id"]))


class LazyCrossEncoderReranker:
    """Thread-safe lazy wrapper so API startup never waits for a model download."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        *,
        model_loader: Callable[..., object] | None = None,
    ) -> None:
        self.model_name = model_name
        self.revision = revision or None
        self.resolved_revision: str | None = None
        self.loaded = False
        self.error: str | None = None
        self._model = None
        self._attempted = False
        self._model_loader = model_loader
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        if self._attempted and self.error:
            raise RuntimeError(self.error)
        with self._lock:
            if self._model is not None:
                return self._model
            if self._attempted and self.error:
                raise RuntimeError(self.error)
            self._attempted = True
            try:
                kwargs = {"revision": self.revision} if self.revision else {}
                if self._model_loader is None:
                    from sentence_transformers import CrossEncoder
                    loader = CrossEncoder
                    # CPU: Apple MPS segfaults when verify reranks from parallel threads.
                    kwargs["device"] = "cpu"
                else:
                    loader = self._model_loader

                self._model = loader(self.model_name, **kwargs)
                config = getattr(getattr(self._model, "model", None), "config", None)
                self.resolved_revision = getattr(config, "_commit_hash", None) or self.revision
                self.loaded = True
                self.error = None
            except Exception as exc:
                self.error = str(exc)
                raise
        return self._model

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._load()
        scores = model.predict(pairs, show_progress_bar=False)
        return [float(value) for value in np.asarray(scores).reshape(-1)]


def _resolved_optional_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else settings.project_root / path


def effective_retriever_signature(*, reranker_revision: str | None = None) -> str:
    """Fingerprint every setting that changes retrieval score distributions."""
    glossary_path = _resolved_optional_path(settings.m4_query_glossary_path)
    glossary_hash = None
    if glossary_path and glossary_path.is_file():
        try:
            glossary_hash = hashlib.sha256(glossary_path.read_bytes()).hexdigest()
        except OSError:
            glossary_hash = "unreadable"
    payload = {
        "version": "hybrid_bm25_rrf_bge_v1",
        "dense_k": settings.m4_hybrid_dense_k,
        "bm25_k": settings.m4_hybrid_bm25_k,
        "fused_k": settings.m4_hybrid_fused_k,
        "final_k": settings.m4_hybrid_final_k,
        "rrf_k": settings.m4_rrf_k,
        "reranker_model": settings.m4_reranker_model,
        "reranker_revision": reranker_revision or settings.m4_reranker_revision or "unpinned",
        "glossary_sha256": glossary_hash,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"hybrid_bm25_rrf_bge_v1:{digest}"


def load_retrieval_policy(path: Path | None) -> dict | None:
    if path is None or not Path(path).is_file():
        return None
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        threshold = raw.get("minimum_reranker_score")
        valid = (
            raw.get("schema_version") == 1
            and raw.get("retriever_signature") == effective_retriever_signature()
            and raw.get("reranker_model") == settings.m4_reranker_model
            and bool(settings.m4_reranker_revision)
            and raw.get("reranker_revision") == settings.m4_reranker_revision
            and isinstance(threshold, (int, float))
            and math.isfinite(float(threshold))
            and isinstance(raw.get("validation"), dict)
            and raw["validation"].get("split") == "validation"
        )
        if not valid:
            raise ValueError("policy does not match the active retriever")
        return raw
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid Member 4 retrieval policy %s: %s", path, exc)
        return None


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        *,
        reranker: Callable[[list[tuple[str, str]]], Iterable[float]] | None = None,
        glossary: QueryGlossary | None = None,
        query_embedder: Callable[[str], object] | None = None,
        policy: dict | None = None,
    ) -> None:
        self.store = store
        self.rows: list[dict] = []
        self._row_by_id: dict[str, dict] = {}
        corpus: list[list[str]] = []
        for meta in store.metadatas:
            row = dict(meta)
            row["chunk_id"] = chunk_id_for_metadata(row)
            self.rows.append(row)
            self._row_by_id[row["chunk_id"]] = row
            corpus.append(tokenise(str(row.get("text") or "")))
        self.bm25 = BM25Okapi(corpus) if corpus else None
        self.glossary = glossary or QueryGlossary.load(_resolved_optional_path(settings.m4_query_glossary_path))
        self.query_embedder = query_embedder or embed_query
        self.reranker = reranker if reranker is not None else LazyCrossEncoderReranker(
            settings.m4_reranker_model, settings.m4_reranker_revision
        )
        self.policy = policy if policy is not None else load_retrieval_policy(
            _resolved_optional_path(settings.m4_retrieval_policy_path)
        )
        self.last_method = "hybrid_rrf"

    def _dense(self, query: str, k: int) -> list[dict]:
        qv = np.asarray(self.query_embedder(query), dtype="float32")
        if isinstance(qv, np.ndarray) and qv.ndim == 2:
            qv = qv[0]
        output = []
        for score, meta in self.store.search(qv, k=k):
            row = dict(meta)
            row.update(chunk_id=chunk_id_for_metadata(meta), dense_score=float(score))
            output.append(row)
        return output

    def _keyword(self, query: str, k: int) -> list[dict]:
        if self.bm25 is None:
            return []
        terms = tokenise(query)
        if not terms:
            return []
        scores = self.bm25.get_scores(terms)
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        output = []
        for index in ranked:
            score = float(scores[index])
            if score <= 0:
                continue
            row = dict(self.rows[index])
            row["keyword_score"] = score
            output.append(row)
            if len(output) >= k:
                break
        return output

    def retrieve(
        self,
        query: str,
        *,
        dense_k: int | None = None,
        keyword_k: int | None = None,
        fused_k: int | None = None,
        final_k: int | None = None,
        use_reranker: bool = True,
        use_expansion: bool = True,
    ) -> list[dict]:
        dense_k = dense_k or settings.m4_hybrid_dense_k
        keyword_k = keyword_k or settings.m4_hybrid_bm25_k
        fused_k = fused_k or settings.m4_hybrid_fused_k
        final_k = final_k or settings.m4_hybrid_final_k
        expansions = self.glossary.expand(query) if use_expansion else []
        expanded_query = " ".join([normalise_query(query), *expansions]).strip()
        dense = self._dense(expanded_query, dense_k)
        keyword = self._keyword(expanded_query, keyword_k)
        fused = reciprocal_rank_fusion(dense, keyword, settings.m4_rrf_k)[:fused_k]
        for row in fused:
            base = self._row_by_id.get(row["chunk_id"], {})
            for key, value in base.items():
                row.setdefault(key, value)
            row.setdefault("dense_score", None)
            row.setdefault("keyword_score", None)
            row["reranker_score"] = None

        method = "hybrid_rrf"
        if use_reranker and fused:
            pairs = [(query, str(row.get("text") or "")) for row in fused]
            try:
                scores = list(self.reranker(pairs))
                if len(scores) != len(fused) or any(not math.isfinite(float(score)) for score in scores):
                    raise ValueError("reranker returned invalid scores")
                for row, score in zip(fused, scores, strict=True):
                    row["reranker_score"] = float(score)
                fused.sort(key=lambda row: (-float(row["reranker_score"]), -float(row["fusion_score"])))
                method = "hybrid_reranked"
                if self.policy:
                    threshold = float(self.policy["minimum_reranker_score"])
                    fused = [row for row in fused if float(row["reranker_score"]) >= threshold]
            except Exception as exc:
                logger.warning("Member 4 reranker unavailable; using RRF ordering: %s", exc)

        self.last_method = method
        output = []
        for row in fused[:final_k]:
            item = dict(row)
            item["retrieval_method"] = method
            # Compatibility: provisional confidence continues to use dense similarity.
            item["score"] = item.get("dense_score")
            output.append(item)
        return output

    def status(self) -> dict:
        reranker = self.reranker
        return {
            "retrieval_method": self.last_method,
            "retriever_signature": effective_retriever_signature(),
            "reranker_model": getattr(reranker, "model_name", None),
            "reranker_revision": getattr(reranker, "resolved_revision", None),
            "reranker_loaded": bool(getattr(reranker, "loaded", False)),
            "reranker_error": getattr(reranker, "error", None),
        }


_instances: dict[int, HybridRetriever] = {}
_instances_lock = threading.Lock()


def get_hybrid_retriever(store: VectorStore) -> HybridRetriever:
    key = id(store)
    with _instances_lock:
        retriever = _instances.get(key)
        if retriever is None or retriever.store is not store:
            retriever = HybridRetriever(store)
            _instances[key] = retriever
        return retriever

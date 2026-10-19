"""Claim verification pipeline (F2 + F3).

Three-stage flow:
  1. Extract claims from student text via structured LLM output.
  2. For each claim: retrieve syllabus context → structured LLM verdict.
  3. Claims that come back 'incorrect' are re-checked via web search (F3).
     If globally correct, the verdict is upgraded to 'correct' with source='web'.

Public API:
  verify_text(text, store, language) -> VerifyResponse
"""
from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict

from backend.components.knowledge_maps import llm_client
from backend.common.config import settings
from backend.components.knowledge_maps.concurrency import run_concurrent
from backend.components.knowledge_maps.confidence import confidence_for_claim
from backend.components.knowledge_maps.evidence import select_syllabus_evidence
from backend.components.knowledge_maps.hybrid_retrieval import effective_retriever_signature, get_hybrid_retriever
from backend.components.knowledge_maps.prompts import (
    CLAIM_EXTRACTION_SYSTEM,
    CLAIM_EXTRACTION_TOOL,
    VERDICT_SYSTEM,
    VERDICT_TOOL,
    language_instruction,
)
from backend.components.knowledge_maps.schemas import Claim, ScoreSummary, VerifyResponse
from backend.components.knowledge_maps.web_search import judge_web_evidence
from backend.common.retrieval import format_context_for_llm
from backend.common.vector_store import VectorStore

logger = logging.getLogger(__name__)

_MAX_CLAIMS = 25
_RETRIEVAL_K = 3  # top-3 reranked passages; fewer prompt tokens per claim
_FAST_RETRIEVAL_K = 2
_VERDICT_MAX_TOKENS = 350
_CACHE_SIZE = 512


class _LruCache:
    """Small thread-safe LRU so re-verifying the same note skips the LLM calls."""

    def __init__(self, maxsize: int) -> None:
        self._data: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key, value) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)


_claims_cache = _LruCache(_CACHE_SIZE)
_verdict_cache = _LruCache(_CACHE_SIZE)


def _cache_key(store: VectorStore, text: str, language: str) -> tuple:
    # The store object (not its id) is part of the key so a reloaded index never hits stale entries.
    return (store, " ".join(text.lower().split()), language, settings.m4_ollama_model)


def retrieve(store: VectorStore, question: str, k: int | None = None) -> list[dict]:
    """Compatibility seam for tests while routing Member 4 through hybrid search."""
    return get_hybrid_retriever(store).retrieve(question, final_k=k or _RETRIEVAL_K)


def _normalize_verdict(value: object) -> str | None:
    """Map small-model verdict variants to the API's strict vocabulary."""
    verdict = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    if verdict == "correct" or verdict in {"true", "right", "supported"}:
        return "correct"
    if verdict == "incorrect" or verdict in {"false", "wrong", "unsupported"}:
        return "incorrect"
    if verdict in {"incomplete", "partial", "partially correct"}:
        return "incomplete"
    if verdict in {"insufficient evidence", "unknown", "undecidable"}:
        return "insufficient_evidence"
    return None


def _extract_claims(text: str, language: str) -> list[str]:
    """Stage 1: extract distinct factual claims from student text."""
    lang_note = language_instruction(language)
    messages = [
        {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM},
        {
            "role": "user",
            "content": (
                f"{lang_note}\n\n"
                f"Student text:\n\"\"\"\n{text.strip()}\n\"\"\""
            ),
        },
    ]
    result = llm_client.chat_with_tools(
        messages=messages,
        tools=[CLAIM_EXTRACTION_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_claims"}},
    )
    raw = result.get("claims", [])
    # Sanitise: strings only, cap at limit
    return [str(c).strip() for c in raw if str(c).strip()][:_MAX_CLAIMS]


def _verdict_for_claim(
    claim: str,
    store: VectorStore,
    language: str,
    fast: bool = False,
) -> tuple[dict, list[dict]]:
    """Stage 2: retrieve syllabus context and get a verdict for one claim.

    ``fast`` sends fewer excerpts to the smaller M4_VERIFY_MODEL (when set).
    """
    hits = retrieve(store, claim, k=_FAST_RETRIEVAL_K if fast else _RETRIEVAL_K)
    hits = [{**hit, "evidence_id": f"S{i}"} for i, hit in enumerate(hits, start=1)]

    if hits:
        context = format_context_for_llm(hits)
        context_block = f"Syllabus context:\n{context}"
    else:
        return {
            "verdict": "insufficient_evidence",
            "explanation": "No relevant syllabus passage was found.",
            "evidence_ids": [],
        }, []

    lang_note = language_instruction(language)
    messages = [
        {"role": "system", "content": VERDICT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"{lang_note}\n\n"
                f"Student claim: \"{claim}\"\n\n"
                f"{context_block}"
            ),
        },
    ]
    result = llm_client.chat_with_tools(
        messages=messages,
        tools=[VERDICT_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_verdict"}},
        max_tokens=_VERDICT_MAX_TOKENS,
        ollama_model=(settings.m4_verify_model.strip() or None) if fast else None,
    )
    return result, hits


def response_for_claims(claims: list[Claim]) -> VerifyResponse:
    counts = {name: sum(c.verdict == name for c in claims) for name in (
        "correct", "incorrect", "incomplete", "insufficient_evidence", "verification_failed"
    )}
    decidable = counts["correct"] + counts["incorrect"] + counts["incomplete"]
    total = len(claims)
    score = round(counts["correct"] / decidable, 4) if decidable else None
    summary = ScoreSummary(
        total_claims=total, decidable_claims=decidable, coverage=round(decidable / total, 4) if total else 0.0,
        **counts,
    )
    return VerifyResponse(overall_score=score, score_summary=summary, claims=claims)


def _artifact_path():
    path = settings.m4_confidence_model_path
    if path is None:
        return None
    return path if path.is_absolute() else settings.project_root / path


def _failed_claim(claim: str, explanation: str) -> Claim:
    return Claim(
        claim=claim, verdict="verification_failed", explanation=explanation,
        source="none", evidence_status="unavailable",
        confidence=confidence_for_claim(
            "verification_failed", "none", [], [], "unavailable",
            retriever_signature=effective_retriever_signature(),
        ),
    )


def _verify_claim(claim_str: str, store: VectorStore, language: str, fast: bool = False) -> Claim:
    """Verify one claim; ``fast`` uses M4_VERIFY_MODEL and skips the web fallback."""
    try:
        raw, syllabus_hits = _verdict_for_claim(claim_str, store, language, fast=fast)
    except Exception as exc:
        logger.warning("Verdict call failed for claim %r: %s", claim_str, exc)
        return _failed_claim(claim_str, "Verification could not be completed. Please retry.")

    verdict = _normalize_verdict(raw.get("verdict"))
    if verdict is None:
        return _failed_claim(claim_str, "The verifier returned an invalid result. Please retry.")
    explanation = raw.get("explanation", "")
    corrected = raw.get("corrected_version") or None
    memory_tip = raw.get("memory_tip") or None
    source = "syllabus" if syllabus_hits else "none"
    selected_evidence, evidence_status = select_syllabus_evidence(
        syllabus_hits,
        raw.get("evidence_ids"),
        verdict,
    )
    if syllabus_hits and evidence_status == "unavailable":
        logger.warning(
            "Verdict for claim %r did not select valid evidence IDs: %r",
            claim_str,
            raw.get("evidence_ids"),
        )

    decisive = verdict in {"correct", "incorrect", "incomplete"} and evidence_status == "cited"
    if not decisive and fast:
        verdict = "insufficient_evidence"
        source = "none"
        corrected = None
        memory_tip = None
    elif not decisive:
        try:
            allowed = settings.m4_web_allowed_domains.split(",")
            web_check = judge_web_evidence(claim_str, allowed)
            if web_check.verdict in {"correct", "incorrect"} and web_check.evidence:
                verdict = web_check.verdict
                source = "web"
                explanation = "The claim was decided using cited approved web evidence."
                corrected = None
                memory_tip = None
                selected_evidence = list(web_check.evidence)
                evidence_status = "cited"
            else:
                verdict = "insufficient_evidence"
                source = "none"
                corrected = None
                memory_tip = None
        except Exception as exc:
            logger.warning("Web search fallback failed for %r: %s", claim_str, exc)
            return _failed_claim(claim_str, "External verification could not be completed. Please retry.")

    confidence = confidence_for_claim(
        verdict, source, selected_evidence, syllabus_hits, evidence_status, _artifact_path(),
        retriever_signature=effective_retriever_signature(),
    )

    return Claim(
        claim=claim_str,
        verdict=verdict,
        explanation=explanation,
        corrected_version=corrected,
        memory_tip=memory_tip,
        source=source,
        evidence_status=evidence_status,
        evidence=selected_evidence,
        confidence=confidence,
    )


def _cached_verify_claim(claim_str: str, store: VectorStore, language: str, fast: bool = False) -> Claim:
    key = (*_cache_key(store, claim_str, language), effective_retriever_signature(), fast)
    cached = _verdict_cache.get(key)
    if cached is not None:
        return cached.model_copy(update={"claim": claim_str})
    claim = _verify_claim(claim_str, store, language, fast=fast)
    if claim.verdict != "verification_failed":
        _verdict_cache.put(key, claim)
    return claim


def verify_claims(claims: list[str | dict], store: VectorStore, language: str = "auto") -> VerifyResponse:
    """Verify supplied claims (in parallel) while retaining optional source provenance."""
    prepared: list[tuple[str, dict]] = []
    for item in claims:
        if isinstance(item, dict):
            claim_str = str(item.get("claim") or "").strip()
            provenance = {
                "claim_id": item.get("claim_id"),
                "source_block_id": item.get("block_id") or item.get("source_block_id"),
                "source_start": item.get("source_start"),
                "source_end": item.get("source_end"),
            }
        else:
            claim_str = str(item).strip()
            provenance = {}
        if claim_str:
            prepared.append((claim_str, provenance))

    results = run_concurrent(
        prepared,
        worker=lambda entry: _cached_verify_claim(entry[0], store, language),
        on_error=lambda entry, exc: _failed_claim(
            entry[0], "Verification could not be completed. Please retry."
        ),
    )
    verified = [
        claim.model_copy(update={k: v for k, v in provenance.items() if v is not None})
        for claim, (_, provenance) in zip(results, prepared)
    ]
    return response_for_claims(verified)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _split_claims(text: str) -> list[str]:
    """Fast-mode claim extraction: one claim per sentence, no LLM call."""
    joined = re.sub(r"(?<!\n)\n(?!\n)", " ", text.strip())
    claims = [s.strip(" -•\t") for s in _SENTENCE_SPLIT.split(joined)]
    return [c for c in claims if len(c.split()) >= 3][:_MAX_CLAIMS]


def verify_text_fast(text: str, store: VectorStore, language: str = "auto") -> VerifyResponse:
    """Low-latency verify: sentence split instead of LLM extraction, no web fallback.

    Verdicts use M4_VERIFY_MODEL when set (a small model that fits fully on the GPU).
    """
    claims_text = _split_claims(text)
    results = run_concurrent(
        claims_text,
        worker=lambda claim: _cached_verify_claim(claim, store, language, fast=True),
        on_error=lambda claim, exc: _failed_claim(claim, "Verification could not be completed. Please retry."),
    )
    return response_for_claims(results)


def verify_text(text: str, store: VectorStore, language: str = "auto") -> VerifyResponse:
    """Run the full 3-stage verification pipeline and return a VerifyResponse."""
    if settings.m4_verify_fast:
        return verify_text_fast(text, store, language)
    key = _cache_key(store, text, language)
    claims_text = _claims_cache.get(key)
    if claims_text is None:
        claims_text = _extract_claims(text, language)
        if claims_text:
            _claims_cache.put(key, claims_text)
    return verify_claims(list(claims_text), store, language)

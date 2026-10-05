"""LLM answer with strict syllabus-only system prompt; English / Sinhala / auto response."""
from __future__ import annotations

import httpx
from openai import OpenAI

from backend.common.config import settings

BASE_SYSTEM = (
    "You are a helpful Sri Lankan OL science tutor. Answer the student's question clearly and accurately "
    "using the provided syllabus context.\n"
    "- Combine definitions, formulas, and facts present across the context chunks to answer comparisons and conceptual questions.\n"
    "- Do not invent outside facts.\n"
    "- If and only if the topic is completely absent and cannot be answered from the context, use the required refusal phrase."
)

REFUSAL_EN = "This is not in the syllabus."
REFUSAL_SI = "මෙය විෂය මාලාවේ හෝ ලබා දී ඇති සම්පත්වල අන්තර්ගත නොවේ."

LANG_RULES = {
    "en": (
        f"If the context truly does not contain the information, reply with exactly: {REFUSAL_EN!r}\n"
        "Otherwise, write a clear, helpful answer in English based on the context."
    ),
    "si": (
        f"If the context truly does not contain the information, reply with exactly: {REFUSAL_SI}\n"
        "Otherwise, write a clear, helpful answer in Sinhala (සිංහල අකුරු භාවිතා කරන්න) based on the context."
    ),
    "auto": (
        f"If the context truly does not contain the information, reply in the student's language: "
        f"use {REFUSAL_EN!r} for English questions and {REFUSAL_SI} for Sinhala questions.\n"
        "Otherwise, answer in the same language as the student question (English or Sinhala) using the context."
    ),
}


def answer_question(
    context: str,
    question: str,
    response_language: str = "auto",
) -> str:
    lang = (response_language or "auto").lower()
    if lang not in LANG_RULES:
        lang = "auto"
    system_content = BASE_SYSTEM + "\n\n" + LANG_RULES[lang]
    if settings.openai_api_key and settings.openai_api_key.strip():
        return _openai_answer(system_content, context, question)
    return _ollama_answer(system_content, context, question)


def _openai_answer(system_content: str, context: str, question: str) -> str:
    client = OpenAI(api_key=settings.openai_api_key)
    user_content = f"""Context from syllabus materials:\n\n{context}\n\nStudent question:\n{question}"""
    r = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )
    if not r.choices or not r.choices[0].message.content:
        return REFUSAL_EN
    return r.choices[0].message.content.strip()


def _ollama_chat_models(base: str) -> list[str]:
    with httpx.Client(timeout=10.0) as client:
        r = client.get(f"{base.rstrip('/')}/api/tags")
        r.raise_for_status()
        names = [
            (m.get("name") or m.get("model") or "").strip()
            for m in (r.json().get("models") or [])
        ]
    skip = ("embed", "nomic")
    return [n for n in names if n and not any(s in n.lower() for s in skip)]


def _pick_installed_model(configured: str, installed: list[str]) -> str | None:
    if not installed:
        return None
    want = (configured or "").strip().lower()
    want_base = want.split(":")[0]
    for name in installed:
        low = name.lower()
        if low == want or low.startswith(f"{want_base}:"):
            return name
        if low.split(":")[0] == want_base:
            return name
    return installed[0]


def _ollama_answer(system_content: str, context: str, question: str) -> str:
    base = settings.ollama_base_url.rstrip("/")
    url = f"{base}/api/chat"
    user_content = f"""Context from syllabus materials:\n\n{context}\n\nStudent question:\n{question}"""
    model = (settings.ollama_chat_model or "").strip() or "llama3.1"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    with httpx.Client(timeout=settings.ollama_timeout_seconds) as client:
        r = client.post(url, json=payload)
        if r.status_code == 404:
            installed = _ollama_chat_models(base)
            fallback = _pick_installed_model(model, installed)
            err = ""
            try:
                err = (r.json() or {}).get("error") or r.text
            except Exception:
                err = r.text
            if not fallback or fallback.lower() == model.lower():
                have = ", ".join(installed) or "(none)"
                raise RuntimeError(
                    f"Ollama has no model '{model}'. Installed chat models: {have}. "
                    f"Run `ollama pull {model}` or set OLLAMA_CHAT_MODEL in .env. ({err})"
                )
            payload["model"] = fallback
            r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    msg = data.get("message", {}) or {}
    content = (msg.get("content") or "").strip()
    return content or REFUSAL_EN

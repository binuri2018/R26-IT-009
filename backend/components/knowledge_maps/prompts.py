"""Centralized prompts and structured-output schemas for Member 4.

Each feature defines a system prompt and tool spec. The LLM client uses native
tool calling with a JSON-schema fallback. Frontend renderers depend on these
exact response shapes.
"""
from __future__ import annotations

from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHESIS — Flashcards / Mindmap / Structured Notes
# ─────────────────────────────────────────────────────────────────────────────


def language_instruction(language: str) -> str:
    """Shared language directive used by every Member-4 prompt."""
    lang = (language or "auto").lower()
    if lang == "si":
        return "Write everything in Sinhala (සිංහල අකුරු භාවිතා කරන්න). Keep scientific names / formulas in their standard form."
    if lang == "en":
        return "Write everything in clear English."
    return "Match the language of the input text. If the text is in Sinhala, reply in Sinhala; otherwise reply in English."


# ── Flashcards ──────────────────────────────────────────────────────────────

FLASHCARDS_SYSTEM = (
    "You are a science tutor generating spaced-repetition flashcards. "
    "Read the source text and produce 10–15 high-quality Q&A pairs that cover "
    "the most important factual content. Front = a focused question. "
    "Back = a concise, complete answer. No multi-sentence questions. "
    "Do not invent facts that are not in the source."
)

FLASHCARDS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_flashcards",
        "description": "Return a deck of 10–15 flashcards generated from the source text.",
        "parameters": {
            "type": "object",
            "properties": {
                "flashcards": {
                    "type": "array",
                    "minItems": 6,
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": {
                            "front": {"type": "string", "description": "Question or prompt."},
                            "back": {"type": "string", "description": "Answer or explanation."},
                        },
                        "required": ["front", "back"],
                    },
                }
            },
            "required": ["flashcards"],
        },
    },
}


# ── Mind map ────────────────────────────────────────────────────────────────

MINDMAP_SYSTEM = (
    "You are a study mind-map generator. Turn the source text into a RADIAL mind "
    "map that has ONE central topic with concepts branching outward from it.\n"
    "- Pick the single most central concept as the root and set 'root' to its id.\n"
    "- Give every other node a 'parent': the id of the concept it hangs off. "
    "Make 2–5 main branches directly under the root, then nest sub-concepts up to "
    "3 levels deeper.\n"
    "- 8–15 nodes total. Labels are 1–4 words. 'group' names the branch/theme the "
    "node belongs to (e.g. 'inputs', 'process', 'products').\n"
    "- 'importance' is a decimal 0–1; the root is the highest.\n"
    "- Use 'edges' ONLY for extra relationships that are NOT parent→child links "
    "(e.g. 'produces', 'requires', 'converts to') between concepts on different "
    "branches. Never restate a parent link as an edge.\n"
    "Every id is a short slug; every 'root', 'parent', 'source' and 'target' MUST "
    "be one of the node ids you define. Do not invent facts absent from the text."
)

MINDMAP_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_mindmap",
        "description": (
            "Return a radial study mind map: one root concept, a parent link on "
            "every other node, plus optional non-hierarchical cross-link edges."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "root": {
                    "type": "string",
                    "description": "id of the single central concept.",
                },
                "nodes": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Stable unique id (slug)."},
                            "label": {"type": "string", "description": "Short display label."},
                            "group": {"type": "string", "description": "Branch / theme name."},
                            "importance": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "parent": {
                                "type": "string",
                                "description": (
                                    "id of this node's parent concept. Omit only "
                                    "for the root node."
                                ),
                            },
                        },
                        "required": ["id", "label", "group", "importance"],
                    },
                },
                "edges": {
                    "type": "array",
                    "description": "Cross-links only — never parent→child pairs.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "id of source node"},
                            "target": {"type": "string", "description": "id of target node"},
                            "relation": {
                                "type": "string",
                                "description": "Short relation label (e.g. 'produces').",
                            },
                        },
                        "required": ["source", "target", "relation"],
                    },
                },
            },
            "required": ["root", "nodes", "edges"],
        },
    },
}


# ── Structured notes ────────────────────────────────────────────────────────

NOTES_SYSTEM = (
    "You are a study-notes generator. Produce a clear, hierarchical outline "
    "of the source text. Each section has a heading, 3–6 bullet points, and "
    "a list of key terms a student should remember. Faithful to the source — "
    "no invented facts. Aim for 3–6 sections covering the whole text."
)

NOTES_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_notes",
        "description": "Return structured study notes for the source text.",
        "parameters": {
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 10,
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "bullets": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                            "key_terms": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["heading", "bullets", "key_terms"],
                    },
                }
            },
            "required": ["sections"],
        },
    },
}


SYNTHESIS_TOOLS = {
    "flashcards": (FLASHCARDS_SYSTEM, FLASHCARDS_TOOL),
    "mindmap": (MINDMAP_SYSTEM, MINDMAP_TOOL),
    "notes": (NOTES_SYSTEM, NOTES_TOOL),
}


def synthesis_user_message(text: str, language: str) -> str:
    """User-side message body used for any synthesis call."""
    return (
        f"{language_instruction(language)}\n\n"
        f"Source text:\n\"\"\"\n{text.strip()}\n\"\"\""
    )


# ─────────────────────────────────────────────────────────────────────────────
# OCR — handwriting / printed text from images via the selected vision model
# ─────────────────────────────────────────────────────────────────────────────

OCR_PROMPT = (
    "You are an OCR engine. Extract ALL legible text from the image in reading "
    "order and preserve line breaks. Preserve English, Sinhala (සිංහල), numbers, "
    "scientific formulas, symbols, punctuation, and original spelling exactly. "
    "Do not describe, correct, summarize, or translate the content. If a word is "
    "illegible, write [?]. Output ONLY the extracted text: no commentary, labels, "
    "or Markdown fences."
)

OCR_REVIEW_PROMPT = (
    "You are an OCR quality reviewer for Sinhala, English, and mixed-language "
    "student science notes. Read the image and return strict JSON only. Preserve "
    "reading order, line breaks, Sinhala characters, English terms, punctuation, "
    "scientific symbols, arrows, formulas, and table structure as plain text. Do "
    "not translate, correct facts, or silently normalize the student's wording. "
    "Return exactly this shape: {\"text\":\"complete OCR text\",\"regions\":["
    "{\"region_id\":\"R1\",\"region_type\":\"paragraph|table|equation|label|diagram\","
    "\"text\":\"region text\",\"confidence\":0.0,\"reading_order\":1,"
    "\"warnings\":[\"mixed_language\"]}],\"uncertain_spans\":[{\"source_text\":\"C02\","
    "\"suggested_text\":\"CO₂\",\"reason\":\"Possible scientific-notation confusion\","
    "\"start\":0,\"end\":3,\"confidence\":0.72,"
    "\"category\":\"formula|terminology|spelling|unclear\"}]}"
)


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION — claim extraction + per-claim syllabus verdict
# ─────────────────────────────────────────────────────────────────────────────

CLAIM_EXTRACTION_SYSTEM = (
    "You are a science fact-checker. Given a piece of student text, extract every "
    "distinct factual claim as a short, self-contained sentence. "
    "A 'claim' is any statement that can be verified (e.g. 'Photosynthesis occurs in "
    "chloroplasts'). Ignore opinions, questions, and definitions that are clearly "
    "definitional rather than factual. Cap at 25 claims."
)

CLAIM_EXTRACTION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_claims",
        "description": "Return all distinct factual claims found in the student text.",
        "parameters": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 25,
                    "items": {"type": "string", "description": "One self-contained factual claim."},
                }
            },
            "required": ["claims"],
        },
    },
}

VERDICT_SYSTEM = (
    "You are a science tutor checking a student's claim against syllabus excerpts. "
    "Use ONLY the provided syllabus context to decide:\n"
    "  'correct'    — the claim is fully supported by the syllabus context.\n"
    "  'incorrect'  — the claim contradicts the syllabus context.\n"
    "  'incomplete' — the claim is partially right but missing important nuance.\n"
    "  'insufficient_evidence' — the supplied excerpts cannot decide the claim.\n"
    "The verdict must be exactly one of correct, incorrect, incomplete, or insufficient_evidence. "
    "Each syllabus excerpt is labelled with an evidence ID such as S1. "
    "Return only the IDs of excerpts that materially justify the verdict. "
    "Return an empty evidence_ids list when none of the excerpts justifies it. "
    "Give a concise explanation (1–2 sentences). "
    "If incorrect or incomplete, provide a corrected version and a short memory tip "
    "a student can use to remember the right answer.\n"
    "Rules: If the excerpt states the same fact, even in different words or units "
    "(e.g. 100 °C = 373 K), the verdict is 'correct'. Use 'incorrect' ONLY when the excerpt "
    "states something that conflicts with the claim. Write the explanation first "
    "(one sentence), then choose the verdict that matches it."
)

VERDICT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_verdict",
        "description": "Return the verdict for one student claim vs syllabus context.",
        "parameters": {
            "type": "object",
            # Explanation comes first so small local models reason before deciding;
            # maxLength stops the explanation running on until the token cap.
            "properties": {
                "explanation": {
                    "type": "string",
                    "maxLength": 220,
                    "description": "One sentence: what the excerpt says compared with the claim.",
                },
                "verdict": {
                    "type": "string",
                    "enum": ["correct", "incorrect", "incomplete", "insufficient_evidence"],
                    "description": "Verdict based solely on the syllabus context provided.",
                },
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                    "description": "IDs such as S1 for excerpts that directly justify the verdict.",
                },
                "corrected_version": {
                    "type": "string",
                    "maxLength": 160,
                    "description": "Corrected claim (only when verdict is incorrect or incomplete).",
                },
                "memory_tip": {
                    "type": "string",
                    "maxLength": 120,
                    "description": "Short mnemonic or tip for a student to remember the correct fact.",
                },
            },
            "required": ["explanation", "verdict", "evidence_ids"],
        },
    },
}

WEB_JUDGE_SYSTEM = (
    "You are a science fact-checker. Given labelled web search results about a claim, "
    "use ONLY those results. Return correct when selected results directly support it, "
    "incorrect when selected results directly refute it, or insufficient_evidence when "
    "they cannot decide it. Select only IDs that materially justify the verdict."
)

WEB_JUDGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_web_verdict",
        "description": "Return a web-grounded verdict and the selected source IDs.",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["correct", "incorrect", "insufficient_evidence"],
                },
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["verdict", "evidence_ids"],
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# YOUTUBE — transcript extraction prompt (used before synthesis/verification)
# ─────────────────────────────────────────────────────────────────────────────

YOUTUBE_CLEAN_SYSTEM = (
    "You are a transcript cleaner. The following text is an auto-generated YouTube "
    "transcript that may contain broken sentences, filler words, and formatting "
    "artifacts. Rewrite it as clean, readable prose paragraphs without changing any "
    "facts or adding new information. Preserve technical terms exactly."
)


def youtube_clean_user(raw: str) -> str:
    return f"Transcript:\n\"\"\"\n{raw.strip()}\n\"\"\""


# ─────────────────────────────────────────────────────────────────────────────
# CORRECT NOTE — rewrite verified-correct claims into a clean note
# ─────────────────────────────────────────────────────────────────────────────

CORRECT_NOTE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_corrected_note",
        "description": (
            "Return a clean, coherent study note built only from the verified-correct claims."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "corrected_note": {
                    "type": "string",
                    "description": (
                        "A well-structured note in clear prose or bullet points. "
                        "Contains ONLY the verified-correct claims. No new facts added."
                    ),
                }
            },
            "required": ["corrected_note"],
        },
    },
}


def correct_note_system(language: str) -> str:
    lang = language_instruction(language)
    return (
        f"{lang} "
        "You are a science note rewriter for Grade 10–11 students. "
        "You are given a list of verified-correct science claims. "
        "Rewrite them as a clean, coherent study note using clear bullet points "
        "or short paragraphs. Do NOT add any new facts. Do NOT include claims "
        "that are not in the list. Keep scientific terms exact."
    )


def correct_note_user(kept_claims: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(kept_claims))
    return f"Verified-correct claims to include:\n{numbered}"


# ─────────────────────────────────────────────────────────────────────────────
# REPAIR NOTE — structure-aware extraction and evidence-constrained correction
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURED_CLAIM_EXTRACTION_SYSTEM = (
    "Extract every factual science claim from the labelled note blocks. "
    "For each claim, return the exact block_id, copy the smallest complete source_fragment "
    "verbatim from that block, and provide a self-contained claim for verification. "
    "Never invent or normalize source_fragment text. Keep claims in source order."
)

STRUCTURED_CLAIM_EXTRACTION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_structured_claims",
        "description": "Return factual claims mapped to exact note fragments.",
        "parameters": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "maxItems": 25,
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_id": {"type": "string"},
                            "block_id": {"type": "string"},
                            "source_fragment": {"type": "string"},
                            "claim": {"type": "string"},
                        },
                        "required": ["claim_id", "block_id", "source_fragment", "claim"],
                    },
                }
            },
            "required": ["claims"],
        },
    },
}

REPAIR_CLAIM_SYSTEM = (
    "Repair one incorrect or incomplete student claim using ONLY the supplied selected evidence. "
    "Preserve the student's intended topic and wording where possible. Add no fact that is absent "
    "from the evidence. Return one self-contained proposed claim, a short change reason, and only "
    "the evidence IDs that directly justify the repair."
)

REPAIR_CLAIM_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_claim_repair",
        "description": "Return an evidence-constrained proposed repair.",
        "parameters": {
            "type": "object",
            "properties": {
                "proposed_claim": {"type": "string"},
                "change_reason": {"type": "string"},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["proposed_claim", "change_reason", "evidence_ids"],
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACT TAGS — auto-tag a corrected note for YouTube topic search
# ─────────────────────────────────────────────────────────────────────────────

EXTRACT_TAGS_SYSTEM = (
    "You are a science topic tagger. Extract 3–6 concise topic tags from the "
    "provided science note. Each tag should be a short searchable phrase "
    "(1–3 words) suitable for a YouTube search (e.g. 'photosynthesis', "
    "'cell division', 'Newton laws'). Focus on science concepts — not generic "
    "words like 'note', 'study', or 'grade'."
)

EXTRACT_TAGS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_tags",
        "description": "Return 3–6 short searchable topic tags extracted from the science note.",
        "parameters": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 6,
                    "items": {"type": "string", "description": "Short searchable topic tag."},
                }
            },
            "required": ["tags"],
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO SUMMARY — lightweight transcript → note-ready summary
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_SUMMARY_SYSTEM = (
    "You are a study assistant. Write a concise 2–3 paragraph summary of the "
    "science content in this transcript, suitable for appending to a student's notes. "
    "Cover the main concepts, key facts, and any examples given. Use clear English. "
    "Do not invent facts not present in the transcript."
)

VIDEO_SUMMARY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_video_summary",
        "description": "Return a concise note-ready summary of the video transcript.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "2–3 paragraph summary of the video content.",
                },
                "key_points": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {"type": "string"},
                    "description": "3–5 key takeaway bullet points.",
                },
            },
            "required": ["summary", "key_points"],
        },
    },
}

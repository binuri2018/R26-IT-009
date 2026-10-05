import os
import json
import pickle
import re
import warnings
import logging

# Suppress noisy sklearn version warnings at startup
warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

from backend.components.narrative_learning.model_config import (
    LLM_PROVIDER,
    VECTOR_DB_DIR,
    get_embeddings,
    get_llm,
)

_PERSONA_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona_model.pkl")
_ENCODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoder.pkl")


def _llm_text(response):
    """Flatten LangChain message content (string, blocks, or reasoning)."""
    parts = []
    content = getattr(response, "content", response)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(block))
    elif content:
        parts.append(str(content))
    extra = getattr(response, "additional_kwargs", None) or {}
    reasoning = extra.get("reasoning_content")
    if reasoning:
        parts.append(str(reasoning))
    return "\n".join(p for p in parts if str(p).strip())


def _parse_json_response(content):
    """Pull the first usable JSON object out of messy LLM text."""
    if hasattr(content, "content"):
        raw = _llm_text(content)
    elif isinstance(content, list):
        raw = _llm_text(content)
    else:
        raw = str(content)
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
        raw = raw.strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

    decoder = json.JSONDecoder()
    best = None
    idx = 0
    while True:
        start = raw.find("{", idx)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict) and ("story" in obj or "science_intro" in obj):
            return obj
        if isinstance(obj, dict):
            best = obj
        idx = end
    return best


def _passage_is_usable(text: str) -> bool:
    """Drop running headers / page numbers that are not real syllabus sentences."""
    t = re.sub(r"\s+", " ", text or "").strip()
    if len(t) < 90:
        return False
    letters = sum(ch.isalpha() for ch in t)
    digits = sum(ch.isdigit() for ch in t)
    if letters < 60:
        return False
    if digits >= 3 and letters / max(len(t), 1) < 0.55:
        return False
    return True


def _trim_context(text: str, limit: int = 3200) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[:limit].rsplit("\n", 1)[0] or t[:limit]


def _faiss_syllabus(topic: str, diagnostic: str = "") -> tuple[str, list]:
    """Real PDF chunks from the Voice Tutor FAISS index (Chroma often only has headers)."""
    try:
        from backend.common.retrieval import retrieve
        from backend.main import get_store

        store = get_store()
        hits = retrieve(store, f"{topic}. {diagnostic}".strip(), k=6)
    except Exception as exc:
        print(f"FAISS syllabus lookup skipped: {exc}", flush=True)
        return "", []

    context_parts = []
    sources = []
    for h in hits:
        text = (h.get("text") or "").strip()
        if not _passage_is_usable(text):
            continue
        passage = text[:700]
        page = h.get("page_start")
        label = h.get("topic") or h.get("source_file") or "syllabus"
        context_parts.append(f"--- {label} | Page: {page} ---\n{passage}")
        sources.append({
            "filename": h.get("source_file"),
            "chapter": h.get("topic"),
            "page": page,
            "similarity": f"{(h.get('score') or 0) * 100:.1f}%",
            "snippet": passage[:120].replace("\n", " ") + "...",
        })
        if len(sources) >= 4:
            break
    return "\n".join(context_parts), sources


def _story_is_thin(story):
    """True if SHOW is a stub instead of a 3-paragraph scene."""
    text = (story or "").strip()
    if not text:
        return True
    words = len(text.split())
    breaks = text.count("\n\n")
    return words < 160 or breaks < 2


def _fallback_chapter(topic, context, sources, error=None):
    """Build a syllabus-grounded chapter if the LLM returns nothing usable."""
    if error:
        print(f"Using textbook fallback after LLM failure: {error}", flush=True)
    passage = (context or "").strip()
    if not passage:
        passage = (
            f"The chapter on {topic} is in the Grade 10–11 science textbook. "
            "Open that lesson and read the definitions, examples, and any given formulae."
        )
    excerpt = passage[:1200].strip()
    return {
        "science_intro": {
            "concept_statement": topic,
            "explanation": excerpt[:800],
            "equations": [],
            "real_world_note": "Use the textbook extract in this lesson as the facts to remember.",
        },
        "story": (
            f"Nimali sits at a classroom desk in Colombo with her science book open to {topic}. "
            "Her teacher has asked the class to explain the idea in ordinary language before they copy notes.\n\n"
            f"{excerpt}\n\n"
            "She underlines the key terms, says the law out loud once, and writes a one-sentence summary "
            "so the textbook wording stays attached to the scene she just imagined."
        ),
        "key_definitions": [],
        "key_equations": [],
        "exam_bullets": [
            f"Recall the textbook statement of {topic}.",
            "Use syllabus terms rather than informal nicknames in answers.",
            "Link one everyday Sri Lankan example to the definition.",
        ],
        "quiz_topic": topic,
        "sources": sources or [],
    }


# Maps human-readable book names to ChromaDB filename metadata values
BOOK_FILENAME_MAP = {
    "Grade 10 - Part I":  "science G-10 P-I E",
    "Grade 10 - Part II": "science G-10 P-II E",
    "Grade 11 - Part I":  "science G-11 P-I E",
    "Grade 11 - Part II": "science G-11  P-II E",
}

class StoryEngine:
    def __init__(self):
        print("Initializing Story Engine...")

        # 1. Load Vector DB (The "Truth")
        # Build the Chroma client ourselves so langchain does not open Settings()
        # with persist_directory="./chroma" (empty store → default_tenant error).
        embeddings = get_embeddings()
        chroma_settings = Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=VECTOR_DB_DIR,
        )
        chroma_client = chromadb.PersistentClient(
            path=VECTOR_DB_DIR,
            settings=chroma_settings,
        )
        self.vectorstore = Chroma(
            client=chroma_client,
            embedding_function=embeddings,
        )

        # 2. Load ML Persona Model & Encoders (The "Vibe")
        try:
            with open(_PERSONA_MODEL_PATH, 'rb') as f:
                self.persona_model = pickle.load(f)
            with open(_ENCODER_PATH, 'rb') as f:
                self.encoders = pickle.load(f)
            self.has_model = True
            print("Successfully loaded ML Persona models.")
        except FileNotFoundError:
            print("Warning: persona_model.pkl or encoder.pkl not found.")
            self.has_model = False

        self.llm = get_llm()
        if LLM_PROVIDER == "ollama":
            try:
                self.llm.invoke('{"ok": true}')
            except Exception:
                pass

    def get_theme_for_student(self, interest, aspiration):
        """Predicts the story theme using the trained ML model."""
        if not self.has_model:
            return "Adventure"

        try:
            le_interest   = self.encoders['interest']
            le_aspiration = self.encoders['aspiration']
            le_theme      = self.encoders['theme']
            i_enc = le_interest.transform([interest])[0]
            a_enc = le_aspiration.transform([aspiration])[0]

            import pandas as pd
            X_pred = pd.DataFrame([[i_enc, a_enc]],
                                  columns=['Interest_Encoded', 'Aspiration_Encoded'])
            pred  = self.persona_model.predict(X_pred)
            return le_theme.inverse_transform(pred)[0]
        except Exception as e:
            # Graceful degradation — map common interests manually
            fallback = {
                "Cricket": "Sports Adventure",
                "Gaming": "Sci-Fi/Cyberpunk",
                "Music": "Drama/Inspirational",
                "Reading": "Mystery/Historical",
                "Art": "Creative/Fantasy",
                "Nature": "Exploration",
                "Robotics": "Futuristic/Technology",
            }
            return fallback.get(interest, "Adventure")

    def get_story_context(self, topic, book_name=None):
        """Searches Chroma, skips header-only pages, then fills gaps from FAISS."""
        print(f"\nSearching textbook database for: '{topic}'...", flush=True)

        search_kwargs = {"k": 10}
        if book_name and book_name in BOOK_FILENAME_MAP:
            filename = BOOK_FILENAME_MAP[book_name]
            search_kwargs["filter"] = {"filename": filename}

        results_with_scores = self.vectorstore.similarity_search_with_score(topic, **search_kwargs)

        context = ""
        sources = []
        for doc, score in results_with_scores:
            similarity_pct = max(0.0, min(100.0, (1.0 - (score / 2.0)) * 100.0))
            if similarity_pct < 55.0 and len(sources) >= 2:
                continue
            if len(sources) >= 4:
                break

            passage = (doc.page_content or "").strip()
            if not _passage_is_usable(passage):
                continue
            passage = passage[:700]
            meta = doc.metadata
            source_label = f"{meta.get('filename')} | Chapter: {meta.get('chapter')} | Page: {meta.get('page_number')}"
            context += f"\n--- {source_label} ---\n{passage}\n"
            sources.append({
                "filename": meta.get("filename"),
                "chapter": meta.get("chapter"),
                "page": meta.get("page_number"),
                "similarity": f"{similarity_pct:.1f}%",
                "snippet": passage[:120].replace("\n", " ") + "...",
            })

        extra_ctx, extra_src = _faiss_syllabus(topic)
        if extra_ctx:
            context = (extra_ctx + "\n" + context).strip()
            seen = {(s.get("filename"), s.get("page"), s.get("snippet")) for s in extra_src}
            merged = list(extra_src)
            for s in sources:
                key = (s.get("filename"), s.get("page"), s.get("snippet"))
                if key not in seen:
                    merged.append(s)
                    seen.add(key)
            sources = merged

        return _trim_context(context, 3200), sources[:6]

    def generate_question(self, topic):
        """Generates a standalone pre/post concept-check question."""
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
Generate a single, simple multiple-choice question to test a Grade 10-11 Sri Lankan student's understanding of: {topic}

Return ONLY raw JSON. No markdown. Format:
{{
    "question": "The question text",
    "options": {{"A": "...", "B": "...", "C": "..."}},
    "correct_answer": "A"
}}
"""
        )
        response = (prompt | self.llm).invoke({"topic": topic})
        try:
            return _parse_json_response(response.content)
        except Exception:
            return None

    def generate_chapter(self, student_theme, topic, diagnostic_query,
                         interest="General", aspiration="Student",
                         struggle_level="Medium", book_name=None,
                         pre_fetched_context=None, pre_fetched_sources=None):
        """Generates the full story chapter JSON.
        Accepts pre-fetched context so the DB is NOT queried twice.
        """
        # Use pre-fetched context if provided; otherwise fetch now
        if pre_fetched_context:
            syllabus_context = pre_fetched_context
            sources          = pre_fetched_sources or []
        else:
            syllabus_context, sources = self.get_story_context(
                f"{topic}. {diagnostic_query}", book_name
            )
        if LLM_PROVIDER == "groq":
            syllabus_context = _trim_context(syllabus_context, 2800)

        print(f"Generating story for '{topic}' with theme '{student_theme}'...", flush=True)

        prompt = PromptTemplate(
            input_variables=["theme","topic","diagnostic","context","interest","aspiration","struggle_level","length_nudge"],
            template="""
You are a science teacher writing a personalized lesson for a Sri Lankan Grade 10-11 student.
Topic: {topic}
Student struggle / focus: {diagnostic}
Theme: {theme}
Student: interested in {interest}, wants to be a {aspiration}, struggle level {struggle_level}.
Use ONLY these textbook passages. Do not invent syllabus facts:
{context}
{length_nudge}

TELL then SHOW. Never mix them.

science_intro = classroom explanation only. No characters, no plot.
- concept_statement: the law in textbook language
- explanation: 3-4 sentences on WHY it works, simple if struggle is High, more technical if Low
- equations: the key formula(s)
- real_world_note: one everyday sentence

story = a REAL SCENE, not a riddle and not a quiz. Give the student a Sri Lankan name.
Write EXACTLY 3 paragraphs, joined by \\n\\n. Each paragraph is 4-6 sentences (about 80-110 words). Total story at least 220 words.
1) Hook: place, people, and a problem tied to {interest}.
2) Action: the science happens through what they do or see. Show, do not ask "what do you need to do?"
3) Bridge: the student pauses and states the textbook law, terms, and numbers that explain the scene. More theory than plot here.

Rules:
- Connect the science to {interest}. Sri Lankan places and everyday objects are good.
- No sci-fi gadgets, no questions to the reader, no "Imagine you are..." lecture.
- Return one valid JSON object only. No markdown fences.

{{
    "science_intro": {{
        "concept_statement": "The exact law/concept in textbook language.",
        "explanation": "3-4 sentences on why it works.",
        "equations": ["F = ma"],
        "real_world_note": "One sentence."
    }},
    "story": "Hook paragraph...\\n\\nAction paragraph...\\n\\nBridge paragraph with the textbook law...",
    "key_definitions": [
        {{"term": "Textbook term", "definition": "Textbook-style definition"}},
        {{"term": "Second term", "definition": "Its definition"}}
    ],
    "key_equations": [
        {{"label": "What this represents", "equation": "F = ma"}}
    ],
    "exam_bullets": [
        "O/L exam point 1",
        "O/L exam point 2",
        "O/L exam point 3"
    ],
    "quiz_topic": "Short topic name (e.g. Moment of a Force)"
}}
"""
        )

        last_error = None
        last_data = None
        length_nudge = ""
        for _attempt in range(2):
            try:
                response = (prompt | self.llm).invoke({
                    "theme": student_theme, "topic": topic,
                    "diagnostic": diagnostic_query, "context": syllabus_context,
                    "interest": interest, "aspiration": aspiration,
                    "struggle_level": struggle_level,
                    "length_nudge": length_nudge,
                })
                data = _parse_json_response(response)
                if not data:
                    snippet = _llm_text(response)[:400].replace("\n", " ")
                    print(f"Could not parse JSON. Snippet: {snippet}", flush=True)
                    last_error = ValueError("unparseable JSON")
                    continue
                last_data = data
                # Local Llama often writes stubs; Groq is fast — keep the first parse.
                if (
                    LLM_PROVIDER != "groq"
                    and _story_is_thin(data.get("story", ""))
                    and _attempt == 0
                ):
                    print("Story too thin — rewriting as a full 3-paragraph scene...")
                    length_nudge = (
                        "PREVIOUS OUTPUT WAS TOO SHORT. Write a full scene now. "
                        "The story field MUST have three long paragraphs and two blank lines. "
                        "Do not write a one-paragraph thought experiment."
                    )
                    continue
                data["sources"] = sources
                return data
            except Exception as e:
                last_error = e
                print(f"Generation attempt failed: {e}", flush=True)
                err = str(e).lower()
                if "413" in err or "too large" in err or "tokens per minute" in err:
                    syllabus_context = _trim_context(syllabus_context, 1600)
        if last_data and (last_data.get("story") or "").strip():
            last_data["sources"] = sources
            return last_data
        if last_error:
            print(f"JSON parse error: {last_error}", flush=True)
        return _fallback_chapter(topic, syllabus_context, sources, last_error)

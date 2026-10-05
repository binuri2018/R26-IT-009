"""Narrative writer + nomic embeddings. Separate from Voice Tutor FAISS/OpenAI."""
import os

from dotenv import load_dotenv

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_COMPONENT_DIR, "..", "..", ".."))
load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=True)


def _resolve_vector_db() -> str:
    env = os.getenv("NARRATIVE_VECTOR_DB", "").strip()
    if env and os.path.isdir(env):
        return env
    local = os.path.join(_COMPONENT_DIR, "science_vector_db")
    if os.path.isdir(local):
        return local
    sibling = os.path.abspath(os.path.join(_REPO_ROOT, "..", "STEM TEXT", "science_vector_db"))
    if os.path.isdir(sibling):
        return sibling
    return local


VECTOR_DB_DIR = _resolve_vector_db()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3.5-lightning:free",
)


def get_embeddings():
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
        keep_alive=-1,
    )


def get_llm():
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing in .env")
        # JSON mode + non-thinking: Qwen 3.6 otherwise spends max_tokens on
        # hidden reasoning and returns an empty story (HTTP 502).
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=api_key,
            temperature=0.4,
            # Groq free/on-demand TPM is 8000: requested tokens = input + max_tokens.
            max_tokens=1800,
            timeout=120.0,
            reasoning_format="hidden",
            reasoning_effort="none",
        ).bind(response_format={"type": "json_object"})

    if LLM_PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing in .env")
        return ChatOpenAI(
            model=OPENROUTER_MODEL,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.5,
            default_headers={
                "HTTP-Referer": "https://github.com/binuri2018/STEM_LearnLK",
                "X-Title": "STEM Learn LK",
            },
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=OLLAMA_LLM_MODEL,
        temperature=0.65,
        format="json",
        base_url=OLLAMA_BASE_URL,
        keep_alive=-1,
        num_ctx=6144,
        num_predict=1800,
    )

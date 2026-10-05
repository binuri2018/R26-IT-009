"""
Application settings loaded from environment / .env.
"""
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project root = parent of `backend/`
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    resource_dir: Path = Path("Resource")
    data_dir: Path = Path("data")
    frontend_dir: Path = Path("frontend")

    # Ingest-time: "local" = sentence-transformers, "openai" = OpenAI embeddings (needs API key)
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # If HF/model HTTPS fails with CERTIFICATE_VERIFY_FAILED, set USE_CERTIFI_SSL=1 (needs certifi).
    use_certifi_ssl: bool = False
    # Dev-only: disables TLS verification for HTTPS (fixes some corporate/broken CA setups). Not for production.
    binuri_disable_ssl_verify: bool = False
    openai_embedding_model: str = "text-embedding-3-small"
    top_k: int = 5

    chunk_min_words: int = 200
    chunk_max_words: int = 400

    # OpenAI (preferred when key is set)
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_tts_model: str = "tts-1"
    openai_tts_voice: str = "nova"

    # Ollama fallback
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "llama3.1"
    # First local inference after pull/start can exceed 120s while the model loads into RAM.
    ollama_timeout_seconds: float = 600.0

    # Member 4 — Azure OpenAI / AI Foundry when M4_LLM_PROVIDER=azure.
    m4_llm_provider: Literal["ollama", "azure"] = "ollama"
    m4_ollama_model: str = "gemma4:cloud"

    faiss_index_name: str = "science_index.faiss"
    metadata_name: str = "science_metadata.jsonl"
    index_manifest_name: str = "index_manifest.json"
    image_faiss_index_name: str = "image_index.faiss"
    image_metadata_name: str = "image_metadata.jsonl"

    # Optional Member 4 cloud fallback, used only when M4_LLM_PROVIDER=azure.
    # Existing /api/ask, /api/transcribe, /api/tts continue to use direct OpenAI.
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-08-01-preview"
    azure_openai_deployment: str | None = None  # the DEPLOYMENT name in Azure Foundry, not model name

    # Member 4 — YouTube Data API v3 (used by /api/m4/youtube-suggest)
    youtube_api_key: str | None = None

    # Only these domains (or their subdomains) may decide a web-backed verdict.
    m4_web_allowed_domains: str = ""
    m4_confidence_model_path: Path | None = None
    m4_hybrid_dense_k: int = 20
    m4_hybrid_bm25_k: int = 20
    # Passages actually scored by the (CPU-only, no GPU) cross-encoder reranker.
    # Reranker cost scales ~linearly with this count (~2s/passage measured), so
    # keep it well below dense_k/bm25_k rather than reranking every fused candidate.
    m4_hybrid_fused_k: int = 8
    m4_hybrid_final_k: int = 5
    m4_rrf_k: int = 60
    m4_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    m4_reranker_revision: str | None = None
    m4_query_glossary_path: Path | None = None
    m4_retrieval_policy_path: Path | None = None
    m4_ocr_glossary_path: str = ""
    m4_ocr_max_dimension: int = 2400
    m4_ocr_confidence_low: float = 0.65
    m4_ocr_confidence_high: float = 0.85

    # Bounded concurrency for the per-claim verify/repair fan-out (backend/components/knowledge_maps/concurrency.py).
    m4_verify_max_workers: int = 5
    m4_web_search_timeout_seconds: float = 8.0

    # ── Adaptive Quiz component ──────────────────────────────────────────────
    # MongoDB connection string (Beanie + PyMongo async). When unset the quiz
    # component stays dormant and the rest of the app is unaffected.
    mongo_uri: str | None = None
    jwt_secret: str = "change-me-in-production"
    jwt_expire_days: int = 7
    # Shared secret gating the teacher-dashboard cross-student endpoints (no teacher accounts).
    teacher_key: str = ""
    # ML micro-service (learning-state prediction + T5 PDF question generation).
    ml_service_url: str = "http://127.0.0.1:8001"
    # Webcam expression model (Ultralytics YOLO checkpoint).
    emotion_model_path: Path = Path("backend/model/best.pt")
    # Learning-state RandomForest classifier (joblib).
    quiz_model_path: Path = Path("backend/components/adaptive_quiz/models/learning_state_model.pkl")
    emotion_predict_imgsz: int = 320
    emotion_yolo_conf: float = 0.10
    # YOLO valence boxes on a webcam face often peak ~0.18–0.50, not 0.45+.
    emotion_min_conf: float = 0.18
    # Slightly higher than happy so dim/noisy frames don't always fire a hint.
    emotion_negative_min_conf: float = 0.22
    # Optional JSON object: raw class name -> expression label.
    emotion_label_map: str = ""

    def resolved_resource_dir(self) -> Path:
        p = self.resource_dir
        return p if p.is_absolute() else (self.project_root / p)

    def resolved_data_dir(self) -> Path:
        p = self.data_dir
        return p if p.is_absolute() else (self.project_root / p)

    def resolved_frontend_dir(self) -> Path:
        p = self.frontend_dir
        return p if p.is_absolute() else (self.project_root / p)

    def resolved_emotion_model_path(self) -> Path:
        p = self.emotion_model_path
        return p if p.is_absolute() else (self.project_root / p)

    def resolved_quiz_model_path(self) -> Path:
        p = self.quiz_model_path
        return p if p.is_absolute() else (self.project_root / p)


settings = Settings()

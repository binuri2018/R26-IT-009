"""
Query-time embeddings must match the backend used during ingest.

- local: sentence-transformers (L2-normalized)
- openai: OpenAI embedding model (vectors L2-normalized for FAISS inner product)
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.common.config import settings

logger = logging.getLogger(__name__)

_st_model: SentenceTransformer | None = None
_runtime: dict[str, Any] | None = None


class EmbeddingLoadError(RuntimeError):
    """Local sentence-transformers model could not be loaded (often SSL or offline cache)."""


_ssl_unverified_patch_done = False


def _disable_ssl_verify_if_requested() -> None:
    """Optional dev workaround when HF/transformers HTTPS fails certificate verification."""
    global _ssl_unverified_patch_done
    if not settings.binuri_disable_ssl_verify or _ssl_unverified_patch_done:
        return
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: S324

    try:
        import httpx
        from huggingface_hub.utils._http import (
            async_hf_request_event_hook,
            async_hf_response_event_hook,
            hf_request_event_hook,
            set_async_client_factory,
            set_client_factory,
        )

        def insecure_hf_client_factory() -> httpx.Client:
            return httpx.Client(
                event_hooks={"request": [hf_request_event_hook]},
                follow_redirects=True,
                timeout=None,
                verify=False,
            )

        def insecure_hf_async_client_factory() -> httpx.AsyncClient:
            return httpx.AsyncClient(
                event_hooks={
                    "request": [async_hf_request_event_hook],
                    "response": [async_hf_response_event_hook],
                },
                follow_redirects=True,
                timeout=None,
                verify=False,
            )

        set_client_factory(insecure_hf_client_factory)
        set_async_client_factory(insecure_hf_async_client_factory)
    except Exception as exc:
        logger.warning("Could not patch Hugging Face Hub httpx client (verify=False): %s", exc)

    _ssl_unverified_patch_done = True
    logger.warning(
        "BINURI_DISABLE_SSL_VERIFY is enabled: TLS certificate verification is disabled for this process."
    )


def _apply_certifi_ssl_if_requested() -> None:
    if not settings.use_certifi_ssl:
        return
    try:
        import certifi

        bundle = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
        os.environ.setdefault("CURL_CA_BUNDLE", bundle)
    except ImportError:
        logger.warning(
            "USE_CERTIFI_SSL is set but certifi is not installed; run pip install certifi"
        )


def configure_embeddings(manifest: dict[str, Any] | None) -> None:
    """Call at API startup after loading the FAISS index manifest."""
    global _st_model, _runtime
    _st_model = None
    _runtime = manifest


def _get_runtime() -> dict[str, Any]:
    if _runtime is None:
        return {
            "embedding_provider": settings.embedding_provider,
            "embedding_model": (
                settings.openai_embedding_model
                if settings.embedding_provider == "openai"
                else settings.embedding_model
            ),
        }
    return _runtime


def get_st_model() -> SentenceTransformer:
    global _st_model
    rt = _get_runtime()
    if rt.get("embedding_provider") == "openai":
        raise RuntimeError(
            "SentenceTransformer must not be used when the index uses OpenAI embeddings."
        )
    if _st_model is None:
        _disable_ssl_verify_if_requested()
        _apply_certifi_ssl_if_requested()
        model_name = rt.get("embedding_model") or settings.embedding_model
        try:
            # CPU: Apple MPS segfaults when Member 4 encodes from parallel worker threads.
            _st_model = SentenceTransformer(model_name, device="cpu")
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if (
                "CERTIFICATE_VERIFY_FAILED" in msg
                or "certificate verify failed" in msg.lower()
                or "SSL" in type(exc).__name__
            ):
                raise EmbeddingLoadError(
                    "Could not load the local embedding model (HTTPS to Hugging Face failed SSL "
                    "verification). Options: set USE_CERTIFI_SSL=1 in .env and reinstall deps "
                    "(pip install certifi); set SSL_CERT_FILE to a valid CA bundle; download the "
                    "model to a local folder and set EMBEDDING_MODEL to that path; or use "
                    "EMBEDDING_PROVIDER=openai with OPENAI_API_KEY (re-run ingest after switching)."
                ) from exc
            if "Cannot send a request, as the client has been closed" in msg:
                raise EmbeddingLoadError(
                    "Embedding model download failed (Hugging Face Hub client closed — often follows "
                    "SSL or network errors). Fix TLS/certificates or use an offline model path / "
                    "EMBEDDING_PROVIDER=openai as described in the SSL error help above."
                ) from exc
            raise EmbeddingLoadError(
                f"Could not load the local embedding model ({model_name}). {msg}"
            ) from exc
    return _st_model


def warm_local_embedding_model() -> None:
    """Load SentenceTransformer once; raises EmbeddingLoadError if local provider cannot start."""
    rt = _get_runtime()
    if rt.get("embedding_provider") == "openai":
        return
    get_st_model()


def embed_texts_local(texts: list[str], batch_size: int = 32) -> np.ndarray:
    model = get_st_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 50,
    )
    return vectors.astype("float32", copy=False)


def embed_texts_openai(texts: list[str], batch_size: int = 100) -> np.ndarray:
    from openai import OpenAI

    key = settings.openai_api_key
    if not key or not str(key).strip():
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings.")
    client = OpenAI(api_key=key)
    rt = _get_runtime()
    model = rt.get("embedding_model") or settings.openai_embedding_model
    all_rows: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=chunk)
        ordered = sorted(resp.data, key=lambda d: d.index)
        for row in ordered:
            all_rows.append(row.embedding)
    arr = np.asarray(all_rows, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    arr = arr / norms
    return arr


def embed_texts_for_ingest(texts: list[str]) -> np.ndarray:
    """Uses current settings.embedding_provider (ingest script)."""
    if settings.embedding_provider == "openai":
        return embed_texts_openai(texts)
    return embed_texts_local(texts)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode many texts using the active runtime manifest (ingest compatibility)."""
    rt = _get_runtime()
    prov = rt.get("embedding_provider", settings.embedding_provider)
    if prov == "openai":
        return embed_texts_openai(texts, batch_size=min(100, max(32, batch_size)))
    return embed_texts_local(texts, batch_size=batch_size)


def embed_query(text: str) -> np.ndarray:
    v = embed_texts([text], batch_size=1)
    return v

"""MongoDB (Beanie + PyMongo async) connection for the Adaptive Quiz component.

Isolated from the rest of the app: if ``MONGO_URI`` is unset or the connection
fails, only quiz endpoints are affected — the RAG store and other components
continue to work.

Beanie 2.x requires ``pymongo.AsyncMongoClient``, not Motor. Passing a Motor
database into ``init_beanie`` raises ``MotorDatabase object is not callable``.
"""
from __future__ import annotations

import logging

from beanie import init_beanie
from pymongo import AsyncMongoClient

from backend.common.config import settings
from backend.components.adaptive_quiz.documents import ALL_DOCUMENTS

logger = logging.getLogger(__name__)

_client: AsyncMongoClient | None = None
_ready: bool = False
_init_error: str | None = None


async def init_quiz_db() -> None:
    """Connect and register Beanie models. Raises if MONGO_URI is missing."""
    global _client, _ready, _init_error
    if _ready:
        return
    if not settings.mongo_uri:
        _init_error = "MONGO_URI is not set"
        raise RuntimeError(_init_error)

    try:
        _client = AsyncMongoClient(settings.mongo_uri)
        db = _client.get_default_database()
        await init_beanie(database=db, document_models=ALL_DOCUMENTS)
    except Exception as exc:
        _init_error = str(exc)
        if _client is not None:
            await _client.close()
            _client = None
        raise
    _ready = True
    _init_error = None
    logger.info("Adaptive Quiz MongoDB connected (Beanie initialised)")


async def close_quiz_db() -> None:
    global _client, _ready
    if _client is not None:
        await _client.close()
        _client = None
    _ready = False


def quiz_db_ready() -> bool:
    return _ready


def quiz_db_error() -> str | None:
    return _init_error

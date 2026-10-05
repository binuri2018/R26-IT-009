"""Provider-neutral LLM client for Member 4 features."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx
from openai import APIStatusError, AuthenticationError, AzureOpenAI, BadRequestError, OpenAI

from backend.common.config import settings


_client: OpenAI | AzureOpenAI | None = None
_client_key: tuple[Any, ...] | None = None
_OLLAMA_USER_CHAR_LIMIT = 10_000
logger = logging.getLogger(__name__)

# Short names people type in .env → the tag actually installed by `ollama pull`.
_OLLAMA_MODEL_ALIASES = {
    "gemma:cloud": "gemma4:cloud",
}


_REASONING_BLOCK_RE = re.compile(
    r"<(?:think|reasoning)>.*?</(?:think|reasoning)>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _strip_reasoning(text: str) -> str:
    """Remove reasoning blocks that must never leak into parsed output."""
    return _REASONING_BLOCK_RE.sub("", text or "").strip()


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from plain, fenced, or lightly wrapped output."""
    cleaned = _strip_reasoning(text)
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.I)
    if fenced:
        cleaned = fenced.group(1).strip()

    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_azure_endpoint(url: str) -> str:
    u = (url or "").strip()
    if u.startswith("hhttps://"):
        u = "https://" + u[len("hhttps://") :]
    if u and not u.endswith("/"):
        u += "/"
    return u


def _azure_ready() -> bool:
    return bool(
        _normalize_azure_endpoint(settings.azure_openai_endpoint or "")
        and (settings.azure_openai_api_key or "").strip()
        and (settings.azure_openai_deployment or "").strip()
    )


def is_configured() -> bool:
    """Return whether the selected provider has its required configuration."""
    if settings.m4_llm_provider == "ollama":
        return bool(settings.ollama_base_url.strip() and settings.m4_ollama_model.strip())
    return _azure_ready()


def _client_and_model() -> tuple[OpenAI | AzureOpenAI, str]:
    """Return a cached client and model for the currently selected provider."""
    global _client, _client_key

    if settings.m4_llm_provider == "ollama":
        import os

        model = _resolve_ollama_model(settings.m4_ollama_model)
        base_url = settings.ollama_base_url.rstrip("/")
        api_key = (os.getenv("OLLAMA_API_KEY") or "").strip() or "ollama"
        key = ("ollama", base_url, model, settings.ollama_timeout_seconds, api_key)
        if not base_url or not model:
            raise RuntimeError("Ollama is not configured. Set OLLAMA_BASE_URL and M4_OLLAMA_MODEL.")
        if _client is None or _client_key != key:
            _client = OpenAI(
                base_url=f"{base_url}/v1",
                api_key=api_key,
                timeout=settings.ollama_timeout_seconds,
                max_retries=1,
            )
            _client_key = key
        return _client, model

    return _azure_client_and_model()


def _azure_client_and_model() -> tuple[OpenAI | AzureOpenAI, str]:
    global _client, _client_key
    endpoint = _normalize_azure_endpoint(settings.azure_openai_endpoint or "")
    deployment = str(settings.azure_openai_deployment or "").strip()
    key = (
        "azure",
        endpoint,
        settings.azure_openai_api_key,
        settings.azure_openai_api_version,
        deployment,
    )
    if not _azure_ready():
        raise RuntimeError(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT."
        )
    if _client is None or _client_key != key:
        # Azure AI Foundry / AI Services uses the OpenAI v1 base URL, not the
        # classic *.openai.azure.com deployment path.
        if "services.ai.azure.com" in endpoint.lower():
            _client = OpenAI(
                base_url=endpoint.rstrip("/") + "/openai/v1/",
                api_key=settings.azure_openai_api_key,
                timeout=120.0,
                max_retries=2,
            )
        else:
            _client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                timeout=120.0,
                max_retries=3,
            )
        _client_key = key
    return _client, deployment


def _canonical_model_name(name: str) -> str:
    return name.removesuffix(":latest")


def _resolve_ollama_model(name: str) -> str:
    """Map aliases such as gemma:cloud onto the pulled Ollama tag."""
    raw = (name or "").strip()
    return _OLLAMA_MODEL_ALIASES.get(raw.lower(), raw)


def ping() -> dict[str, Any]:
    """Check provider reachability without raising or loading an Ollama model."""
    provider = settings.m4_llm_provider
    model = (
        _resolve_ollama_model(settings.m4_ollama_model)
        if provider == "ollama"
        else settings.azure_openai_deployment
    )
    if not is_configured():
        return {
            "ok": False,
            "provider": provider,
            "model": model,
            "error": "selected LLM provider is not configured",
        }

    start = time.perf_counter()
    try:
        if provider == "ollama":
            url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
            response = httpx.get(url, timeout=5.0)
            response.raise_for_status()
            installed = [
                item.get("name", "") for item in response.json().get("models", [])
            ]
            wanted = _canonical_model_name(_resolve_ollama_model(settings.m4_ollama_model))
            found = any(
                _canonical_model_name(name) == wanted
                or _canonical_model_name(_resolve_ollama_model(name)) == wanted
                for name in installed
            )
            result: dict[str, Any] = {
                "ok": found,
                "provider": provider,
                "model": model,
                "latency_ms": int((time.perf_counter() - start) * 1000),
            }
            if not found:
                result["error"] = (
                    f"Model {model!r} is not installed. Run: ollama pull {model}"
                )
            return result

        client, selected_model = _client_and_model()
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": "Reply with the single word: pong"},
                {"role": "user", "content": "ping"},
            ],
            max_tokens=4,
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip() if response.choices else ""
        return {
            "ok": bool(text),
            "provider": provider,
            "model": model,
            "latency_ms": int((time.perf_counter() - start) * 1000),
        }
    except Exception as exc:
        return {
            "ok": False,
            "provider": provider,
            "model": model,
            "error": f"{type(exc).__name__}: {exc!s}",
        }


def _ollama_extras() -> dict[str, Any]:
    if settings.m4_llm_provider == "ollama":
        return {"extra_body": {"reasoning_effort": "none"}}
    return {}


def _prepare_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clamp combined Ollama user text while leaving system instructions intact."""
    copied = [dict(message) for message in messages]
    if settings.m4_llm_provider != "ollama":
        return copied

    remaining = _OLLAMA_USER_CHAR_LIMIT
    truncated = False
    for message in copied:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        if len(content) > remaining:
            message["content"] = content[:remaining]
            truncated = True
        remaining = max(0, remaining - len(message["content"]))
    if truncated:
        logger.warning(
            "Clamped Ollama user-message text to %d characters for context safety",
            _OLLAMA_USER_CHAR_LIMIT,
        )
    return copied


def _fallback_messages(
    messages: list[dict[str, Any]], function: dict[str, Any]
) -> list[dict[str, Any]]:
    schema = json.dumps(function["parameters"], ensure_ascii=False)
    instruction = (
        "\n\nReturn ONE raw JSON object validating against this JSON Schema. "
        "Do not use Markdown fences. Respect every enum, minItems, minimum, "
        f"and maximum constraint.\nSchema:\n{schema}"
    )
    copied = [dict(message) for message in messages]
    for index in range(len(copied) - 1, -1, -1):
        if copied[index].get("role") == "user" and isinstance(copied[index].get("content"), str):
            copied[index]["content"] = copied[index]["content"] + instruction
            break
    else:
        copied.append({"role": "user", "content": instruction.strip()})
    return copied


def _json_mode_call(
    client: OpenAI | AzureOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    function: dict[str, Any],
    temperature: float,
) -> dict[str, Any] | None:
    formats = [
        {
            "type": "json_schema",
            "json_schema": {
                "name": function["name"],
                "schema": function["parameters"],
            },
        },
        {"type": "json_object"},
    ]
    fallback_messages = _fallback_messages(messages, function)
    for response_format in formats:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=fallback_messages,
                response_format=response_format,
                temperature=temperature,
                **_ollama_extras(),
            )
        except BadRequestError:
            continue
        if not response.choices:
            continue
        parsed = _try_parse_json(response.choices[0].message.content or "")
        if parsed:
            return parsed
    return None


def chat_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: dict[str, Any] | str | None = None,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Return structured arguments, with JSON mode as a tool-call fallback."""
    if not tools:
        raise ValueError("At least one tool schema is required")
    client, model = _client_and_model()
    prepared_messages = _prepare_messages(messages)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=prepared_messages,
            tools=tools,
            tool_choice=tool_choice or "auto",
            temperature=temperature,
            **_ollama_extras(),
        )
    except BadRequestError:
        response = None

    if response and response.choices:
        message = response.choices[0].message
        if getattr(message, "tool_calls", None):
            parsed = _try_parse_json(message.tool_calls[0].function.arguments or "{}")
            if parsed:
                return parsed

    chosen_name = None
    if isinstance(tool_choice, dict):
        chosen_name = tool_choice.get("function", {}).get("name")
    function = next(
        (tool["function"] for tool in tools if tool["function"].get("name") == chosen_name),
        tools[0]["function"],
    )
    parsed = _json_mode_call(client, model, prepared_messages, function, temperature)
    if parsed:
        return parsed
    raise RuntimeError(f"Model did not return valid structured output for {function['name']}")


def _vision_completion(client: OpenAI | AzureOpenAI, model: str, prompt: str, image_b64: str, mime: str, temperature: float) -> str:
    extras = _ollama_extras() if settings.m4_llm_provider == "ollama" else {}
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                ],
            }
        ],
        temperature=temperature,
        **extras,
    )
    if not response.choices:
        return ""
    return _strip_reasoning(response.choices[0].message.content or "")


def vision_extract(
    prompt: str,
    image_b64: str,
    mime: str = "image/jpeg",
    temperature: float = 0.0,
) -> str:
    """Extract text from an image using the selected Member 4 LLM provider."""
    client, model = _client_and_model()
    try:
        return _vision_completion(client, model, prompt, image_b64, mime, temperature)
    except AuthenticationError as exc:
        if settings.m4_llm_provider == "ollama":
            raise RuntimeError(
                f"Ollama returned 401 for {model!r}. For cloud models run "
                f"`ollama signin` then `ollama pull {model}`."
            ) from exc
        raise RuntimeError(
            "Azure rejected the API key (401 Unauthorized). In Azure Portal open the "
            "resource → Keys and Endpoint, copy KEY 1 and the exact endpoint URL."
        ) from exc
    except APIStatusError as exc:
        label = "Ollama" if settings.m4_llm_provider == "ollama" else "Azure"
        raise RuntimeError(f"{label} vision request failed: {exc}") from exc
    except Exception as exc:
        msg = str(exc).lower()
        if settings.m4_llm_provider == "azure" and (
            "getaddrinfo" in msg or "name or service not known" in msg or "11001" in msg
        ):
            host = _normalize_azure_endpoint(settings.azure_openai_endpoint or "")
            raise RuntimeError(
                f"Azure endpoint hostname does not resolve: {host}. "
                "Copy the exact Endpoint from Azure Portal (Keys and Endpoint). "
                "Typical forms: https://YOUR-RESOURCE.openai.azure.com/ "
                "or https://YOUR-RESOURCE.services.ai.azure.com/"
            ) from exc
        if settings.m4_llm_provider == "ollama":
            raise RuntimeError(f"Ollama vision request failed for {model!r}: {exc}") from exc
        raise


def chat_json(system: str, user: str, temperature: float = 0.1) -> dict[str, Any]:
    """Return a parsed JSON object, or an empty object for malformed output."""
    client, model = _client_and_model()
    response = client.chat.completions.create(
        model=model,
        messages=_prepare_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        ),
        response_format={"type": "json_object"},
        temperature=temperature,
        **_ollama_extras(),
    )
    if not response.choices:
        return {}
    return _try_parse_json(response.choices[0].message.content or "") or {}


def chat_plain(
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 16,
) -> str:
    """Return reasoning-free plain text from the selected provider."""
    client, model = _client_and_model()
    response = client.chat.completions.create(
        model=model,
        messages=_prepare_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        **_ollama_extras(),
    )
    if not response.choices:
        return ""
    return _strip_reasoning(response.choices[0].message.content or "")

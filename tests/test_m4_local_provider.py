from __future__ import annotations

import unittest
import base64
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.common.config import Settings


class Member4ConfigurationTests(unittest.TestCase):
    def test_member4_defaults_to_local_gemma(self) -> None:
        settings = Settings(_env_file=None)

        self.assertEqual(settings.m4_llm_provider, "ollama")
        self.assertEqual(settings.m4_ollama_model, "gemma4:cloud")


class LlmClientParsingTests(unittest.TestCase):
    def test_reasoning_and_fences_are_removed_before_json_parsing(self) -> None:
        from backend.components.knowledge_maps import llm_client

        raw = '<think>hidden chain</think>\n```json\n{"claims": ["ATP"]}\n```'

        self.assertEqual(llm_client._try_parse_json(raw), {"claims": ["ATP"]})


class LlmClientProviderTests(unittest.TestCase):
    def test_ollama_client_uses_openai_compatible_endpoint_and_model(self) -> None:
        from backend.common.config import settings
        from backend.components.knowledge_maps import llm_client

        old_provider = settings.m4_llm_provider
        old_url = settings.ollama_base_url
        old_model = settings.m4_ollama_model
        self.addCleanup(setattr, settings, "m4_llm_provider", old_provider)
        self.addCleanup(setattr, settings, "ollama_base_url", old_url)
        self.addCleanup(setattr, settings, "m4_ollama_model", old_model)
        settings.m4_llm_provider = "ollama"
        settings.ollama_base_url = "http://ollama.test/"
        settings.m4_ollama_model = "gemma4:cloud"
        llm_client._client = None
        llm_client._client_key = None

        fake_client = MagicMock()
        with patch.object(llm_client, "OpenAI", return_value=fake_client) as constructor:
            client, model = llm_client._client_and_model()

        self.assertIs(client, fake_client)
        self.assertEqual(model, "gemma4:cloud")
        constructor.assert_called_once_with(
            base_url="http://ollama.test/v1",
            api_key="ollama",
            timeout=settings.ollama_timeout_seconds,
            max_retries=1,
        )

    def test_gemma_cloud_alias_resolves_to_pulled_tag(self) -> None:
        from backend.components.knowledge_maps import llm_client

        self.assertEqual(llm_client._resolve_ollama_model("gemma:cloud"), "gemma4:cloud")
        self.assertEqual(llm_client._resolve_ollama_model("gemma4:cloud"), "gemma4:cloud")

    def test_tool_call_falls_back_to_json_schema_when_model_returns_prose(self) -> None:
        from backend.common.config import settings
        from backend.components.knowledge_maps import llm_client

        tool = {
            "type": "function",
            "function": {
                "name": "submit_claims",
                "description": "Return claims",
                "parameters": {
                    "type": "object",
                    "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
                    "required": ["claims"],
                },
            },
        }
        prose = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Here are the claims", tool_calls=None))]
        )
        json_reply = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"claims":["Cells respire"]}', tool_calls=None))]
        )
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [prose, json_reply]

        with patch.object(llm_client, "_client_and_model", return_value=(fake_client, "gemma4:cloud")):
            result = llm_client.chat_with_tools(
                [{"role": "user", "content": "Extract claims"}],
                [tool],
                tool_choice={"type": "function", "function": {"name": "submit_claims"}},
            )

        self.assertEqual(result, {"claims": ["Cells respire"]})
        second_request = fake_client.chat.completions.create.call_args_list[1].kwargs
        self.assertEqual(second_request["response_format"]["type"], "json_schema")
        self.assertEqual(
            second_request["response_format"]["json_schema"]["schema"],
            tool["function"]["parameters"],
        )
        self.assertEqual(settings.m4_llm_provider, "ollama")

    def test_plain_and_json_calls_return_clean_content(self) -> None:
        from backend.components.knowledge_maps import llm_client

        plain_reply = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="<think>hidden</think>correct"))]
        )
        json_reply = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='```json\n{"cleaned":"ATP"}\n```'))]
        )
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [plain_reply, json_reply]

        with patch.object(llm_client, "_client_and_model", return_value=(fake_client, "gemma4:cloud")):
            plain = llm_client.chat_plain("judge", "claim", max_tokens=8)
            structured = llm_client.chat_json("clean", "transcript")

        self.assertEqual(plain, "correct")
        self.assertEqual(structured, {"cleaned": "ATP"})

    def test_long_ollama_user_input_is_clamped_but_system_prompt_is_preserved(self) -> None:
        from backend.components.knowledge_maps import llm_client

        prepared = llm_client._prepare_messages(
            [
                {"role": "system", "content": "system rules"},
                {"role": "user", "content": "x" * 12_000},
            ]
        )

        self.assertEqual(prepared[0]["content"], "system rules")
        self.assertEqual(len(prepared[1]["content"]), 10_000)
        self.assertEqual(prepared[1]["content"], "x" * 10_000)

    def test_ollama_ping_reports_installed_model_without_loading_it(self) -> None:
        from backend.components.knowledge_maps import llm_client

        response = MagicMock()
        response.json.return_value = {"models": [{"name": "gemma4:cloud"}]}
        with patch.object(llm_client.httpx, "get", return_value=response) as get:
            result = llm_client.ping()

        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "ollama")
        self.assertEqual(result["model"], "gemma4:cloud")
        get.assert_called_once_with("http://127.0.0.1:11434/api/tags", timeout=5.0)
        response.raise_for_status.assert_called_once_with()

    def test_switching_to_azure_rebuilds_client_with_existing_timeout(self) -> None:
        from backend.common.config import settings
        from backend.components.knowledge_maps import llm_client

        original = {
            "m4_llm_provider": settings.m4_llm_provider,
            "azure_openai_endpoint": settings.azure_openai_endpoint,
            "azure_openai_api_key": settings.azure_openai_api_key,
            "azure_openai_deployment": settings.azure_openai_deployment,
        }
        for name, value in original.items():
            self.addCleanup(setattr, settings, name, value)
        settings.m4_llm_provider = "azure"
        settings.azure_openai_endpoint = "https://azure.test"
        settings.azure_openai_api_key = "test-key"
        settings.azure_openai_deployment = "gpt-4o"
        llm_client._client = MagicMock(name="old-ollama-client")
        llm_client._client_key = ("ollama",)

        azure_client = MagicMock(name="azure-client")
        with patch.object(llm_client, "AzureOpenAI", return_value=azure_client) as constructor:
            client, model = llm_client._client_and_model()

        self.assertIs(client, azure_client)
        self.assertEqual(model, "gpt-4o")
        constructor.assert_called_once_with(
            azure_endpoint="https://azure.test/",
            api_key="test-key",
            api_version=settings.azure_openai_api_version,
            timeout=120.0,
            max_retries=3,
        )


class GemmaVisionOcrTests(unittest.TestCase):
    def test_ocr_sends_the_original_image_to_provider_neutral_vision(self) -> None:
        from io import BytesIO
        from PIL import Image
        from backend.components.knowledge_maps import llm_client, ocr

        buf = BytesIO()
        Image.new("RGB", (20, 20), "white").save(buf, format="PNG")
        image = buf.getvalue()
        with patch.object(llm_client, "vision_extract", return_value="Cell membrane") as vision:
            text = ocr.extract_text(image, "image/png")

        self.assertEqual(text, "Cell membrane")
        prompt, encoded = vision.call_args.args
        self.assertIn("Extract", prompt)
        self.assertEqual(base64.b64decode(encoded), image)
        self.assertEqual(vision.call_args.kwargs["mime"], "image/png")

    def test_vision_client_builds_an_openai_compatible_data_url(self) -> None:
        from backend.components.knowledge_maps import llm_client

        reply = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="<think>x</think>ATP"))]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = reply

        with patch.object(llm_client, "_client_and_model", return_value=(client, "gemma4:cloud")):
            result = llm_client.vision_extract("Extract text", "YWJj", mime="image/png")

        self.assertEqual(result, "ATP")
        request = client.chat.completions.create.call_args.kwargs
        image_part = request["messages"][0]["content"][1]
        self.assertEqual(image_part["image_url"]["url"], "data:image/png;base64,YWJj")
        self.assertEqual(request["extra_body"], {"reasoning_effort": "none"})

    def test_ocr_rejects_empty_unsupported_and_oversized_uploads(self) -> None:
        from backend.components.knowledge_maps import ocr

        with self.assertRaisesRegex(ValueError, "Empty image"):
            ocr.extract_text(b"", "image/png")
        with self.assertRaisesRegex(ValueError, "Unsupported image type"):
            ocr.extract_text(b"pdf", "application/pdf")
        with self.assertRaisesRegex(ValueError, "Image too large"):
            ocr.extract_text(b"x" * (ocr.MAX_IMAGE_BYTES + 1), "image/png")


class Member4OutputNormalizationTests(unittest.TestCase):
    def test_mindmap_clamps_importance_and_removes_dangling_edges(self) -> None:
        from backend.components.knowledge_maps.synthesis import _sanitize_mindmap

        raw = {
            "nodes": [
                {"id": "cell", "label": "Cell", "group": "topic", "importance": 1.7},
                {"id": "atp", "label": "ATP", "group": "energy", "importance": -0.2},
            ],
            "edges": [
                {"source": "cell", "target": "atp", "relation": "uses"},
                {"source": "cell", "target": "missing", "relation": "contains"},
            ],
        }

        result = _sanitize_mindmap(raw)

        self.assertEqual([node["importance"] for node in result["nodes"]], [1.0, 0.0])
        self.assertEqual(result["edges"], [raw["edges"][0]])

    def test_verdict_normalization_maps_case_synonyms_and_unknown_values(self) -> None:
        from backend.components.knowledge_maps.verification import _normalize_verdict

        self.assertEqual(_normalize_verdict("Correct"), "correct")
        self.assertEqual(_normalize_verdict("wrong"), "incorrect")
        self.assertEqual(_normalize_verdict("partially correct"), "incomplete")
        self.assertIsNone(_normalize_verdict("maybe"))


class Member4RouteTests(unittest.TestCase):
    def test_health_uses_provider_neutral_fields(self) -> None:
        from backend.components.knowledge_maps import router as routes

        with (
            patch.object(routes.llm_client, "is_configured", return_value=True),
            patch.object(
                routes.llm_client,
                "ping",
                return_value={
                    "ok": True,
                    "provider": "ollama",
                    "model": "gemma4:cloud",
                    "latency_ms": 4,
                },
            ),
        ):
            payload = routes.health().model_dump()

        self.assertEqual(payload["llm_provider"], "ollama")
        self.assertEqual(payload["llm_model"], "gemma4:cloud")
        self.assertTrue(payload["llm_ok"])
        self.assertNotIn("azure_ok", payload)

    def test_ollama_connection_errors_include_start_and_pull_hint(self) -> None:
        import httpx
        import openai

        from backend.components.knowledge_maps import router as routes

        error = openai.APIConnectionError(request=httpx.Request("POST", "http://ollama.test"))
        mapped = routes._map_error(error)

        self.assertEqual(mapped.status_code, 502)
        self.assertIn("ollama serve", mapped.detail)
        self.assertIn("ollama pull gemma4:cl", mapped.detail)


if __name__ == "__main__":
    unittest.main()

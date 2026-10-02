from __future__ import annotations

import json

import httpx
import pytest

from audiobooks.config import Settings
from audiobooks.errors import LLMError, LLMResponseError
from audiobooks.llm.client import OpenAICompatClient
from audiobooks.llm.json_repair import JSONExtractionError, extract_json_object
from audiobooks.llm.openai_provider import OpenAICompatibleProvider, parse_chunk_analysis
from audiobooks.llm.prompts import build_user_prompt
from audiobooks.llm.schemas import ChunkRequest, SpanInput

REQUEST = ChunkRequest(
    book_title="T",
    chapter_title="C",
    language="ru",
    spans=[SpanInput(0, "narration", "Анна вошла."), SpanInput(1, "dialogue", "Привет!")],
)
GOOD = {
    "characters": [{"id": "anna", "name": "Анна", "gender": "female", "age_group": "adult"}],
    "segments": [
        {"id": 0, "type": "narration", "speaker": "narrator", "emotion": "neutral"},
        {"id": 1, "type": "dialogue", "speaker": "anna", "emotion": "happy"},
    ],
}


def test_extract_plain_json() -> None:
    assert extract_json_object(json.dumps(GOOD)) == GOOD


def test_extract_from_fence_with_think_and_trailing_comma() -> None:
    raw = '<think>let me see</think>\nHere you go:\n```json\n{"a": [1, 2,],}\n```'
    assert extract_json_object(raw) == {"a": [1, 2]}


def test_extract_object_embedded_in_prose() -> None:
    raw = 'Sure! {"segments": [], "note": "brace } in string"} Hope this helps.'
    assert extract_json_object(raw)["note"] == "brace } in string"


@pytest.mark.parametrize("raw", ["", "no json here", "[1, 2, 3]", '{"unterminated": '])
def test_extract_failures(raw: str) -> None:
    with pytest.raises(JSONExtractionError):
        extract_json_object(raw)


def test_parse_normalizes_values() -> None:
    data = json.loads(json.dumps(GOOD))
    data["segments"][1]["emotion"] = "quiet"  # synonym -> whisper
    data["segments"][1]["type"] = "Dialog"
    data["characters"][0]["gender"] = "Female"
    analysis = parse_chunk_analysis(json.dumps(data), REQUEST)
    assert analysis.segments[1].emotion == "whisper"
    assert analysis.segments[1].type == "dialogue"
    assert analysis.characters[0].gender == "female"


def test_parse_rejects_wrong_schema() -> None:
    with pytest.raises(LLMResponseError):
        parse_chunk_analysis('{"characters": [], "segments": "nope"}', REQUEST)


def test_parse_rejects_mostly_missing_spans() -> None:
    data = {"characters": [], "segments": [GOOD["segments"][0]]}  # 1 of 2 = 50% missing
    with pytest.raises(LLMResponseError):
        parse_chunk_analysis(json.dumps(data), REQUEST)


def test_parse_drops_unknown_and_duplicate_ids() -> None:
    data = json.loads(json.dumps(GOOD))
    data["segments"].append({"id": 99, "type": "narration", "speaker": "x", "emotion": "neutral"})
    data["segments"].append(dict(GOOD["segments"][0], emotion="sad"))
    analysis = parse_chunk_analysis(json.dumps(data), REQUEST)
    assert [s.id for s in analysis.segments] == [0, 1]
    assert analysis.segments[0].emotion == "neutral"


def test_prompt_contains_spans_and_known_characters() -> None:
    text = build_user_prompt(REQUEST)
    assert '"text": "Привет!"' in text
    assert "KNOWN CHARACTERS: none yet" in text


def _provider(responses: list[str], settings: Settings) -> tuple[OpenAICompatibleProvider, list]:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        content = responses[min(len(seen) - 1, len(responses) - 1)]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
        )

    client = OpenAICompatClient(
        "http://llm.test/v1", "ornith", transport=httpx.MockTransport(handler)
    )
    return OpenAICompatibleProvider(settings, client=client), seen


def test_provider_retries_malformed_json_then_succeeds(settings: Settings) -> None:
    provider, seen = _provider(["not json at all", json.dumps(GOOD)], settings)
    analysis = provider.analyze_chunk(REQUEST)
    assert analysis.segments[1].speaker == "anna"
    assert len(seen) == 2
    # the retry tells the model what was wrong and keeps the original task
    assert seen[1]["messages"][-1]["role"] == "user"
    assert "could not be used" in seen[1]["messages"][-1]["content"]
    assert seen[0]["response_format"]["type"] == "json_schema"
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_provider_fails_clearly_after_retries(settings: Settings) -> None:
    provider, seen = _provider(["garbage"], settings)
    with pytest.raises(LLMResponseError, match="failed to produce valid analysis"):
        provider.analyze_chunk(REQUEST)
    assert len(seen) == 1 + settings.llama_max_retries


def test_connection_error_is_actionable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = OpenAICompatClient("http://llm.test/v1", "m", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="llama-server running"):
        client.chat([{"role": "user", "content": "hi"}])

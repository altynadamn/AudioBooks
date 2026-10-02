"""LLM provider backed by an OpenAI-compatible server (default: llama-server + Ornith)."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from audiobooks.config import Settings
from audiobooks.errors import LLMResponseError
from audiobooks.llm.client import OpenAICompatClient
from audiobooks.llm.json_repair import JSONExtractionError, extract_json_object
from audiobooks.llm.prompts import RETRY_PROMPT, SYSTEM_PROMPT, build_user_prompt
from audiobooks.llm.schemas import CHUNK_ANALYSIS_JSON_SCHEMA, ChunkAnalysis, ChunkRequest

log = logging.getLogger(__name__)

# A response that leaves more than this share of spans unlabeled is rejected and retried.
MAX_MISSING_RATIO = 0.2


def parse_chunk_analysis(raw: str, request: ChunkRequest) -> ChunkAnalysis:
    """Extract, validate and sanity-check a chunk analysis. Raises ``LLMResponseError``."""
    try:
        data = extract_json_object(raw)
    except JSONExtractionError as exc:
        raise LLMResponseError(f"malformed JSON: {exc}") from exc
    try:
        analysis = ChunkAnalysis.model_validate(data)
    except ValidationError as exc:
        raise LLMResponseError(
            f"schema validation failed: {exc.error_count()} error(s): "
            f"{exc.errors()[0]['msg']} at {exc.errors()[0]['loc']}"
        ) from exc

    expected = {s.id for s in request.spans}
    labels = {}
    for label in analysis.segments:
        if label.id in expected and label.id not in labels:
            labels[label.id] = label
    missing = expected - labels.keys()
    if expected and len(missing) / len(expected) > MAX_MISSING_RATIO:
        raise LLMResponseError(f"response labels only {len(labels)}/{len(expected)} spans")
    analysis.segments = sorted(labels.values(), key=lambda s: s.id)
    return analysis


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(self, settings: Settings, client: OpenAICompatClient | None = None) -> None:
        self.settings = settings
        self.client = client or OpenAICompatClient(
            settings.llama_base_url,
            settings.llama_model,
            api_key=settings.llama_api_key,
            timeout=settings.llama_timeout,
            disable_thinking=settings.llama_disable_thinking,
        )

    def health(self) -> bool:
        return self.client.health()

    def close(self) -> None:
        self.client.close()

    def analyze_chunk(self, request: ChunkRequest) -> ChunkAnalysis:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(request)},
        ]
        schema = CHUNK_ANALYSIS_JSON_SCHEMA if self.settings.llama_json_schema else None
        attempts = 1 + max(0, self.settings.llama_max_retries)
        last_error: LLMResponseError | None = None
        for attempt in range(1, attempts + 1):
            raw = self.client.chat(
                messages,
                temperature=self.settings.llama_temperature if attempt == 1 else 0.0,
                max_tokens=self.settings.llama_max_tokens,
                json_schema=schema,
            )
            try:
                return parse_chunk_analysis(raw, request)
            except LLMResponseError as exc:
                last_error = exc
                log.warning("LLM response rejected (attempt %d/%d): %s", attempt, attempts, exc)
                messages = [
                    *messages[:2],
                    {"role": "assistant", "content": raw[:4000]},
                    {
                        "role": "user",
                        "content": RETRY_PROMPT.format(
                            error=exc, ids=", ".join(str(s.id) for s in request.spans)
                        ),
                    },
                ]
        raise LLMResponseError(f"LLM failed to produce valid analysis: {last_error}")

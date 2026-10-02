"""LLM provider interface. The rest of the app depends only on this protocol."""

from __future__ import annotations

from typing import Protocol

from audiobooks.llm.schemas import ChunkAnalysis, ChunkRequest


class LLMProvider(Protocol):
    name: str

    def analyze_chunk(self, request: ChunkRequest) -> ChunkAnalysis:
        """Label every span of the chunk and report observed characters.

        Implementations must return a validated ``ChunkAnalysis`` or raise
        ``LLMError``/``LLMResponseError``; they must never return partial garbage.
        """
        ...

    def health(self) -> bool: ...

    def close(self) -> None: ...

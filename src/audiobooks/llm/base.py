"""LLM provider interface. The rest of the app depends only on this protocol."""

from __future__ import annotations

from typing import Protocol

from audiobooks.llm.schemas import CharacterCard, ChunkAnalysis, ChunkRequest, PairVerdict


class LLMProvider(Protocol):
    name: str

    def analyze_chunk(self, request: ChunkRequest) -> ChunkAnalysis:
        """Label every span of the chunk and report observed characters.

        Implementations must return a validated ``ChunkAnalysis`` or raise
        ``LLMError``/``LLMResponseError``; they must never return partial garbage.
        """
        ...

    def same_person(self, pairs: list[tuple[CharacterCard, CharacterCard]]) -> list[PairVerdict]:
        """For each candidate pair of cast entries: are they the same person?"""
        ...

    def health(self) -> bool: ...

    def close(self) -> None: ...

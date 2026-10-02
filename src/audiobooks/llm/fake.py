"""Deterministic stand-in LLM for tests and offline development.

It is NOT an analysis model: it keeps the segmenter's type hints, attributes dialogue only
when a speech tag explicitly names a configured character, and otherwise says "unknown".
"""

from __future__ import annotations

import re

from audiobooks.llm.schemas import (
    CharacterObservation,
    ChunkAnalysis,
    ChunkRequest,
    SpanLabel,
)


class FakeLLMProvider:
    name = "fake"

    def __init__(self, characters: list[CharacterObservation] | None = None) -> None:
        self.characters = characters or []
        self.calls: list[ChunkRequest] = []

    def health(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def _speaker_in(self, text: str) -> str | None:
        for ch in self.characters:
            for name in [ch.name, *ch.aliases]:
                if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
                    return ch.id
        return None

    def analyze_chunk(self, request: ChunkRequest) -> ChunkAnalysis:
        self.calls.append(request)
        labels: list[SpanLabel] = []
        spans = request.spans
        seen: set[str] = set()
        for i, span in enumerate(spans):
            if span.hint == "dialogue":
                # look at the following narration span for a speech tag ("— сказала Анна")
                tag = spans[i + 1].text if i + 1 < len(spans) else ""
                speaker = self._speaker_in(tag) or "unknown"
                if speaker != "unknown":
                    seen.add(speaker)
                labels.append(SpanLabel(id=span.id, type="dialogue", speaker=speaker))
            else:
                labels.append(SpanLabel(id=span.id, type="narration", speaker="narrator"))
        observed = [c for c in self.characters if c.id in seen]
        return ChunkAnalysis(characters=observed, segments=labels)

"""Deterministic stand-in LLM for tests and offline development.

It is NOT an analysis model: it keeps the segmenter's type hints, attributes dialogue only
when a speech tag explicitly names a configured character, and otherwise says "unknown".
"""

from __future__ import annotations

import re

from audiobooks.llm.schemas import (
    CharacterCard,
    CharacterObservation,
    ChunkAnalysis,
    ChunkRequest,
    PairVerdict,
    SpanLabel,
)


class FakeLLMProvider:
    name = "fake"

    def __init__(
        self,
        characters: list[CharacterObservation] | None = None,
        same: set[frozenset[str]] | None = None,
    ) -> None:
        self.characters = characters or []
        self.same = same or set()  # id pairs the fake considers the same person
        self.calls: list[ChunkRequest] = []
        self.duplicate_checks: list[list[tuple[CharacterCard, CharacterCard]]] = []

    def same_person(self, pairs: list[tuple[CharacterCard, CharacterCard]]) -> list[PairVerdict]:
        self.duplicate_checks.append(pairs)
        return [
            PairVerdict(pair=n, same_person=frozenset({a.id, b.id}) in self.same)
            for n, (a, b) in enumerate(pairs)
        ]

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
        # like a real model, report characters that speak or are mentioned in the chunk
        text = " ".join(s.text for s in spans)
        mentioned = {
            c.id
            for c in self.characters
            if any(re.search(rf"{re.escape(n)}", text, re.IGNORECASE) for n in [c.name, *c.aliases])
        }
        observed = [c for c in self.characters if c.id in seen | mentioned]
        return ChunkAnalysis(characters=observed, segments=labels)

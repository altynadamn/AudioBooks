"""Deterministic character-to-voice assignment and per-segment voice resolution."""

from __future__ import annotations

import logging
from collections import Counter

from audiobooks.models.character import AgeGroup, Character, Gender
from audiobooks.models.job import GenerationMode
from audiobooks.models.script import NARRATOR, AudiobookSegment, SegmentType
from audiobooks.models.voice import Voice
from audiobooks.voices.library import VoiceLibrary
from audiobooks.voices.registry import CharacterRegistry

log = logging.getLogger(__name__)

_AGE_ORDER = [AgeGroup.CHILD, AgeGroup.TEEN, AgeGroup.YOUNG_ADULT, AgeGroup.ADULT, AgeGroup.ELDERLY]


def _age_distance(a: AgeGroup, b: AgeGroup) -> int:
    if AgeGroup.UNKNOWN in (a, b):
        return 1
    return abs(_AGE_ORDER.index(a) - _AGE_ORDER.index(b))


def score_voice(character: Character, voice: Voice, usage: int) -> float:
    score = 0.0
    if Gender.UNKNOWN not in (character.gender, voice.gender):
        score += 4.0 if character.gender == voice.gender else -6.0
    score -= 0.75 * _age_distance(character.age_group, voice.age_group)
    score -= 2.5 * usage  # spread characters across voices
    return score


class VoiceCaster:
    """Assigns library voices to registry characters.

    Assignments are persistent: a character that already has a voice keeps it. New
    characters are cast in order of how much they speak, so main characters get the most
    distinct voices. Ties are broken by voice id, which makes casting reproducible.
    """

    def __init__(self, library: VoiceLibrary, narrator_voice_id: str) -> None:
        self.library = library
        self.narrator_voice_id = narrator_voice_id

    def cast(self, registry: CharacterRegistry) -> dict[str, str]:
        """Assign voices to characters that have none. Returns {character_id: voice_id}."""
        pool = [v for v in self.library.available() if "character" in v.roles]
        non_narrator = [v for v in pool if v.id != self.narrator_voice_id]
        if non_narrator:
            pool = non_narrator
        usage: Counter[str] = Counter(
            c.voice_id for c in registry.characters.values() if c.voice_id
        )
        assigned: dict[str, str] = {}
        pending = [c for c in registry.characters.values() if not c.voice_id]
        for character in sorted(pending, key=lambda c: (-c.line_count, c.first_chapter, c.id)):
            if not pool:
                character.voice_id = self.narrator_voice_id
            else:
                best = max(pool, key=lambda v: (score_voice(character, v, usage[v.id]), v.id))
                character.voice_id = best.id
            usage[character.voice_id] += 1
            assigned[character.id] = character.voice_id
        if assigned and not pool:
            log.warning(
                "no character voices available; all characters use the narrator voice. "
                "Add reference voices or enable built-in voices (see docs/VOICE_SYSTEM.md)."
            )
        for character in registry.characters.values():
            if character.voice_id and character.voice_id not in self.library:
                log.warning(
                    "character %s uses voice %r which is missing from the library",
                    character.id, character.voice_id,
                )  # fmt: skip
        return assigned


def resolve_voice(
    segment: AudiobookSegment,
    *,
    mode: GenerationMode,
    registry: CharacterRegistry,
    narrator_voice_id: str,
    unknown_voice_id: str = "",
) -> str:
    """Voice id for one segment under the given generation mode."""
    if mode is GenerationMode.SIMPLE or segment.type is SegmentType.NARRATION:
        return narrator_voice_id
    if segment.speaker == NARRATOR:
        return narrator_voice_id
    character = registry.get(segment.speaker)
    if character is not None and character.voice_id:
        return character.voice_id
    return unknown_voice_id or narrator_voice_id

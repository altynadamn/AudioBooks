"""Persistent per-book Character Registry.

Guarantees:
* a character keeps the same id (and therefore the same voice) for the whole book;
* aliases are merged only on strong evidence: the LLM reused a known id, said
  ``same_as``, or a name/alias matches exactly one known character;
* uncertain matches are kept as separate characters linked via ``possibly_same_as``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from audiobooks.llm.schemas import CharacterObservation, KnownCharacter
from audiobooks.models.character import AgeGroup, Character, CharacterRegistryData, Gender
from audiobooks.models.script import NARRATOR, UNKNOWN_SPEAKER
from audiobooks.utils import read_json, slugify, write_json

log = logging.getLogger(__name__)

RESERVED_IDS = {NARRATOR, UNKNOWN_SPEAKER}


def normalize_name(name: str) -> str:
    name = name.casefold().replace("ё", "е")
    return re.sub(r"[^\w]+", " ", name).strip()


class CharacterRegistry:
    def __init__(self, data: CharacterRegistryData) -> None:
        self.data = data

    # ---------------------------------------------------------------- persistence
    @classmethod
    def load(cls, path: Path, book_id: str) -> CharacterRegistry:
        if path.is_file():
            data = CharacterRegistryData.model_validate(read_json(path))
            if data.book_id != book_id:
                raise ValueError(f"{path} belongs to book {data.book_id!r}, not {book_id!r}")
            return cls(data)
        return cls(CharacterRegistryData(book_id=book_id))

    def save(self, path: Path) -> None:
        write_json(path, self.data)

    # -------------------------------------------------------------------- queries
    @property
    def characters(self) -> dict[str, Character]:
        return self.data.characters

    def get(self, char_id: str) -> Character | None:
        return self.data.characters.get(char_id)

    def find_by_name(self, name: str) -> list[Character]:
        key = normalize_name(name)
        if not key:
            return []
        return [
            c
            for c in self.data.characters.values()
            if any(normalize_name(n) == key for n in c.all_names())
        ]

    def resolve_speaker(self, speaker: str, chunk_ids: dict[str, str] | None = None) -> str:
        """Map an LLM speaker reference (id or name) to a registry id, or 'unknown'."""
        raw = speaker.strip()
        if not raw:
            return UNKNOWN_SPEAKER
        lowered = raw.lower()
        if lowered in RESERVED_IDS:
            return lowered
        slug = slugify(raw, fallback="")
        if chunk_ids and slug in chunk_ids:
            return chunk_ids[slug]
        if slug in self.data.characters:
            return slug
        matches = self.find_by_name(raw)
        if len(matches) == 1:
            return matches[0].id
        return UNKNOWN_SPEAKER

    def known_characters(self, limit: int = 40) -> list[KnownCharacter]:
        """Most frequent characters first, for the LLM prompt."""
        ranked = sorted(self.data.characters.values(), key=lambda c: (-c.line_count, c.id))
        return [
            KnownCharacter(
                id=c.id,
                name=c.name,
                aliases=c.aliases[:5],
                gender=c.gender.value,
                description=c.description[:160],
            )
            for c in ranked[:limit]
        ]

    # ------------------------------------------------------------------- updates
    def observe(self, obs: CharacterObservation, chapter: int) -> str:
        """Merge one LLM observation; returns the canonical registry id."""
        target = self._match(obs)
        if target is not None:
            self._merge_into(target, obs)
            return target.id

        new_id = self._unique_id(obs.id or slugify(obs.name, fallback="character"))
        possibly = [i for i in obs.possibly_same_as if i in self.data.characters]
        ambiguous = [c.id for n in [obs.name, *obs.aliases] for c in self.find_by_name(n)]
        for cid in ambiguous:
            if cid not in possibly:
                possibly.append(cid)
        character = Character(
            id=new_id,
            name=obs.name,
            aliases=_dedupe(
                [a for a in obs.aliases if normalize_name(a) != normalize_name(obs.name)]
            ),
            gender=obs.gender,
            age_group=obs.age_group,
            description=obs.description,
            possibly_same_as=possibly,
            first_chapter=chapter,
        )
        self.data.characters[new_id] = character
        if possibly:
            log.info("new character %s (%s) may be the same as %s", new_id, obs.name, possibly)
        return new_id

    def count_lines(self, speakers: list[str]) -> None:
        for sid in speakers:
            if (c := self.data.characters.get(sid)) is not None:
                c.line_count += 1

    def _match(self, obs: CharacterObservation) -> Character | None:
        chars = self.data.characters

        def compatible(c: Character) -> bool:
            return Gender.UNKNOWN in (c.gender, obs.gender) or c.gender == obs.gender

        if obs.same_as and obs.same_as in chars:
            return chars[obs.same_as]
        if obs.id and obs.id in chars:
            # The LLM reused a known id: trust it unless the genders contradict.
            if compatible(chars[obs.id]):
                return chars[obs.id]
            log.warning("LLM reused id %r for %r but gender differs", obs.id, obs.name)
        candidates = {
            c.id: c
            for name in [obs.name, *obs.aliases]
            for c in self.find_by_name(name)
            if compatible(c)
        }
        if len(candidates) == 1:
            return next(iter(candidates.values()))
        return None

    @staticmethod
    def _merge_into(target: Character, obs: CharacterObservation) -> None:
        names = {normalize_name(n) for n in target.all_names()}
        for name in [obs.name, *obs.aliases]:
            if name.strip() and normalize_name(name) not in names:
                target.aliases.append(name.strip())
                names.add(normalize_name(name))
        if target.gender is Gender.UNKNOWN and obs.gender is not Gender.UNKNOWN:
            target.gender = obs.gender
        if target.age_group is AgeGroup.UNKNOWN and obs.age_group is not AgeGroup.UNKNOWN:
            target.age_group = obs.age_group
        if not target.description and obs.description:
            target.description = obs.description

    def _unique_id(self, base: str) -> str:
        base = slugify(base, fallback="character")
        if base in RESERVED_IDS:
            base = f"{base}_character"
        candidate, n = base, 2
        while candidate in self.data.characters:
            candidate = f"{base}_{n}"
            n += 1
        return candidate


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        key = normalize_name(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out

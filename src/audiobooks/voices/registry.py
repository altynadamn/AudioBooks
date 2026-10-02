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


def mixed_script(name: str) -> bool:
    """True for garbled names that mix Cyrillic and Latin letters ("Родька Колokolov")."""
    cyrillic = any("\u0400" <= ch <= "\u04ff" for ch in name)
    latin = any(ch.isascii() and ch.isalpha() for ch in name)
    return cyrillic and latin


def repair_name(name: str, source_text: str) -> str:
    """Fix tokens like "Родion" using Cyrillic words of the source text; drop the rest."""
    if not mixed_script(name):
        return name
    words = re.findall(r"[А-Яа-яЁё]+", source_text)
    fixed: list[str] = []
    for token in name.split():
        if not any(ch.isascii() and ch.isalpha() for ch in token):
            fixed.append(token)
            continue
        prefix = re.match(r"[А-Яа-яЁё]*", token).group(0)
        candidates = [
            w
            for w in words
            if len(prefix) >= 2
            and w.lower().startswith(prefix.lower())
            and abs(len(w) - len(token)) <= 3
        ]
        if candidates:
            fixed.append(max(set(candidates), key=candidates.count))
    return " ".join(fixed) or name


_PATRONYMIC = re.compile(r"(вич|вна|ична|инична|ыч)$")


def _grounded(token: str, text_lower: str) -> bool:
    """A name word occurs in the text, allowing for case endings (first 4+ letters)."""
    stem = token.lower()[: max(4, len(token) - 2)]
    return stem in text_lower


def ground_name(name: str, aliases: list[str], source_text: str) -> tuple[str, list[str]]:
    """Keep only name words that occur in the source text (the LLM sometimes invents
    surnames, e.g. "Родьон Ромашов" for Raskolnikov). Falls back to the first fully
    grounded alias, then to the original name. Ungrounded aliases are dropped."""
    text_lower = source_text.lower()
    name = repair_name(name, source_text)
    kept = [t for t in name.split() if _grounded(t, text_lower)]
    clean_aliases = [
        a
        for a in aliases
        if not mixed_script(a) and all(_grounded(t, text_lower) for t in a.split())
    ]
    if len(kept) == len(name.split()):
        return name, clean_aliases
    if kept:
        return " ".join(kept), clean_aliases
    for alias in clean_aliases:
        if any(ch.isupper() for ch in alias):
            return alias, [a for a in clean_aliases if a != alias]
    return name, clean_aliases


def name_tokens(name: str) -> frozenset[str]:
    return frozenset(normalize_name(name).split())


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
                [
                    a
                    for a in obs.aliases
                    if normalize_name(a) != normalize_name(obs.name) and not mixed_script(a)
                ]
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

    def recount_lines(self, speakers: list[str]) -> None:
        """Reset line counts from the full list of dialogue speakers of the book."""
        for c in self.data.characters.values():
            c.line_count = 0
        self.count_lines(speakers)

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
            # The LLM reused a known id: trust it unless the genders contradict or the
            # proper name is unrelated (ids are model-made slugs and can collide).
            known = chars[obs.id]
            if compatible(known) and _related(obs, known):
                return known
            log.warning(
                "LLM reused id %r for %r; not merging with %r", obs.id, obs.name, known.name
            )
        candidates = {
            c.id: c
            for name in [obs.name, *obs.aliases]
            for c in self.find_by_name(name)
            if compatible(c)
        }
        if len(candidates) == 1:
            return next(iter(candidates.values()))
        if candidates:
            return None
        # "Раскольников" vs "Родион Романович Раскольников": one name contains the other
        obs_tokens = name_tokens(obs.name)
        contained = {
            c.id: c
            for c in chars.values()
            if compatible(c) and any(_contains(obs_tokens, name_tokens(n)) for n in c.all_names())
        }
        if len(contained) == 1:
            return next(iter(contained.values()))
        return None

    @staticmethod
    def _merge_into(target: Character, obs: CharacterObservation) -> None:
        if mixed_script(target.name) and not mixed_script(obs.name):
            target.name = obs.name  # replace a garbled first spelling with a clean one
        elif _is_upgrade(target.name, obs):
            # "Старушка" -> "Алена Ивановна": keep the description as an alias
            target.aliases.append(target.name)
            target.name = obs.name
        names = {normalize_name(n) for n in target.all_names()}
        for name in [obs.name, *obs.aliases]:
            if mixed_script(name):
                continue
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


def _is_upgrade(current: str, obs: CharacterObservation) -> bool:
    """The observation carries a real name for an entry that holds only a description."""
    cur, new = name_tokens(current), name_tokens(obs.name)
    if not new or cur == new or mixed_script(obs.name):
        return False
    listed_as_alias = normalize_name(current) in {normalize_name(a) for a in obs.aliases}
    return listed_as_alias or (len(cur) == 1 and len(new) >= 2)


def _related(obs: CharacterObservation, known: Character) -> bool:
    """False only for two different multi-word names ("Алена Ивановна" vs "Пульхерия
    Александровна"). A one-word name is often a description ("Старушка") that the model
    later replaces with the real name, so it is considered related."""
    if len(name_tokens(obs.name)) < 2 or len(name_tokens(known.name)) < 2:
        return True
    obs_words = {t for n in [obs.name, *obs.aliases] for t in name_tokens(n)}
    known_words = {t for n in known.all_names() for t in name_tokens(n)}
    return any(len(t) >= 3 and not _PATRONYMIC.search(t) for t in obs_words & known_words)


def _contains(a: frozenset[str], b: frozenset[str]) -> bool:
    """True if one token set contains the other and the smaller one is a real name
    (not just a patronymic or a single short word)."""
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    if not small or small == big or not small <= big:
        return False
    return any(len(t) >= 4 and not _PATRONYMIC.search(t) for t in small)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        key = normalize_name(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out

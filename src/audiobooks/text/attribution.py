"""Deterministic speaker attribution from explicit speech tags.

In a dialogue paragraph such as "— Ещё бы! — заметил, зевая, хозяин." the author's words
name the speaker. When a tag names exactly one known character (by name or alias, in
the nominative form stored in the registry), that character is the speaker of every
dialogue span in the paragraph. Inflected forms ("сказал он Ивану") do not match, which
keeps addressees from being mistaken for speakers. The LLM's label is used whenever no
tag decides.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from audiobooks.models.script import AudiobookSegment, SegmentType
from audiobooks.voices.registry import CharacterRegistry, normalize_name

TAG_WINDOW = 6  # only the first words of the author's remark are a speech tag
_PRONOUNS = {
    "он", "она", "оно", "они", "я", "ты", "вы", "мы", "тот", "та", "те", "кто", "все",
    "he", "she", "it", "they", "i", "you", "we", "him", "her", "them",
}  # fmt: skip
_WORD = re.compile(r"\w+", re.UNICODE)
# stems of speech / thought verbs: a remark without one is not a speech tag
_SPEECH_VERB = re.compile(
    r"^(сказ|ответ|отвеч|спрос|спраш|прогово|произн|крикн|кричал|вскрик|вскрич|закрич|"
    r"прошеп|шепн|шепт|пробормо|бормот|замет|приба|доба|продолж|нача|подхват|возраз|"
    r"воскли|переб|подума|думал|промолв|молв|бурк|встав|повтор|объяв|говор|провор|ворч|"
    r"отозв|обрат|завоп|вопи|взвизг|пролепет|лепет|выговор|рявкн|простон|протян|"
    r"said|says|asked|replied|answered|cried|shouted|whispered|muttered|murmured|added|"
    r"continued|exclaimed|thought|called|began|repeated|snapped|sighed)",
    re.IGNORECASE,
)


def _words(text: str) -> list[str]:
    return _WORD.findall(normalize_name(text))


def name_index(registry: CharacterRegistry) -> dict[tuple[str, ...], set[str]]:
    """Map each usable name/alias (as a word tuple) to the characters carrying it."""
    index: dict[tuple[str, ...], set[str]] = {}
    for character in registry.characters.values():
        for name in character.all_names():
            words = tuple(_words(name))
            if not words or (len(words) == 1 and (words[0] in _PRONOUNS or len(words[0]) < 3)):
                continue
            index.setdefault(words, set()).add(character.id)
    return index


def tagged_speaker(tag: str, index: dict[tuple[str, ...], set[str]]) -> str | None:
    """The single character named in the first words of a speech tag, if any."""
    words = _words(tag)[:TAG_WINDOW]
    if not any(_SPEECH_VERB.match(w) for w in words):
        return None
    found: set[str] = set()
    for name, ids in index.items():
        n = len(name)
        if any(tuple(words[i : i + n]) == name for i in range(len(words) - n + 1)):
            found |= ids
    return next(iter(found)) if len(found) == 1 else None


def apply_speech_tags(
    segments: list[AudiobookSegment],
    registry: CharacterRegistry,
    hints: dict[int, SegmentType] | None = None,
) -> list[tuple[int, str, str]]:
    """Override dialogue speakers named by an explicit tag. Returns (index, old, new).

    ``hints`` are the segmenter's typographic types per segment index. With them, the
    paragraph structure ("— line, — tag.") is taken from typography, so a line the LLM
    mislabeled as narration is restored to dialogue when its tag names a speaker.
    """
    index = name_index(registry)
    if not index:
        return []

    def kind(seg: AudiobookSegment) -> SegmentType:
        return hints.get(seg.index, seg.type) if hints else seg.type

    changes: list[tuple[int, str, str]] = []
    for paragraph in _paragraphs(segments):
        # only "— line, — tag." paragraphs: they open with dialogue, narration is the tag
        if kind(paragraph[0]) is not SegmentType.DIALOGUE:
            continue
        tags = [s.text for s in paragraph if kind(s) is SegmentType.NARRATION]
        speakers = {sp for t in tags if (sp := tagged_speaker(t, index)) is not None}
        if len(speakers) != 1:
            continue
        (speaker,) = speakers
        for seg in paragraph:
            if kind(seg) is not SegmentType.DIALOGUE:
                continue
            if seg.type is not SegmentType.DIALOGUE or seg.speaker != speaker:
                changes.append((seg.index, seg.speaker, speaker))
                seg.type = SegmentType.DIALOGUE
                seg.speaker = speaker
    return changes


def _paragraphs(segments: Iterable[AudiobookSegment]) -> list[list[AudiobookSegment]]:
    groups: list[list[AudiobookSegment]] = []
    for seg in segments:
        if groups and groups[-1][0].paragraph == seg.paragraph:
            groups[-1].append(seg)
        else:
            groups.append([seg])
    return groups

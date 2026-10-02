"""Deterministic pre-segmentation of chapter text into narration/dialogue spans.

The LLM never rewrites text. Instead this module splits each paragraph into spans using
typographic dialogue conventions, and the LLM only *labels* the spans (type, speaker,
emotion). This guarantees the author's prose reaches TTS unchanged.

Supported conventions:
* Russian dialogue dash:  "— Реплика, — сказал он. — Продолжение."
* Quotes: «…», „…“, “…”, "…"
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from audiobooks.models.script import SegmentType

DASHES = "—–"
# paragraph opens with a dialogue dash (also "-" followed by space)
_DIALOGUE_START = re.compile(rf"^\s*(?:[{DASHES}]|-(?=\s))\s*")
# author-words switch: punctuation (optionally a closing quote) followed by " — "
_SWITCH = re.compile(rf"(?<=[,.!?…:;»”\"])\s+[{DASHES}]\s+")
_QUOTED = re.compile(r"«[^«»]+»|„[^„“]+“|“[^“”]+”|\"[^\"]+\"")
_SENTENCE_END = re.compile(r"(?<=[.!?…])[»”\"]?\s+(?=[\"«“„(\[]?[A-ZА-ЯЁ0-9—–-])")
_SOFT_BREAK = re.compile(r"(?<=[,;:])\s+")


@dataclass(frozen=True)
class Span:
    """A piece of original text with a heuristic type guess."""

    index: int  # position within the chapter, 0-based
    paragraph: int
    text: str
    hint: SegmentType


def split_paragraph(paragraph: str) -> list[tuple[str, SegmentType]]:
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    m = _DIALOGUE_START.match(paragraph)
    if m:
        return _split_dash_dialogue(paragraph[m.end() :])
    return _split_quotes(paragraph)


def _split_dash_dialogue(body: str) -> list[tuple[str, SegmentType]]:
    parts = _SWITCH.split(body)
    result = []
    for i, part in enumerate(parts):
        part = part.strip()
        if part:
            kind = SegmentType.DIALOGUE if i % 2 == 0 else SegmentType.NARRATION
            result.append((part, kind))
    return result


def _split_quotes(paragraph: str) -> list[tuple[str, SegmentType]]:
    result: list[tuple[str, SegmentType]] = []
    pos = 0
    for m in _QUOTED.finditer(paragraph):
        quoted = m.group(0)[1:-1].strip()
        if not _looks_like_speech(quoted):
            continue
        before = paragraph[pos : m.start()].strip(" ,—–-")
        if before:
            result.append((before, SegmentType.NARRATION))
        result.append((quoted, SegmentType.DIALOGUE))
        pos = m.end()
    tail = paragraph[pos:].strip(" ,—–-") if result else paragraph
    if tail:
        result.append((tail, SegmentType.NARRATION))
    return result


def _looks_like_speech(quoted: str) -> bool:
    """Short quoted names/titles (e.g. «Война и мир») are not dialogue."""
    words = quoted.split()
    return len(words) >= 4 or bool(re.search(r"[.!?…,]$", quoted))


def split_long(text: str, max_chars: int) -> list[str]:
    """Split text at sentence (then clause, then word) boundaries to fit ``max_chars``."""
    if len(text) <= max_chars:
        return [text]
    pieces = _pack(_SENTENCE_END.split(text), max_chars)
    result: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            result.append(piece)
            continue
        for clause in _pack(_SOFT_BREAK.split(piece), max_chars):
            if len(clause) <= max_chars:
                result.append(clause)
            else:
                result.extend(_pack(clause.split(), max_chars))
    return result


def _pack(parts: list[str], max_chars: int) -> list[str]:
    packed: list[str] = []
    current = ""
    for part in (p.strip() for p in parts):
        if not part:
            continue
        candidate = f"{current} {part}" if current else part
        if current and len(candidate) > max_chars:
            packed.append(current)
            current = part
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def segment_paragraphs(paragraphs: list[str], *, max_chars: int = 400) -> list[Span]:
    spans: list[Span] = []
    for p_index, paragraph in enumerate(paragraphs):
        for text, hint in split_paragraph(paragraph):
            for piece in split_long(text, max_chars):
                spans.append(Span(index=len(spans), paragraph=p_index, text=piece, hint=hint))
    return spans

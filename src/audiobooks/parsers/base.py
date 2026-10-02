"""Parser interface and text heuristics shared by concrete parsers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from audiobooks.models.book import ParsedBook, ParsedChapter

# "Глава 1", "ГЛАВА ПЕРВАЯ", "Chapter IV. The Storm", "Часть 2: Название", "Пролог", "XII"
_NUMBER = r"""(?:\d{1,3}|[ivxlcdm]{1,7}
    |[а-яё]+(?:ая|ой|ый|ий|ья|ье|ое|ть|ать|цать)
    |one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|[a-z]+teen|twenty
    |first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|[a-z]+th)"""
_HEADING_RE = re.compile(
    rf"""^\s*(
        (?:глава|часть|книга|chapter|part|book)\s+{_NUMBER}\.?(?:\s*[.:—–-]\s*.{{1,80}})?
      | (?:пролог|эпилог|предисловие|послесловие|prologue|epilogue|preface|afterword)
        (?:\s*[.:—–-]\s*.{{1,60}})?
      | [ivxlcdm]{{1,7}}\.?
      | \d{{1,3}}\.?
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

_MAJOR_RE = re.compile(
    r"^\s*(?:часть|книга|part|book|пролог|эпилог|предисловие|послесловие|prologue|epilogue"
    r"|preface|afterword)\b",
    re.IGNORECASE,
)
_NUMBER_ONLY = re.compile(r"^\s*(\d{1,3}|[ivxlcdm]{1,7})\.?\s*$", re.IGNORECASE)

# Bump when parsing output changes; books parsed by an older version are re-parsed.
PARSER_VERSION = 2

MAX_HEADING_LEN = 90
FALLBACK_CHAPTER_CHARS = 20_000


def clean_paragraph(text: str) -> str:
    """Collapse whitespace, drop soft hyphens and zero-width characters."""
    text = text.replace("\u00ad", "").replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def is_heading(paragraph: str) -> bool:
    return len(paragraph) <= MAX_HEADING_LEN and bool(_HEADING_RE.match(paragraph))


def split_into_chapters(
    paragraphs: list[str], *, default_title: str = "Chapter"
) -> list[ParsedChapter]:
    """Detect chapter headings in a flat paragraph list.

    Part-level headings ("Часть 1", "Book II", "Эпилог") prefix the chapters inside them,
    and bare numbers under a part become "Часть 1. Глава 3" / "Part 1. Chapter 3". Falls
    back to size-based pseudo-chapters (split on paragraph boundaries) when the text has
    no recognizable headings, so huge unstructured files still get processed in
    resumable pieces.
    """
    paragraphs = [p for p in (clean_paragraph(p) for p in paragraphs) if p]
    if not any(is_heading(p) for p in paragraphs):
        return split_by_size(paragraphs, default_title=default_title)

    sample = " ".join(paragraphs[:200])
    chapter_word = "Глава" if detect_language(sample) == "ru" else "Chapter"
    chapters: list[ParsedChapter] = []
    major = ""
    title = ""
    body: list[str] = []

    def emit() -> None:
        if not body:
            return
        if not title and sum(len(p) for p in body) <= 200:
            return  # tiny front matter before the first heading (title page)
        chapters.append(_chapter(len(chapters) + 1, title, list(body), default_title))

    for p in paragraphs:
        if not is_heading(p):
            body.append(p)
            continue
        emit()
        body = []
        if _MAJOR_RE.match(p):
            major = title = p
        else:
            number = _NUMBER_ONLY.match(p)
            name = f"{chapter_word} {number.group(1)}" if number else p
            title = f"{major}. {name}" if major else name
    emit()
    return chapters


def split_by_size(
    paragraphs: list[str], *, default_title: str = "Part", max_chars: int = FALLBACK_CHAPTER_CHARS
) -> list[ParsedChapter]:
    chapters: list[ParsedChapter] = []
    current: list[str] = []
    size = 0
    for p in paragraphs:
        if current and size + len(p) > max_chars:
            chapters.append(_chapter(len(chapters) + 1, "", current, default_title))
            current, size = [], 0
        current.append(p)
        size += len(p)
    if current:
        chapters.append(_chapter(len(chapters) + 1, "", current, default_title))
    return chapters


def _chapter(index: int, title: str, body: list[str], default_title: str) -> ParsedChapter:
    return ParsedChapter(
        index=index,
        title=title or f"{default_title} {index}",
        paragraphs=body,
        has_title=bool(title),
    )


def renumber(chapters: list[ParsedChapter]) -> list[ParsedChapter]:
    """Drop empty chapters and make indices contiguous (1..n)."""
    result = []
    for ch in chapters:
        paras = [p for p in (clean_paragraph(x) for x in ch.paragraphs) if p]
        if paras:
            result.append(ch.model_copy(update={"index": len(result) + 1, "paragraphs": paras}))
    return result


def detect_language(text: str) -> str:
    """Very small heuristic: 'ru' if Cyrillic dominates letters, else 'en'."""
    sample = text[:20_000]
    cyr = sum(1 for ch in sample if "\u0400" <= ch <= "\u04ff")
    lat = sum(1 for ch in sample if ch.isascii() and ch.isalpha())
    if cyr == 0 and lat == 0:
        return ""
    return "ru" if cyr >= lat else "en"


class BookParser(ABC):
    """Turns a file into a ``ParsedBook``. Implementations must never execute content."""

    extensions: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, path: Path) -> ParsedBook: ...

    def finalize(
        self,
        *,
        path: Path,
        title: str,
        author: str,
        language: str,
        chapters: list[ParsedChapter],
        warnings: list[str] | None = None,
    ) -> ParsedBook:
        chapters = renumber(chapters)
        warnings = list(warnings or [])
        if not chapters:
            from audiobooks.errors import ParseError

            raise ParseError(f"{path.name}: no readable text found")
        text_sample = " ".join(p for ch in chapters[:3] for p in ch.paragraphs[:50])
        return ParsedBook(
            title=title.strip() or path.stem,
            author=author.strip(),
            language=language or detect_language(text_sample),
            source_format=self.extensions[0].lstrip("."),
            parser_version=PARSER_VERSION,
            chapters=chapters,
            warnings=warnings,
        )

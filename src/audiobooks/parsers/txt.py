"""Plain-text parser with encoding detection (UTF-8, CP1251, ...)."""

from __future__ import annotations

import re
from pathlib import Path

from audiobooks.errors import ParseError
from audiobooks.models.book import ParsedBook
from audiobooks.parsers.base import BookParser, split_into_chapters

_DIALOGUE_LINE = re.compile(r"^\s*[—–-]\s")


def decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16"):
        if encoding == "utf-16" and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            continue
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(raw).best()
        if best is not None:
            return str(best)
    except ImportError:
        pass
    # cp1251 is the most common legacy encoding for Russian books
    return raw.decode("cp1251", errors="replace")


def text_to_paragraphs(text: str) -> list[str]:
    """Split plain text into paragraphs.

    If the text uses blank lines between paragraphs, hard-wrapped lines inside a block are
    joined; lines starting with a dialogue dash always start a new paragraph. Without blank
    lines, every non-empty line is a paragraph.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    has_blank_separators = sum(1 for ln in lines if not ln.strip()) >= max(1, len(lines) // 50)
    if not has_blank_separators:
        return [ln.strip() for ln in lines if ln.strip()]

    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        current: list[str] = []
        for line in block.split("\n"):
            if not line.strip():
                continue
            if current and (_DIALOGUE_LINE.match(line) or line.startswith(("    ", "\t"))):
                paragraphs.append(" ".join(current))
                current = []
            current.append(line.strip())
        if current:
            paragraphs.append(" ".join(current))
    return paragraphs


class TxtParser(BookParser):
    extensions = (".txt",)

    def parse(self, path: Path) -> ParsedBook:
        try:
            text = decode_bytes(path.read_bytes())
        except OSError as exc:
            raise ParseError(f"cannot read {path}: {exc}") from exc
        paragraphs = text_to_paragraphs(text)
        chapters = split_into_chapters(paragraphs)
        return self.finalize(path=path, title=path.stem, author="", language="", chapters=chapters)

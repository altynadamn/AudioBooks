"""PDF parser (PyMuPDF). Uses the text layer; image-only pages can go to a vision OCR hook.

PDF has no notion of paragraphs, so they are rebuilt from text lines:

* running headers/footers (text repeated in the page margins on many pages, page numbers)
  are removed;
* a new paragraph starts at a dialogue dash, an indented line, a line in a different font
  size (headings), or after a short line that ends a sentence;
* words hyphenated at line ends are re-joined ("пра-" + "вом" -> "правом"), keeping real
  hyphens such as "кого-нибудь".
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from audiobooks.errors import ParseError
from audiobooks.models.book import ParsedBook, ParsedChapter
from audiobooks.parsers.base import (
    BookParser,
    clean_paragraph,
    is_heading,
    split_into_chapters,
)
from audiobooks.parsers.txt import text_to_paragraphs
from audiobooks.vision.base import PageTextExtractor

log = logging.getLogger(__name__)

MIN_PAGE_TEXT_CHARS = 25
MARGIN_ZONE = 0.1  # top/bottom share of the page where running headers live
REPEAT_SHARE = 0.3  # a margin line on >= 30% of pages is a running header/footer
_PAGE_NUMBER = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
_DIALOGUE_START = re.compile(r"^\s*[—–-]\s")
_SENTENCE_END = re.compile(r"[.!?…:;»\"”)]$")
_TRAILING_HYPHEN = re.compile(r"(\w+)-$")
# second halves that mean the line-end hyphen is a real one ("кого-нибудь", "вот-с")
_KEEP_HYPHEN_AFTER = {"нибудь", "либо", "таки", "ка", "де", "с"}
_TO_PRONOUNS = {
    "что", "кто", "как", "где", "когда", "куда", "откуда", "какой", "какая", "какое",
    "какие", "каким", "какую", "чей", "чья", "чьи", "почему", "зачем", "отчего", "чего",
    "кого", "кому", "чем", "ком", "сколько", "это", "тот", "та",
}  # fmt: skip


@dataclass
class _Line:
    text: str
    x0: float
    x1: float
    size: float
    page: int
    in_margin: bool
    short: bool = False
    indented: bool = False


class PdfParser(BookParser):
    extensions = (".pdf",)

    def __init__(self, ocr: PageTextExtractor | None = None, ocr_dpi: int = 150) -> None:
        self.ocr = ocr
        self.ocr_dpi = ocr_dpi

    def parse(self, path: Path) -> ParsedBook:
        try:
            import pymupdf
        except ImportError as exc:  # pragma: no cover
            raise ParseError("PyMuPDF is not installed") from exc
        try:
            doc = pymupdf.open(path)
        except Exception as exc:
            raise ParseError(f"{path.name}: cannot open PDF: {exc}") from exc

        warnings: list[str] = []
        with doc:
            if doc.needs_pass:
                raise ParseError(f"{path.name}: PDF is password protected")
            meta = doc.metadata or {}
            pages: list[list[_Line]] = []
            ocr_pages: dict[int, list[str]] = {}
            image_only: list[int] = []
            for page in doc:
                lines = _page_lines(page)
                if sum(len(ln.text) for ln in lines) < MIN_PAGE_TEXT_CHARS and page.get_images():
                    image_only.append(page.number + 1)
                    ocr_pages[page.number] = self._ocr_page(page)
                    lines = []
                pages.append(lines)
            toc = doc.get_toc(simple=True)

        if image_only:
            if self.ocr is None:
                warnings.append(
                    f"{len(image_only)} page(s) have no text layer and were skipped "
                    f"(enable VISION_ENABLED for OCR): {_ranges(image_only)}"
                )
            else:
                warnings.append(f"{len(image_only)} page(s) were read with vision OCR")

        removed = _drop_running_headers(pages)
        if removed:
            log.info("%s: removed %d running header/footer line(s)", path.name, removed)
        paragraphs = _build_paragraphs(pages, ocr_pages)

        chapters = _chapters_from_toc(toc, paragraphs, len(pages))
        if not chapters:
            chapters = split_into_chapters([text for _, text in paragraphs])
        return self.finalize(
            path=path,
            title=meta.get("title") or "",
            author=meta.get("author") or "",
            language="",
            chapters=chapters,
            warnings=warnings,
        )

    def _ocr_page(self, page: object) -> list[str]:
        if self.ocr is None:
            return []
        pix = page.get_pixmap(dpi=self.ocr_dpi)  # type: ignore[attr-defined]
        try:
            text = self.ocr.extract_text(pix.tobytes("png"))
        except Exception as exc:
            log.warning("OCR failed for page %s: %s", page.number + 1, exc)  # type: ignore[attr-defined]
            return []
        return text_to_paragraphs(text)


def _page_lines(page: object) -> list[_Line]:
    data = page.get_text("dict", sort=True)  # type: ignore[attr-defined]
    height = float(page.rect.height) or 1.0  # type: ignore[attr-defined]
    lines: list[_Line] = []
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        for ln in block["lines"]:
            text = "".join(span["text"] for span in ln["spans"]).rstrip()
            if text.endswith(chr(0xAD)):  # soft hyphen at a line break acts like "-"
                text = text[:-1] + "-"
            text = clean_paragraph(text)
            if not text:
                continue
            sizes = [span["size"] for span in ln["spans"] if span["text"].strip()]
            x0, y0, x1, y1 = ln["bbox"]
            in_margin = y1 < height * MARGIN_ZONE or y0 > height * (1 - MARGIN_ZONE)
            lines.append(
                _Line(text, x0, x1, round(max(sizes), 1) if sizes else 0.0,
                      page.number, in_margin)  # type: ignore[attr-defined]
            )  # fmt: skip
    return lines


def _header_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.casefold())


def _drop_running_headers(pages: list[list[_Line]]) -> int:
    """Remove page numbers and margin lines that repeat on many pages."""
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({_header_key(ln.text) for ln in lines if ln.in_margin})
    threshold = max(3, int(len(pages) * REPEAT_SHARE))
    removed = 0
    for i, lines in enumerate(pages):
        kept = [
            ln
            for ln in lines
            if not (
                ln.in_margin
                and (counts[_header_key(ln.text)] >= threshold or _PAGE_NUMBER.match(ln.text))
            )
        ]
        removed += len(lines) - len(kept)
        pages[i] = kept
    return removed


def _join(prev: str, nxt: str) -> str:
    m = _TRAILING_HYPHEN.search(prev)
    if not m:
        return f"{prev} {nxt}"
    first = re.sub(r"[^\w]", "", nxt.split(maxsplit=1)[0] if nxt else "").casefold()
    keep = first in _KEEP_HYPHEN_AFTER or (first == "то" and m.group(1).casefold() in _TO_PRONOUNS)
    return prev + nxt if keep else prev[:-1] + nxt


def _build_paragraphs(
    pages: list[list[_Line]], ocr_pages: dict[int, list[str]]
) -> list[tuple[int, str]]:
    """Rebuild (page, paragraph) pairs from lines across the whole document."""
    weights: Counter[float] = Counter()
    for lines in pages:
        for ln in lines:
            weights[ln.size] += len(ln.text)
    body_size = weights.most_common(1)[0][0] if weights else 0.0

    # per-page geometry: left/right text edges of body lines
    for lines in pages:
        body = [ln for ln in lines if abs(ln.size - body_size) <= 0.6]
        if not body:
            continue
        left = min(ln.x0 for ln in body)
        right = max(ln.x1 for ln in body)
        slack = max(25.0, (right - left) * 0.08)
        for ln in lines:
            ln.short = ln.x1 < right - slack
            ln.indented = ln.x0 > left + 8

    result: list[tuple[int, str]] = []
    current = ""
    current_page = 0
    prev: _Line | None = None

    def flush() -> None:
        nonlocal current
        if current.strip():
            result.append((current_page, current.strip()))
        current = ""

    for page_no, lines in enumerate(pages):
        if page_no in ocr_pages:
            flush()
            result.extend((page_no, p) for p in ocr_pages[page_no])
            prev = None
            continue
        for ln in lines:
            heading = abs(ln.size - body_size) > 0.6
            starts_new = (
                prev is None
                or heading
                or abs(prev.size - body_size) > 0.6
                or bool(_DIALOGUE_START.match(ln.text))
                or ln.indented
                or (prev.short and bool(_SENTENCE_END.search(prev.text)))
                # same-size headings ("Глава 3", a bare "3") stand alone
                or (prev.short and is_heading(prev.text))
                or (ln.short and is_heading(ln.text))
            )
            if starts_new:
                flush()
                current, current_page = ln.text, page_no
            else:
                current = _join(current, ln.text)
            prev = ln
    flush()
    return result


def _chapters_from_toc(
    toc: list[list[object]], paragraphs: list[tuple[int, str]], page_count: int
) -> list[ParsedChapter]:
    """Use top-level PDF bookmarks as chapter boundaries (page granularity)."""
    if not toc:
        return []
    top_level = min(int(entry[0]) for entry in toc)
    marks = [(str(e[1]), int(e[2]) - 1) for e in toc if int(e[0]) == top_level and int(e[2]) > 0]
    if len(marks) < 2:
        return []
    chapters: list[ParsedChapter] = []
    front = [t for page, t in paragraphs if page < marks[0][1]]
    if sum(len(p) for p in front) > 200:
        chapters.append(ParsedChapter(index=1, title="Front matter", paragraphs=front))
    for i, (title, start) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else page_count
        body = [t for page, t in paragraphs if start <= page < max(end, start + 1)]
        if body and clean_paragraph(body[0]).casefold() == clean_paragraph(title).casefold():
            body = body[1:]
        if body:
            chapters.append(ParsedChapter(index=len(chapters) + 1, title=title, paragraphs=body))
    return chapters


def _ranges(numbers: list[int]) -> str:
    parts: list[str] = []
    start = prev = numbers[0]
    for n in [*numbers[1:], None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return ", ".join(parts)

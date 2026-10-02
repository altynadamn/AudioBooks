"""PDF parser (PyMuPDF). Uses the text layer; image-only pages can go to a vision OCR hook."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from audiobooks.errors import ParseError
from audiobooks.models.book import ParsedBook, ParsedChapter
from audiobooks.parsers.base import BookParser, clean_paragraph, split_into_chapters
from audiobooks.parsers.txt import text_to_paragraphs
from audiobooks.vision.base import PageTextExtractor

log = logging.getLogger(__name__)

MIN_PAGE_TEXT_CHARS = 25
_PAGE_NUMBER = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")


class PdfParser(BookParser):
    extensions = (".pdf",)

    def __init__(self, ocr: PageTextExtractor | None = None, ocr_dpi: int = 150) -> None:
        self.ocr = ocr
        self.ocr_dpi = ocr_dpi

    def parse(self, path: Path) -> ParsedBook:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover
            raise ParseError("PyMuPDF is not installed") from exc
        try:
            doc = fitz.open(path)
        except Exception as exc:
            raise ParseError(f"{path.name}: cannot open PDF: {exc}") from exc

        warnings: list[str] = []
        with doc:
            if doc.needs_pass:
                raise ParseError(f"{path.name}: PDF is password protected")
            meta = doc.metadata or {}
            page_paragraphs: list[list[str]] = []
            image_only: list[int] = []
            for page in doc:
                paragraphs = self._page_paragraphs(page)
                if sum(len(p) for p in paragraphs) < MIN_PAGE_TEXT_CHARS and page.get_images():
                    image_only.append(page.number + 1)
                    paragraphs = self._ocr_page(page)
                page_paragraphs.append(paragraphs)
            toc = doc.get_toc(simple=True)

        if image_only:
            if self.ocr is None:
                warnings.append(
                    f"{len(image_only)} page(s) have no text layer and were skipped "
                    f"(enable VISION_ENABLED for OCR): {_ranges(image_only)}"
                )
            else:
                warnings.append(f"{len(image_only)} page(s) were read with vision OCR")

        chapters = self._chapters_from_toc(toc, page_paragraphs)
        if not chapters:
            flat = _join_page_breaks(page_paragraphs)
            chapters = split_into_chapters(flat)
        return self.finalize(
            path=path,
            title=meta.get("title") or "",
            author=meta.get("author") or "",
            language="",
            chapters=chapters,
            warnings=warnings,
        )

    @staticmethod
    def _page_paragraphs(page: object) -> list[str]:
        blocks = page.get_text("blocks", sort=True)  # type: ignore[attr-defined]
        paragraphs = []
        for block in blocks:
            if block[6] != 0:  # image block
                continue
            text = _HYPHEN_BREAK.sub(r"\1\2", block[4])
            text = clean_paragraph(text)
            if text and not _PAGE_NUMBER.match(text):
                paragraphs.append(text)
        return paragraphs

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

    @staticmethod
    def _chapters_from_toc(toc: list[list[object]], pages: list[list[str]]) -> list[ParsedChapter]:
        """Use top-level PDF bookmarks as chapter boundaries (page granularity)."""
        if not toc:
            return []
        top_level = min(int(entry[0]) for entry in toc)
        marks = [(str(e[1]), int(e[2])) for e in toc if int(e[0]) == top_level and int(e[2]) > 0]
        if len(marks) < 2:
            return []
        chapters = []
        first_page = marks[0][1]
        if first_page > 1:
            front = _join_page_breaks(pages[: first_page - 1])
            if sum(len(p) for p in front) > 200:
                chapters.append(ParsedChapter(index=1, title="Front matter", paragraphs=front))
        for i, (title, start) in enumerate(marks):
            end = marks[i + 1][1] if i + 1 < len(marks) else len(pages) + 1
            body = _join_page_breaks(pages[start - 1 : max(start, end - 1)])
            if body and clean_paragraph(body[0]).lower() == clean_paragraph(title).lower():
                body = body[1:]
            if body:
                chapters.append(
                    ParsedChapter(index=len(chapters) + 1, title=title, paragraphs=body)
                )
        return chapters


def _join_page_breaks(pages: list[list[str]]) -> list[str]:
    """Merge a paragraph split across a page break (previous does not end a sentence)."""
    result: list[str] = []
    for paragraphs in pages:
        for i, p in enumerate(paragraphs):
            if (
                i == 0
                and result
                and not re.search(r"[.!?…:»\"”)]$", result[-1])
                and p[:1].islower()
            ):
                result[-1] = f"{result[-1]} {p}"
            else:
                result.append(p)
    return result


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

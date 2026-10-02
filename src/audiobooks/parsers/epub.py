"""EPUB parser: spine order, TOC titles, heading-based splitting."""

from __future__ import annotations

import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from audiobooks.errors import ParseError
from audiobooks.models.book import ParsedBook, ParsedChapter
from audiobooks.parsers.base import BookParser, clean_paragraph

_BLOCK_TAGS = ["h1", "h2", "h3", "h4", "p", "div", "li", "blockquote", "pre"]
_HEADINGS = {"h1", "h2", "h3"}
MIN_CHAPTER_CHARS = 300  # smaller documents (cover, copyright) are merged/skipped


class EpubParser(BookParser):
    extensions = (".epub",)

    def parse(self, path: Path) -> ParsedBook:
        try:
            from ebooklib import ITEM_DOCUMENT, epub
        except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
            raise ParseError("ebooklib is not installed") from exc
        try:
            book = epub.read_epub(str(path), options={"ignore_ncx": False})
        except Exception as exc:  # ebooklib raises a variety of exception types
            raise ParseError(f"{path.name}: cannot open EPUB: {exc}") from exc

        title = _first_meta(book, "title")
        author = _first_meta(book, "creator")
        lang = _first_meta(book, "language")
        toc_titles = _toc_titles(book.toc)

        chapters: list[ParsedChapter] = []
        warn: list[str] = []
        for item_id, _linear in book.spine:
            item = book.get_item_with_id(item_id)
            if item is None or item.get_type() != ITEM_DOCUMENT:
                continue
            fallback = toc_titles.get(item.get_name().split("#")[0], "")
            for n, (sec_title, paragraphs) in enumerate(_split_document(item.get_content())):
                text_len = sum(len(p) for p in paragraphs)
                if n == 0:
                    sec_title = sec_title or fallback
                if text_len < MIN_CHAPTER_CHARS and chapters and not sec_title:
                    chapters[-1].paragraphs.extend(paragraphs)
                    continue
                if text_len == 0:
                    continue
                chapters.append(
                    ParsedChapter(
                        index=len(chapters) + 1,
                        title=sec_title or f"Chapter {len(chapters) + 1}",
                        paragraphs=paragraphs,
                    )
                )
        if not chapters:
            warn.append("EPUB contains no text documents")
        return self.finalize(
            path=path, title=title, author=author, language=lang, chapters=chapters, warnings=warn
        )


def _first_meta(book: object, name: str) -> str:
    try:
        values = book.get_metadata("DC", name)  # type: ignore[attr-defined]
    except Exception:
        return ""
    return str(values[0][0]) if values else ""


def _toc_titles(toc: object) -> dict[str, str]:
    """Map document href -> TOC title (first occurrence wins)."""
    result: dict[str, str] = {}

    def walk(entries: object) -> None:
        if not isinstance(entries, list | tuple):
            entries = [entries]
        for entry in entries:
            if isinstance(entry, tuple) and len(entry) == 2:
                walk(entry[0])
                walk(entry[1])
                continue
            href = getattr(entry, "href", None)
            title = getattr(entry, "title", None)
            if href and title:
                result.setdefault(href.split("#")[0], clean_paragraph(title))

    walk(toc)
    return result


def _split_document(html: bytes) -> list[tuple[str, list[str]]]:
    """Split one XHTML document at h1-h3 headings into (title, paragraphs) sections."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()
    body = soup.body or soup

    sections: list[tuple[str, list[str]]] = [("", [])]
    for el in body.find_all(_BLOCK_TAGS):
        # skip containers whose block children will be visited themselves
        if el.find(_BLOCK_TAGS):
            continue
        text = clean_paragraph(el.get_text())
        if not text:
            continue
        if el.name in _HEADINGS:
            title, paras = sections[-1]
            if not paras:
                sections[-1] = (f"{title}. {text}" if title else text, paras)
            else:
                sections.append((text, []))
        else:
            sections[-1][1].append(text)
    return [(t, p) for t, p in sections if p]

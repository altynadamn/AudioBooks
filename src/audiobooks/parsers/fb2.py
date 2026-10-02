"""FictionBook 2 (FB2 / FB2.ZIP) parser."""

from __future__ import annotations

import zipfile
from pathlib import Path

from lxml import etree

from audiobooks.errors import ParseError
from audiobooks.models.book import ParsedBook, ParsedChapter
from audiobooks.parsers.base import BookParser, clean_paragraph

# Paragraph-like elements whose text becomes a paragraph
_TEXT_TAGS = {"p", "v", "subtitle", "text-author"}


def _local(tag: object) -> str:
    return etree.QName(tag).localname if isinstance(tag, str) else ""


def _text(el: etree._Element) -> str:
    return clean_paragraph("".join(el.itertext()))


class Fb2Parser(BookParser):
    extensions = (".fb2", ".fb2.zip")

    def parse(self, path: Path) -> ParsedBook:
        raw = self._read(path)
        # Hardened parser: no network, no entity expansion (XXE / billion laughs)
        parser = etree.XMLParser(
            resolve_entities=False, no_network=True, huge_tree=True, recover=True
        )
        try:
            root = etree.fromstring(raw, parser=parser)
        except etree.XMLSyntaxError as exc:
            raise ParseError(f"{path.name}: invalid FB2 XML: {exc}") from exc
        if root is None:
            raise ParseError(f"{path.name}: empty FB2 document")

        title, author, lang = self._metadata(root)
        bodies = [el for el in root if _local(el.tag) == "body"]
        main = [b for b in bodies if b.get("name") not in {"notes", "comments", "footnotes"}]
        if not main:
            raise ParseError(f"{path.name}: FB2 has no <body>")

        chapters: list[ParsedChapter] = []
        for body in main:
            # the body <title> repeats author/book title: not part of chapter titles
            self._collect(body, [], chapters, use_title=False)
        return self.finalize(
            path=path, title=title, author=author, language=lang, chapters=chapters
        )

    @staticmethod
    def _read(path: Path) -> bytes:
        try:
            if path.name.lower().endswith(".zip"):
                with zipfile.ZipFile(path) as zf:
                    names = [n for n in zf.namelist() if n.lower().endswith(".fb2")]
                    if not names:
                        raise ParseError(f"{path.name}: archive contains no .fb2 file")
                    return zf.read(names[0])
            return path.read_bytes()
        except (OSError, zipfile.BadZipFile) as exc:
            raise ParseError(f"cannot read {path}: {exc}") from exc

    @staticmethod
    def _metadata(root: etree._Element) -> tuple[str, str, str]:
        title = author = lang = ""
        for el in root.iter():
            name = _local(el.tag)
            if name == "title-info":
                for child in el.iter():
                    cname = _local(child.tag)
                    if cname == "book-title" and not title:
                        title = _text(child)
                    elif cname == "author" and not author:
                        parts = [
                            _text(x)
                            for x in child
                            if _local(x.tag) in {"first-name", "middle-name", "last-name"}
                        ]
                        author = " ".join(p for p in parts if p)
                    elif cname == "lang" and not lang:
                        lang = _text(child)
                break
        return title, author, lang

    def _collect(
        self,
        section: etree._Element,
        parents: list[str],
        out: list[ParsedChapter],
        *,
        use_title: bool = True,
    ) -> None:
        """Leaf sections become chapters; parent titles are prefixed ("Часть 1. Глава 2")."""
        title = ""
        paragraphs: list[str] = []
        subsections: list[etree._Element] = []
        for child in section:
            name = _local(child.tag)
            if name == "title":
                title = " ".join(_text(p) for p in child if _text(p))
            elif name == "section":
                subsections.append(child)
            elif name in _TEXT_TAGS:
                if t := _text(child):
                    paragraphs.append(t)
            elif name in {"poem", "cite", "epigraph"}:
                paragraphs.extend(
                    t for el in child.iter() if _local(el.tag) in _TEXT_TAGS if (t := _text(el))
                )

        path = [*parents, title] if title and use_title else parents
        if subsections:
            if paragraphs:  # text before the first subsection (e.g. a part's intro)
                out.append(self._make(out, path, paragraphs))
            for sub in subsections:
                self._collect(sub, path, out)
        elif paragraphs:
            out.append(self._make(out, path, paragraphs))

    @staticmethod
    def _make(out: list[ParsedChapter], path: list[str], paragraphs: list[str]) -> ParsedChapter:
        index = len(out) + 1
        title = ". ".join(p.rstrip(".") for p in path[-2:]) if path else f"Chapter {index}"
        return ParsedChapter(index=index, title=title, paragraphs=paragraphs, has_title=bool(path))

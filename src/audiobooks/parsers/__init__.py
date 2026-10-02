"""Book parsers. ``get_parser`` picks the implementation from the file extension."""

from __future__ import annotations

from pathlib import Path

from audiobooks.errors import UnsupportedFormatError
from audiobooks.parsers.base import BookParser
from audiobooks.parsers.epub import EpubParser
from audiobooks.parsers.fb2 import Fb2Parser
from audiobooks.parsers.pdf import PdfParser
from audiobooks.parsers.txt import TxtParser
from audiobooks.vision.base import PageTextExtractor

SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".epub", ".fb2", ".fb2.zip")


def book_extension(path: Path | str) -> str:
    name = Path(path).name.lower()
    if name.endswith(".fb2.zip"):
        return ".fb2.zip"
    return Path(name).suffix


def get_parser(path: Path, *, ocr: PageTextExtractor | None = None) -> BookParser:
    ext = book_extension(path)
    if ext == ".txt":
        return TxtParser()
    if ext in {".fb2", ".fb2.zip"}:
        return Fb2Parser()
    if ext == ".epub":
        return EpubParser()
    if ext == ".pdf":
        return PdfParser(ocr=ocr)
    raise UnsupportedFormatError(
        f"unsupported file type {ext or '(none)'}; supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


__all__ = ["SUPPORTED_EXTENSIONS", "BookParser", "book_extension", "get_parser"]

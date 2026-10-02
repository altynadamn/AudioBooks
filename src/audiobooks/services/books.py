"""Book import/inspection."""

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from audiobooks.errors import NotFoundError, UnsupportedFormatError
from audiobooks.models.book import Book, Chapter, ParsedBook
from audiobooks.parsers import SUPPORTED_EXTENSIONS, book_extension, get_parser
from audiobooks.services.context import AppContext
from audiobooks.utils import read_json, sanitize_filename, sha256_file, slugify, write_json
from audiobooks.vision.base import PageTextExtractor

log = logging.getLogger(__name__)


def validate_book_path(path: Path) -> Path:
    if not path.is_file():
        raise NotFoundError(f"file not found: {path}")
    if book_extension(path) not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"{path.name}: unsupported type; supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return path


class BookService:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    def _ocr(self) -> PageTextExtractor | None:
        if not self.ctx.settings.vision_enabled:
            return None
        from audiobooks.llm.client import OpenAICompatClient
        from audiobooks.vision.llama_vision import LlamaVisionExtractor

        s = self.ctx.settings
        client = OpenAICompatClient(
            s.llama_base_url, s.llama_model, api_key=s.llama_api_key, timeout=s.llama_timeout
        )
        return LlamaVisionExtractor(client)

    def inspect(self, path: Path) -> ParsedBook:
        path = validate_book_path(path)
        return get_parser(path, ocr=self._ocr()).parse(path)

    def import_book(self, path: Path) -> Book:
        """Register a book (idempotent: the same file content maps to the same book id)."""
        path = validate_book_path(path)
        digest = sha256_file(path)
        existing = self.ctx.db.find_book_by_hash(digest)
        if (
            existing is not None
            and self.ctx.layout(existing.id).root.joinpath("book.json").is_file()
        ):
            log.info("book already imported as %s", existing.id)
            return existing

        parsed = get_parser(path, ocr=self._ocr()).parse(path)
        book_id = f"{slugify(parsed.title, max_len=40, fallback='book')}-{digest[:8]}"
        layout = self.ctx.layout(book_id)
        layout.source_dir.mkdir(parents=True, exist_ok=True)
        source_copy = layout.source_dir / sanitize_filename(path.name)
        if not source_copy.exists():
            shutil.copyfile(path, source_copy)

        book = Book(
            id=book_id,
            title=parsed.title,
            author=parsed.author,
            language=parsed.language,
            source_format=parsed.source_format,
            source_path=str(source_copy),
            source_sha256=digest,
            chapter_count=len(parsed.chapters),
            created_at=existing.created_at if existing else datetime.now(UTC),
        )
        chapters = [
            Chapter(book_id=book_id, index=c.index, title=c.title, char_count=c.char_count)
            for c in parsed.chapters
        ]
        write_json(layout.root / "book.json", parsed)
        self.ctx.db.upsert_book(book, chapters)
        self.write_metadata(book)
        for warning in parsed.warnings:
            log.warning("%s: %s", path.name, warning)
        return book

    def load_parsed(self, book_id: str) -> ParsedBook:
        path = self.ctx.layout(book_id).root / "book.json"
        if not path.is_file():
            raise NotFoundError(f"parsed text for {book_id!r} is missing; re-import the book")
        return ParsedBook.model_validate(read_json(path))

    def write_metadata(self, book: Book) -> None:
        """Human-readable summary of the book and per-chapter state."""
        chapters = self.ctx.db.list_chapters(book.id)
        write_json(
            self.ctx.layout(book.id).metadata_file,
            {
                "book": book.model_dump(mode="json"),
                "chapters": [c.model_dump(mode="json") for c in chapters],
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

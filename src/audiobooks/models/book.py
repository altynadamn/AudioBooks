"""Book-level schemas produced by parsers and stored in the database."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ParsedChapter(BaseModel):
    """A chapter as extracted from the source file. ``index`` is 1-based."""

    index: int = Field(ge=1)
    title: str
    paragraphs: list[str]
    has_title: bool = Field(
        default=True, description="False when the title was generated (e.g. 'Chapter 3')."
    )

    @property
    def char_count(self) -> int:
        return sum(len(p) for p in self.paragraphs)


class ParsedBook(BaseModel):
    title: str
    author: str = ""
    language: str = ""
    source_format: str
    parser_version: int = 0
    chapters: list[ParsedChapter]
    warnings: list[str] = Field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(c.char_count for c in self.chapters)


class Book(BaseModel):
    """A registered book (one row in ``books``)."""

    id: str
    title: str
    author: str = ""
    language: str = ""
    source_format: str
    source_path: str
    source_sha256: str
    chapter_count: int
    created_at: datetime


class Chapter(BaseModel):
    """Per-chapter processing state (one row in ``chapters``)."""

    book_id: str
    index: int
    title: str
    char_count: int
    analysis_status: str = "pending"  # pending | done | failed
    analysis_method: str = ""  # llm | heuristic
    synthesis_status: str = "pending"  # pending | done | failed
    audio_path: str = ""
    duration_sec: float = 0.0
    error: str = ""

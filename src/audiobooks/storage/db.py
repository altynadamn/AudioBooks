"""SQLite persistence for books, chapters and jobs.

The repository exposes plain Pydantic models so a different backend (e.g. PostgreSQL)
can implement the same methods later. A connection is opened per operation, which keeps
the class safe to share between the CLI, the API thread pool and background workers.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from audiobooks.errors import NotFoundError
from audiobooks.models.book import Book, Chapter
from audiobooks.models.job import JobKind, JobOptions, JobStage, JobStatus, ProcessingJob

SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT '',
    language TEXT NOT NULL DEFAULT '',
    source_format TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL UNIQUE,
    chapter_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chapters (
    book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    title TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    analysis_status TEXT NOT NULL DEFAULT 'pending',
    analysis_method TEXT NOT NULL DEFAULT '',
    synthesis_status TEXT NOT NULL DEFAULT 'pending',
    audio_path TEXT NOT NULL DEFAULT '',
    duration_sec REAL NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (book_id, idx)
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    options TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress_done INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_book ON jobs(book_id);
"""


def _now() -> datetime:
    return datetime.now(UTC)


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ books
    def upsert_book(self, book: Book, chapters: list[Chapter]) -> None:
        """Insert a book and its chapters; existing chapter state is preserved."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO books (id, title, author, language, source_format, source_path,
                       source_sha256, chapter_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET title=excluded.title, author=excluded.author,
                       language=excluded.language, source_path=excluded.source_path,
                       chapter_count=excluded.chapter_count""",
                (
                    book.id, book.title, book.author, book.language, book.source_format,
                    book.source_path, book.source_sha256, book.chapter_count,
                    book.created_at.isoformat(),
                ),
            )  # fmt: skip
            for ch in chapters:
                conn.execute(
                    """INSERT INTO chapters (book_id, idx, title, char_count)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(book_id, idx) DO UPDATE SET title=excluded.title,
                           char_count=excluded.char_count""",
                    (book.id, ch.index, ch.title, ch.char_count),
                )

    def get_book(self, book_id: str) -> Book:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"book {book_id!r} not found")
        return Book(**dict(row))

    def find_book_by_hash(self, sha256: str) -> Book | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM books WHERE source_sha256 = ?", (sha256,)
            ).fetchone()
        return Book(**dict(row)) if row else None

    def list_books(self) -> list[Book]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM books ORDER BY created_at DESC").fetchall()
        return [Book(**dict(r)) for r in rows]

    # --------------------------------------------------------------- chapters
    def list_chapters(self, book_id: str) -> list[Chapter]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chapters WHERE book_id = ? ORDER BY idx", (book_id,)
            ).fetchall()
        return [_chapter(r) for r in rows]

    def get_chapter(self, book_id: str, index: int) -> Chapter:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE book_id = ? AND idx = ?", (book_id, index)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"chapter {index} of {book_id!r} not found")
        return _chapter(row)

    _CHAPTER_FIELDS = frozenset(
        {"analysis_status", "analysis_method", "synthesis_status", "audio_path",
         "duration_sec", "error"}
    )  # fmt: skip

    def update_chapter(self, book_id: str, index: int, **fields: object) -> None:
        unknown = set(fields) - self._CHAPTER_FIELDS
        if unknown:
            raise ValueError(f"unknown chapter fields: {sorted(unknown)}")
        if not fields:
            return
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE chapters SET {assignments} WHERE book_id = ? AND idx = ?",
                (*fields.values(), book_id, index),
            )

    # ------------------------------------------------------------------- jobs
    def create_job(self, book_id: str, kind: JobKind, options: JobOptions) -> ProcessingJob:
        now = _now()
        job = ProcessingJob(
            id=uuid.uuid4().hex[:12],
            book_id=book_id,
            kind=kind,
            options=options,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO jobs (id, book_id, kind, options, status, stage, created_at,
                       updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.id, job.book_id, job.kind, job.options.model_dump_json(),
                    job.status, job.stage, now.isoformat(), now.isoformat(),
                ),
            )  # fmt: skip
        return job

    _JOB_FIELDS = frozenset(
        {"status", "stage", "progress_done", "progress_total", "message", "error"}
    )

    def update_job(self, job_id: str, **fields: object) -> None:
        unknown = set(fields) - self._JOB_FIELDS
        if unknown:
            raise ValueError(f"unknown job fields: {sorted(unknown)}")
        fields["updated_at"] = _now().isoformat()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?", (*fields.values(), job_id)
            )

    def get_job(self, job_id: str) -> ProcessingJob:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"job {job_id!r} not found")
        return _job(row)

    def list_jobs(self, book_id: str | None = None) -> list[ProcessingJob]:
        query = "SELECT * FROM jobs"
        params: tuple[str, ...] = ()
        if book_id:
            query += " WHERE book_id = ?"
            params = (book_id,)
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY created_at DESC", params).fetchall()
        return [_job(r) for r in rows]


def _chapter(row: sqlite3.Row) -> Chapter:
    data = dict(row)
    data["index"] = data.pop("idx")
    return Chapter(**data)


def _job(row: sqlite3.Row) -> ProcessingJob:
    data = dict(row)
    data["options"] = JobOptions.model_validate_json(data["options"])
    data["kind"] = JobKind(data["kind"])
    data["status"] = JobStatus(data["status"])
    data["stage"] = JobStage(data["stage"])
    return ProcessingJob(**data)

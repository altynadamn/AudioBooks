from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from audiobooks.errors import NotFoundError
from audiobooks.models.book import Book, Chapter
from audiobooks.models.job import GenerationMode, JobKind, JobOptions, JobStatus
from audiobooks.models.script import AudiobookSegment, ChapterScript, normalize_emotion
from audiobooks.storage.db import Database
from audiobooks.utils import ensure_within, sanitize_filename, slugify


def test_slugify_transliterates() -> None:
    assert slugify("Анна Сергеевна") == "anna_sergeevna"
    assert slugify("Война и мир!") == "voina_i_mir"
    assert slugify("???", fallback="x") == "x"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\Windows\\system32\\evil.txt", "evil.txt"),
        ('bad<>:"|?*.epub', "bad_______.epub"),
        ("CON.txt", "_CON.txt"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_ensure_within(tmp_path: Path) -> None:
    assert ensure_within(tmp_path, tmp_path / "a" / "b") == (tmp_path / "a" / "b").resolve()
    with pytest.raises(ValueError):
        ensure_within(tmp_path / "voices", tmp_path / "voices" / ".." / "secret.wav")


def test_segment_serialization_roundtrip() -> None:
    script = ChapterScript(
        book_id="b",
        chapter=1,
        title="Глава 1",
        analysis="llm",
        segments=[
            AudiobookSegment(index=0, type="narration", text="Снег.", paragraph=0),
            AudiobookSegment(
                index=1, type="dialogue", speaker="anna", text="Ты пришёл?", emotion="quiet"
            ),
        ],
    )
    restored = ChapterScript.model_validate_json(script.model_dump_json())
    assert restored == script
    assert restored.segments[1].emotion == "whisper"
    assert restored.speakers == ["narrator", "anna"]


def test_normalize_emotion() -> None:
    assert normalize_emotion("ANGRY") == "angry"
    assert normalize_emotion("furious") == "angry"
    assert normalize_emotion("bewildered") == "neutral"
    assert normalize_emotion(None) == "neutral"


def test_database_books_chapters_jobs(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite3")
    book = Book(
        id="b1", title="T", source_format="txt", source_path="x", source_sha256="h",
        chapter_count=2, created_at=datetime.now(UTC),
    )  # fmt: skip
    chapters = [Chapter(book_id="b1", index=i, title=f"C{i}", char_count=10) for i in (1, 2)]
    db.upsert_book(book, chapters)
    db.update_chapter("b1", 1, analysis_status="done", analysis_method="llm")
    db.upsert_book(book, chapters)  # re-import keeps state
    assert db.get_chapter("b1", 1).analysis_status == "done"
    assert db.find_book_by_hash("h").id == "b1"

    job = db.create_job("b1", JobKind.GENERATE, JobOptions(mode=GenerationMode.SIMPLE))
    db.update_job(job.id, status=JobStatus.RUNNING, progress_done=3, progress_total=10)
    loaded = db.get_job(job.id)
    assert loaded.status is JobStatus.RUNNING
    assert loaded.options.mode is GenerationMode.SIMPLE
    assert loaded.progress_done == 3
    assert [j.id for j in db.list_jobs("b1")] == [job.id]

    with pytest.raises(ValueError):
        db.update_job(job.id, book_id="evil")
    with pytest.raises(NotFoundError):
        db.get_book("missing")

"""Deterministic on-disk layout for one book's output directory.

output/<book-id>/
    metadata.json
    character_registry.json
    source/<original file>
    script/chapter_001.json
    analysis/chapter_001/chunk_0001.json      (per-chunk LLM cache, enables resume)
    audio/chapter_001/segment_0001.wav
    audio/chapter_001/manifest.json            (segment cache keys)
    audio/chapter_001.m4a
    <book-id>.m4b
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BookLayout:
    root: Path
    book_id: str

    @classmethod
    def for_book(cls, output_dir: Path, book_id: str) -> BookLayout:
        return cls(root=output_dir / book_id, book_id=book_id)

    @property
    def metadata_file(self) -> Path:
        return self.root / "metadata.json"

    @property
    def registry_file(self) -> Path:
        return self.root / "character_registry.json"

    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    @property
    def script_dir(self) -> Path:
        return self.root / "script"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    def script_file(self, chapter: int) -> Path:
        return self.script_dir / f"chapter_{chapter:03d}.json"

    def analysis_dir(self, chapter: int) -> Path:
        return self.root / "analysis" / f"chapter_{chapter:03d}"

    def chunk_cache_file(self, chapter: int, chunk: int) -> Path:
        return self.analysis_dir(chapter) / f"chunk_{chunk:04d}.json"

    def chapter_audio_dir(self, chapter: int) -> Path:
        return self.audio_dir / f"chapter_{chapter:03d}"

    def segment_file(self, chapter: int, segment_index: int) -> Path:
        # 1-based file names are friendlier to humans browsing the folder
        return self.chapter_audio_dir(chapter) / f"segment_{segment_index + 1:04d}.wav"

    def segment_manifest(self, chapter: int) -> Path:
        return self.chapter_audio_dir(chapter) / "manifest.json"

    def chapter_audio_file(self, chapter: int, fmt: str) -> Path:
        return self.audio_dir / f"chapter_{chapter:03d}.{fmt}"

    @property
    def m4b_file(self) -> Path:
        return self.root / f"{self.book_id}.m4b"

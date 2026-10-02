"""Chapter and book assembly through FFmpeg."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from audiobooks.audio.ffmpeg import (
    ChapterMark,
    FFmpeg,
    chapter_command,
    concat_list,
    ffmetadata,
    m4b_command,
    resample_command,
    silence_command,
)
from audiobooks.errors import AudioBooksError
from audiobooks.models.book import Book
from audiobooks.models.script import AudiobookSegment, ChapterScript
from audiobooks.services.context import AppContext
from audiobooks.storage.layout import BookLayout
from audiobooks.utils import atomic_write_text, read_json, sha256_text, write_json

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssembledChapter:
    chapter: int
    title: str
    path: Path
    duration: float


class AssemblyService:
    def __init__(self, ctx: AppContext, ffmpeg: FFmpeg | None = None) -> None:
        self.ctx = ctx
        self._ffmpeg = ffmpeg

    @property
    def ffmpeg(self) -> FFmpeg:
        if self._ffmpeg is None:
            self._ffmpeg = FFmpeg.from_config(self.ctx.settings.ffmpeg_path)
        return self._ffmpeg

    def pause_ms(self, prev: AudiobookSegment | None, seg: AudiobookSegment) -> int:
        s = self.ctx.settings
        if prev is None:
            return s.pause_chapter_start_ms
        if prev.paragraph != seg.paragraph:
            return s.pause_paragraph_ms
        if prev.speaker != seg.speaker or prev.type != seg.type:
            return s.pause_speaker_change_ms
        return s.pause_same_paragraph_ms

    def assemble_chapter(self, book: Book, script: ChapterScript) -> AssembledChapter:
        settings = self.ctx.settings
        layout = self.ctx.layout(book.id)
        manifest = _read_dict(layout.segment_manifest(script.chapter))
        missing = [
            s.index
            for s in script.segments
            if str(s.index) not in manifest
            or not layout.segment_file(script.chapter, s.index).is_file()
        ]
        if missing:
            raise AudioBooksError(
                f"chapter {script.chapter}: {len(missing)} segment(s) have no audio yet "
                f"(first: {missing[0]}); run synthesis first"
            )

        rates = Counter(int(manifest[str(s.index)]["sample_rate"]) for s in script.segments)
        rate = rates.most_common(1)[0][0] if rates else 24_000
        output = layout.chapter_audio_file(script.chapter, settings.audio_format)
        key = sha256_text(
            "asm-v1",
            settings.audio_format, settings.audio_bitrate, str(settings.audio_sample_rate),
            str(settings.audio_loudnorm), book.title, script.title,
            *(f"{s.index}:{manifest[str(s.index)]['key']}:{self.pause_ms(p, s)}"
              for p, s in zip([None, *script.segments[:-1]], script.segments, strict=True)),
        )[:24]  # fmt: skip
        state_file = layout.chapter_audio_dir(script.chapter) / "assembly.json"
        state = _read_dict(state_file)
        if output.is_file() and state.get("key") == key:
            log.info("chapter %d: audio up to date", script.chapter)
            return AssembledChapter(script.chapter, script.title, output, state["duration"])

        files: list[Path] = []
        duration = 0.0
        prev: AudiobookSegment | None = None
        for seg in script.segments:
            pause = self.pause_ms(prev, seg)
            if pause > 0:
                files.append(self._silence(layout, pause, rate))
                duration += pause / 1000
            entry = manifest[str(seg.index)]
            seg_file = layout.segment_file(script.chapter, seg.index)
            if int(entry["sample_rate"]) != rate:
                seg_file = self._resampled(layout, seg_file, rate)
            files.append(seg_file)
            duration += float(entry["duration"])
            prev = seg
        files.append(self._silence(layout, settings.pause_paragraph_ms, rate))
        duration += settings.pause_paragraph_ms / 1000

        list_file = layout.tmp_dir / f"chapter_{script.chapter:03d}.ffconcat"
        atomic_write_text(list_file, concat_list(files))
        tmp_out = output.with_name(f".{output.stem}.partial{output.suffix}")
        self.ffmpeg.run(
            chapter_command(
                list_file,
                tmp_out,
                fmt=settings.audio_format,
                bitrate=settings.audio_bitrate,
                sample_rate=settings.audio_sample_rate,
                loudnorm=settings.audio_loudnorm,
                title=script.title,
                album=book.title,
                artist=book.author,
                track=script.chapter,
            )
        )
        tmp_out.replace(output)
        write_json(state_file, {"key": key, "duration": round(duration, 3)})
        log.info("chapter %d: wrote %s (%.1f s)", script.chapter, output.name, duration)
        return AssembledChapter(script.chapter, script.title, output, duration)

    def assemble_book(self, book: Book, chapters: list[AssembledChapter]) -> Path:
        """Join chapter files into a single M4B with chapter markers."""
        layout = self.ctx.layout(book.id)
        marks: list[ChapterMark] = []
        cursor = 0
        for ch in chapters:
            length = round(ch.duration * 1000)
            marks.append(ChapterMark(ch.title, cursor, cursor + length))
            cursor += length
        list_file = layout.tmp_dir / "book.ffconcat"
        meta_file = layout.tmp_dir / "book.ffmetadata"
        atomic_write_text(list_file, concat_list([c.path for c in chapters]))
        atomic_write_text(meta_file, ffmetadata(book.title, book.author, marks))
        output = layout.m4b_file
        tmp_out = output.with_name(f".{output.stem}.partial.m4b")
        self.ffmpeg.run(
            m4b_command(
                list_file,
                meta_file,
                tmp_out,
                source_fmt=self.ctx.settings.audio_format,
                bitrate=self.ctx.settings.audio_bitrate,
            )
        )
        tmp_out.replace(output)
        log.info("wrote %s (%d chapters, %.1f min)", output.name, len(chapters), cursor / 60000)
        return output

    def _silence(self, layout: BookLayout, ms: int, rate: int) -> Path:
        path = layout.tmp_dir / "silence" / f"silence_{ms}ms_{rate}.wav"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            self.ffmpeg.run(silence_command(path, duration_ms=ms, sample_rate=rate))
        return path

    def _resampled(self, layout: BookLayout, source: Path, rate: int) -> Path:
        target = layout.tmp_dir / "resampled" / source.parent.name / f"{source.stem}_{rate}.wav"
        if not target.is_file() or target.stat().st_mtime < source.stat().st_mtime:
            target.parent.mkdir(parents=True, exist_ok=True)
            self.ffmpeg.run(resample_command(source, target, sample_rate=rate))
        return target


def _read_dict(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = read_json(path)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}

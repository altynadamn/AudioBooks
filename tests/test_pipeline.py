"""Integration tests of the full pipeline with fake LLM/TTS (FFmpeg is real if installed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from audiobooks.audio.ffmpeg import find_ffmpeg
from audiobooks.errors import AudioBooksError, FFmpegError, TTSError
from audiobooks.llm.fake import FakeLLMProvider
from audiobooks.models.job import GenerationMode, JobKind, JobOptions, JobStatus
from audiobooks.models.script import ChapterScript
from audiobooks.services.books import BookService
from audiobooks.services.context import AppContext
from audiobooks.services.pipeline import Pipeline
from audiobooks.tts.fake import FakeTTSBackend
from audiobooks.utils import read_json
from audiobooks.voices.registry import CharacterRegistry


def _has_ffmpeg() -> bool:
    try:
        find_ffmpeg("")
    except FFmpegError:
        return False
    return True


needs_ffmpeg = pytest.mark.skipif(not _has_ffmpeg(), reason="FFmpeg not installed")


def _job(ctx: AppContext, book_id: str, kind: JobKind, **opts: object) -> str:
    return Pipeline(ctx).create_job(book_id, kind, JobOptions(**opts)).id


def test_import_is_idempotent(ctx: AppContext, sample_txt: Path) -> None:
    a = BookService(ctx).import_book(sample_txt)
    b = BookService(ctx).import_book(sample_txt)
    assert a.id == b.id
    assert a.chapter_count == 2
    assert len(ctx.db.list_chapters(a.id)) == 2
    assert ctx.layout(a.id).metadata_file.is_file()


def test_analysis_builds_scripts_and_registry(
    ctx: AppContext, sample_txt: Path, fake_llm: FakeLLMProvider
) -> None:
    book = BookService(ctx).import_book(sample_txt)
    job = Pipeline(ctx).run(_job(ctx, book.id, JobKind.ANALYZE, mode=GenerationMode.CAST))
    assert job.status is JobStatus.DONE

    layout = ctx.layout(book.id)
    script = ChapterScript.model_validate(read_json(layout.script_file(1)))
    assert script.analysis == "llm"
    texts = [s.text for s in script.segments]
    assert texts[0] == "Глава 1"  # real titles are voiced
    dialogue = [s for s in script.segments if s.type == "dialogue"]
    assert [(s.speaker, s.text) for s in dialogue] == [
        ("anna", "Ты всё-таки пришёл?"),
        ("ivan", "Я обещал,"),
    ]
    # the original text is preserved exactly (minus dialogue dashes)
    assert "тихо спросила Анна." in texts

    registry = CharacterRegistry.load(layout.registry_file, book.id)
    assert set(registry.characters) == {"anna", "ivan"}
    assert registry.get("anna").voice_id.startswith("female")
    assert registry.get("ivan").voice_id.startswith("male")
    assert registry.get("anna").line_count == 2  # chapter 1 + chapter 2

    # re-running analysis uses stored scripts: no new LLM calls
    calls = len(fake_llm.calls)
    Pipeline(ctx).run(_job(ctx, book.id, JobKind.ANALYZE, mode=GenerationMode.CAST))
    assert len(fake_llm.calls) == calls
    assert CharacterRegistry.load(layout.registry_file, book.id).get("anna").line_count == 2


def test_chunk_cache_resumes_interrupted_analysis(
    ctx: AppContext, sample_txt: Path, fake_llm: FakeLLMProvider
) -> None:
    book = BookService(ctx).import_book(sample_txt)
    original = fake_llm.analyze_chunk
    state = {"n": 0}

    def flaky(request):
        state["n"] += 1
        if state["n"] == 2:  # chapter 2's chunk fails once
            raise AudioBooksError("LLM crashed")
        return original(request)

    fake_llm.analyze_chunk = flaky  # type: ignore[method-assign]
    job_id = _job(ctx, book.id, JobKind.ANALYZE, mode=GenerationMode.CAST)
    with pytest.raises(AudioBooksError):
        Pipeline(ctx).run(job_id)
    assert ctx.db.get_job(job_id).status is JobStatus.FAILED
    assert ctx.db.get_chapter(book.id, 1).analysis_status == "done"
    assert ctx.db.get_chapter(book.id, 2).analysis_status == "failed"

    Pipeline(ctx).run(job_id)  # resume
    assert ctx.db.get_job(job_id).status is JobStatus.DONE
    assert state["n"] == 3  # only chapter 2 was re-requested


@needs_ffmpeg
def test_generate_cast_end_to_end_and_resume(
    ctx: AppContext, sample_txt: Path, fake_tts: FakeTTSBackend
) -> None:
    book = BookService(ctx).import_book(sample_txt)
    job = Pipeline(ctx).run(_job(ctx, book.id, JobKind.GENERATE, mode=GenerationMode.CAST))
    assert job.status is JobStatus.DONE
    layout = ctx.layout(book.id)

    first_calls = len(fake_tts.calls)
    assert first_calls > 0
    assert fake_tts.closed  # TTS resources released after synthesis
    voices = {c.voice.id for c in fake_tts.calls}
    assert "narrator_01" in voices and len(voices) == 3  # narrator + Anna + Ivan

    for ch in (1, 2):
        assert layout.chapter_audio_file(ch, "m4a").stat().st_size > 0
        assert layout.segment_file(ch, 0).is_file()
        assert ctx.db.get_chapter(book.id, ch).duration_sec > 0
    assert layout.m4b_file.stat().st_size > 0

    # running again regenerates nothing
    fake_tts.calls.clear()
    Pipeline(ctx).run(_job(ctx, book.id, JobKind.GENERATE, mode=GenerationMode.CAST))
    assert fake_tts.calls == []

    # deleting one segment regenerates only that segment
    layout.segment_file(2, 1).unlink()
    Pipeline(ctx).run(_job(ctx, book.id, JobKind.GENERATE, mode=GenerationMode.CAST))
    assert len(fake_tts.calls) == 1


@needs_ffmpeg
def test_simple_mode_needs_no_llm_and_uses_one_voice(
    ctx: AppContext, sample_txt: Path, fake_tts: FakeTTSBackend, fake_llm: FakeLLMProvider
) -> None:
    book = BookService(ctx).import_book(sample_txt)
    job = Pipeline(ctx).run(
        _job(ctx, book.id, JobKind.GENERATE, mode=GenerationMode.SIMPLE, chapters=[2])
    )
    assert job.status is JobStatus.DONE
    assert fake_llm.calls == []
    assert {c.voice.id for c in fake_tts.calls} == {"narrator_01"}
    layout = ctx.layout(book.id)
    assert layout.chapter_audio_file(2, "m4a").is_file()
    assert not layout.chapter_audio_file(1, "m4a").exists()
    assert not layout.m4b_file.exists()  # partial book -> no m4b


def test_tts_failure_keeps_finished_segments(
    ctx: AppContext, sample_txt: Path, fake_tts: FakeTTSBackend
) -> None:
    book = BookService(ctx).import_book(sample_txt)
    original = fake_tts.synthesize
    state = {"n": 0}

    def flaky(request):
        state["n"] += 1
        if state["n"] == 4:
            raise TTSError("CUDA out of memory")
        return original(request)

    fake_tts.synthesize = flaky  # type: ignore[method-assign]
    job_id = _job(ctx, book.id, JobKind.GENERATE, mode=GenerationMode.SIMPLE, build_m4b=False)
    with pytest.raises(TTSError):
        Pipeline(ctx).run(job_id)
    assert fake_tts.closed  # released even on failure
    done = [p for p in ctx.layout(book.id).audio_dir.rglob("segment_*.wav")]
    assert len(done) == 3

    fake_tts.synthesize = original  # type: ignore[method-assign]
    fake_tts.calls.clear()
    if not _has_ffmpeg():
        pytest.skip("FFmpeg not installed")
    Pipeline(ctx).run(job_id)
    total = sum(
        len(ChapterScript.model_validate(read_json(ctx.layout(book.id).script_file(c))).segments)
        for c in (1, 2)
    )
    assert len(fake_tts.calls) == total - 3


def test_cast_mode_without_llm_server_fails_clearly(ctx: AppContext, sample_txt: Path) -> None:
    class DownLLM(FakeLLMProvider):
        def health(self) -> bool:
            return False

    ctx.llm_factory = lambda _s: DownLLM()
    book = BookService(ctx).import_book(sample_txt)
    with pytest.raises(AudioBooksError, match="not reachable"):
        Pipeline(ctx).run(_job(ctx, book.id, JobKind.ANALYZE, mode=GenerationMode.CAST))


def test_unknown_chapter_is_rejected(ctx: AppContext, sample_txt: Path) -> None:
    book = BookService(ctx).import_book(sample_txt)
    with pytest.raises(AudioBooksError, match="do not exist"):
        Pipeline(ctx).run(_job(ctx, book.id, JobKind.ANALYZE, chapters=[9]))

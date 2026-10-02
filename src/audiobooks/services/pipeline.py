"""End-to-end orchestration of a processing job.

Stages run strictly one after another so that only one large model occupies the GPU:

    analysis (LLM via llama-server)  ->  [llama-server stopped if auto-managed]
    casting                          ->  synthesis (Qwen3-TTS, unloaded afterwards)
    assembly (FFmpeg)                ->  M4B

Every stage skips work that is already done (cached chunk analyses, valid scripts,
segment WAVs with matching cache keys, up-to-date chapter files), so ``resume`` simply
runs the same job again.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Protocol

from audiobooks.errors import AudioBooksError, LLMError
from audiobooks.llm.base import LLMProvider
from audiobooks.models.book import Book, ParsedChapter
from audiobooks.models.job import (
    GenerationMode,
    JobKind,
    JobOptions,
    JobStage,
    JobStatus,
    ProcessingJob,
)
from audiobooks.models.script import ChapterScript
from audiobooks.services.analysis import HEURISTIC, LLM, AnalysisService
from audiobooks.services.assembly import AssembledChapter, AssemblyService
from audiobooks.services.books import BookService
from audiobooks.services.context import AppContext
from audiobooks.services.synthesis import SynthesisService
from audiobooks.voices.casting import VoiceCaster
from audiobooks.voices.registry import CharacterRegistry

log = logging.getLogger(__name__)


class ProgressReporter(Protocol):
    def stage(self, stage: JobStage, total: int, description: str) -> None: ...

    def advance(self, done: int, total: int, message: str = "") -> None: ...


class NullReporter:
    def stage(self, stage: JobStage, total: int, description: str) -> None:
        pass

    def advance(self, done: int, total: int, message: str = "") -> None:
        pass


def analysis_method(options: JobOptions) -> str:
    return LLM if options.mode is GenerationMode.CAST or options.force_llm else HEURISTIC


class Pipeline:
    def __init__(self, ctx: AppContext, reporter: ProgressReporter | None = None) -> None:
        self.ctx = ctx
        self.reporter = reporter or NullReporter()
        self.books = BookService(ctx)
        self.analysis = AnalysisService(ctx)
        self.synthesis = SynthesisService(ctx)
        self.assembly = AssemblyService(ctx)

    # ------------------------------------------------------------------ jobs
    def create_job(self, book_id: str, kind: JobKind, options: JobOptions) -> ProcessingJob:
        self.ctx.db.get_book(book_id)  # validates existence
        return self.ctx.db.create_job(book_id, kind, options)

    def run(self, job_id: str) -> ProcessingJob:
        db = self.ctx.db
        job = db.get_job(job_id)
        db.update_job(job_id, status=JobStatus.RUNNING, error="", message="")
        try:
            self._run(job)
        except AudioBooksError as exc:
            db.update_job(job_id, status=JobStatus.FAILED, error=str(exc))
            raise
        except Exception as exc:
            log.exception("job %s crashed", job_id)
            db.update_job(job_id, status=JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            with contextlib.suppress(Exception):
                self.books.write_metadata(db.get_book(job.book_id))
        db.update_job(job_id, status=JobStatus.DONE, stage=JobStage.FINISHED, message="done")
        return db.get_job(job_id)

    def _run(self, job: ProcessingJob) -> None:
        book = self.ctx.db.get_book(job.book_id)
        parsed = self.books.load_parsed(book.id)
        chapters = _select(parsed.chapters, job.options.chapters)
        layout = self.ctx.layout(book.id)
        registry = CharacterRegistry.load(layout.registry_file, book.id)

        scripts = self._analysis_stage(job, book, chapters, registry)
        self._casting_stage(job, registry)
        registry.save(layout.registry_file)
        if job.kind is JobKind.ANALYZE:
            return
        self._synthesis_stage(job, book, scripts, registry)
        assembled = self._assembly_stage(job, book, scripts)
        complete = job.options.chapters is None or len(assembled) == len(parsed.chapters)
        if job.options.build_m4b and complete and assembled:
            self.ctx.db.update_job(job.id, message="building M4B")
            self.assembly.assemble_book(book, assembled)

    # ---------------------------------------------------------------- stages
    def _analysis_stage(
        self,
        job: ProcessingJob,
        book: Book,
        chapters: list[ParsedChapter],
        registry: CharacterRegistry,
    ) -> list[ChapterScript]:
        db = self.ctx.db
        layout = self.ctx.layout(book.id)
        method = analysis_method(job.options)
        db.update_job(
            job.id, stage=JobStage.ANALYSIS, progress_done=0, progress_total=len(chapters)
        )
        self.reporter.stage(JobStage.ANALYSIS, len(chapters), f"Analyzing ({method})")

        scripts: dict[int, ChapterScript] = {}
        todo: list[ParsedChapter] = []
        for ch in chapters:
            existing = self.analysis.existing_script(layout, ch, method)
            if existing is not None:
                scripts[ch.index] = existing
            else:
                todo.append(ch)
        done = len(scripts)
        self.reporter.advance(done, len(chapters), f"{done} chapter(s) already analyzed")

        if todo:
            with self._llm_session(method) as llm:
                for ch in todo:
                    db.update_job(job.id, message=f"analyzing chapter {ch.index}: {ch.title}")

                    def on_chunk(
                        i: int, n: int, ch: ParsedChapter = ch, finished: int = done
                    ) -> None:
                        self.reporter.advance(
                            finished, len(chapters), f"chapter {ch.index}: chunk {i}/{n}"
                        )

                    try:
                        script = self.analysis.analyze_chapter(
                            book, ch, method=method, registry=registry, llm=llm, on_chunk=on_chunk
                        )
                    except AudioBooksError as exc:
                        db.update_chapter(
                            book.id, ch.index, analysis_status="failed", error=str(exc)
                        )
                        raise
                    scripts[ch.index] = script
                    db.update_chapter(
                        book.id, ch.index, analysis_status="done", analysis_method=method, error=""
                    )
                    done += 1
                    db.update_job(job.id, progress_done=done)
                    self.reporter.advance(done, len(chapters), f"chapter {ch.index} analyzed")
        registry.save(layout.registry_file)
        return [scripts[ch.index] for ch in chapters]

    @contextlib.contextmanager
    def _llm_session(self, method: str) -> Iterator[LLMProvider | None]:
        """Provide an LLM for the analysis stage; stop a managed server afterwards."""
        if method != LLM:
            yield None
            return
        manager = None
        if self.ctx.should_manage_llm:
            from audiobooks.llm.server import LlamaServerManager

            manager = LlamaServerManager(
                self.ctx.settings, log_dir=self.ctx.settings.data_dir / "logs"
            )
            manager.start()
        provider = self.ctx.llm_factory(self.ctx.settings)
        try:
            if not provider.health():
                raise LLMError(
                    f"LLM server at {self.ctx.settings.llama_base_url} is not reachable. "
                    "Start llama-server (see README) or set LLM_AUTO_MANAGE=true with "
                    "LLAMA_MODEL_PATH, or use --mode simple."
                )
            yield provider
        finally:
            provider.close()
            if manager is not None:
                manager.stop()

    def _casting_stage(self, job: ProcessingJob, registry: CharacterRegistry) -> None:
        if job.options.mode is not GenerationMode.CAST:
            return
        self.ctx.db.update_job(job.id, stage=JobStage.CASTING, message="assigning voices")
        caster = VoiceCaster(self.ctx.library, self.ctx.settings.narrator_voice)
        for char_id, voice_id in caster.cast(registry).items():
            log.info("cast %s -> %s", char_id, voice_id)

    def _synthesis_stage(
        self,
        job: ProcessingJob,
        book: Book,
        scripts: list[ChapterScript],
        registry: CharacterRegistry,
    ) -> None:
        db = self.ctx.db
        backend = self.ctx.tts_factory(self.ctx.settings, self.ctx.library)
        try:
            items, total = self.synthesis.plan(book, scripts, registry, job.options.mode, backend)
            cached = total - len(items)
            db.update_job(
                job.id,
                stage=JobStage.SYNTHESIS,
                progress_done=cached,
                progress_total=total,
                message=f"{len(items)} segment(s) to synthesize, {cached} cached",
            )
            self.reporter.stage(JobStage.SYNTHESIS, total, "Synthesizing speech")
            self.reporter.advance(cached, total, f"{cached} segment(s) cached")

            def on_segment(n: int, _count: int) -> None:
                if n % 5 == 0 or n == len(items):
                    db.update_job(job.id, progress_done=cached + n)
                item = items[n - 1]
                self.reporter.advance(
                    cached + n, total, f"ch {item.chapter} seg {item.index + 1} ({item.voice.id})"
                )

            # on failure the job is marked failed; finished segments stay cached for resume
            self.synthesis.run(book, items, backend, on_segment)
        finally:
            backend.close()  # releases TTS model + CUDA cache
        for s in scripts:
            db.update_chapter(book.id, s.chapter, synthesis_status="done")

    def _assembly_stage(
        self, job: ProcessingJob, book: Book, scripts: list[ChapterScript]
    ) -> list[AssembledChapter]:
        db = self.ctx.db
        db.update_job(job.id, stage=JobStage.ASSEMBLY, progress_done=0, progress_total=len(scripts))
        self.reporter.stage(JobStage.ASSEMBLY, len(scripts), "Assembling chapters")
        assembled = []
        for n, script in enumerate(scripts, start=1):
            result = self.assembly.assemble_chapter(book, script)
            assembled.append(result)
            db.update_chapter(
                book.id,
                script.chapter,
                audio_path=str(result.path),
                duration_sec=round(result.duration, 2),
            )
            db.update_job(job.id, progress_done=n)
            self.reporter.advance(n, len(scripts), f"chapter {script.chapter} assembled")
        return assembled


def _select(chapters: list[ParsedChapter], wanted: list[int] | None) -> list[ParsedChapter]:
    if not wanted:
        return list(chapters)
    by_index = {c.index: c for c in chapters}
    unknown = [i for i in wanted if i not in by_index]
    if unknown:
        raise AudioBooksError(f"chapter(s) {unknown} do not exist (book has {len(chapters)})")
    return [by_index[i] for i in sorted(set(wanted))]

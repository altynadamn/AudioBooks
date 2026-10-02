"""Local HTTP API (FastAPI). Thin layer over the same services the CLI uses.

Jobs run on a single background worker thread: the GPU can only serve one heavy stage
at a time, so jobs are queued rather than run in parallel.
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from audiobooks import __version__
from audiobooks.config import get_settings
from audiobooks.errors import AudioBooksError, NotFoundError, UnsupportedFormatError
from audiobooks.models.book import Book, Chapter
from audiobooks.models.character import Character
from audiobooks.models.job import GenerationMode, JobKind, JobOptions, ProcessingJob
from audiobooks.models.script import ChapterScript
from audiobooks.parsers import SUPPORTED_EXTENSIONS, book_extension
from audiobooks.services.analysis import AnalysisService
from audiobooks.services.books import BookService
from audiobooks.services.context import AppContext
from audiobooks.services.pipeline import Pipeline
from audiobooks.utils import sanitize_filename
from audiobooks.voices.registry import CharacterRegistry

log = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    chapters: list[int] | None = None
    heuristic: bool = False


class GenerateRequest(BaseModel):
    mode: GenerationMode = GenerationMode.CAST
    chapters: list[int] | None = None
    use_llm: bool = Field(default=False, description="Simple mode: still run LLM analysis")
    build_m4b: bool = True


class VoiceAssignment(BaseModel):
    voice_id: str


class JobRunner:
    """Single-worker queue so heavy GPU stages never overlap."""

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audiobooks-job")

    def submit(self, job_id: str) -> None:
        self.executor.submit(self._run, job_id)

    def _run(self, job_id: str) -> None:
        try:
            Pipeline(self.ctx).run(job_id)
        except Exception:  # state is recorded in the jobs table by the pipeline
            log.exception("job %s failed", job_id)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


def _ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def _runner(request: Request) -> JobRunner:
    return request.app.state.runner


router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    return {
        "status": "ok",
        "version": __version__,
        "voices_available": len(ctx.library.available()),
        "tts_backend": ctx.settings.tts_backend,
        "llm_base_url": ctx.settings.llama_base_url,
    }


@router.post("/books", status_code=201)
async def upload_book(request: Request, file: UploadFile = File(...)) -> Book:
    ctx = _ctx(request)
    name = sanitize_filename(file.filename or "book")
    if book_extension(name) not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"unsupported file type; supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    limit = ctx.settings.max_upload_mb * 1024 * 1024
    upload_dir = ctx.settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"{uuid.uuid4().hex[:8]}_{name}"
    size = 0
    try:
        with target.open("wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"file exceeds {ctx.settings.max_upload_mb} MB")
                out.write(chunk)
        return BookService(ctx).import_book(target)
    finally:
        target.unlink(missing_ok=True)  # the book keeps its own copy in output/<id>/source


@router.get("/books")
def list_books(request: Request) -> list[Book]:
    return _ctx(request).db.list_books()


@router.get("/books/{book_id}")
def get_book(request: Request, book_id: str) -> dict[str, Any]:
    ctx = _ctx(request)
    book = ctx.db.get_book(book_id)
    layout = ctx.layout(book_id)
    return {
        "book": book,
        "chapters": ctx.db.list_chapters(book_id),
        "jobs": ctx.db.list_jobs(book_id)[:10],
        "audiobook_ready": layout.m4b_file.is_file(),
    }


@router.post("/books/{book_id}/analyze", status_code=202)
def analyze_book(request: Request, book_id: str, body: AnalyzeRequest) -> ProcessingJob:
    ctx = _ctx(request)
    mode = GenerationMode.SIMPLE if body.heuristic else GenerationMode.CAST
    job = Pipeline(ctx).create_job(
        book_id, JobKind.ANALYZE, JobOptions(mode=mode, chapters=body.chapters)
    )
    _runner(request).submit(job.id)
    return job


@router.post("/books/{book_id}/generate", status_code=202)
def generate_book(request: Request, book_id: str, body: GenerateRequest) -> ProcessingJob:
    ctx = _ctx(request)
    options = JobOptions(
        mode=body.mode, chapters=body.chapters, force_llm=body.use_llm, build_m4b=body.build_m4b
    )
    job = Pipeline(ctx).create_job(book_id, JobKind.GENERATE, options)
    _runner(request).submit(job.id)
    return job


@router.get("/books/{book_id}/characters")
def get_characters(request: Request, book_id: str) -> list[Character]:
    ctx = _ctx(request)
    ctx.db.get_book(book_id)
    registry = CharacterRegistry.load(ctx.layout(book_id).registry_file, book_id)
    return sorted(registry.characters.values(), key=lambda c: -c.line_count)


@router.put("/books/{book_id}/characters/{character_id}/voice")
def assign_voice(
    request: Request, book_id: str, character_id: str, body: VoiceAssignment
) -> Character:
    ctx = _ctx(request)
    ctx.db.get_book(book_id)
    ctx.library.get(body.voice_id)
    layout = ctx.layout(book_id)
    registry = CharacterRegistry.load(layout.registry_file, book_id)
    character = registry.get(character_id)
    if character is None:
        raise NotFoundError(f"character {character_id!r} not found")
    character.voice_id = body.voice_id
    character.voice_locked = True
    registry.save(layout.registry_file)
    return character


@router.get("/books/{book_id}/chapters")
def get_chapters(request: Request, book_id: str) -> list[Chapter]:
    ctx = _ctx(request)
    ctx.db.get_book(book_id)
    return ctx.db.list_chapters(book_id)


@router.get("/books/{book_id}/chapters/{chapter}/script")
def get_script(request: Request, book_id: str, chapter: int) -> ChapterScript:
    ctx = _ctx(request)
    ctx.db.get_chapter(book_id, chapter)
    script = AnalysisService(ctx).load_script(ctx.layout(book_id), chapter)
    if script is None:
        raise NotFoundError(f"chapter {chapter} has not been analyzed yet")
    return script


@router.get("/books/{book_id}/chapters/{chapter}/audio")
def get_chapter_audio(request: Request, book_id: str, chapter: int) -> FileResponse:
    ctx = _ctx(request)
    ctx.db.get_chapter(book_id, chapter)
    path = ctx.layout(book_id).chapter_audio_file(chapter, ctx.settings.audio_format)
    if not path.is_file():
        raise NotFoundError(f"chapter {chapter} audio has not been generated yet")
    return FileResponse(path, filename=path.name)


@router.get("/books/{book_id}/audiobook")
def get_audiobook(request: Request, book_id: str) -> FileResponse:
    ctx = _ctx(request)
    ctx.db.get_book(book_id)
    path = ctx.layout(book_id).m4b_file
    if not path.is_file():
        raise NotFoundError("the audiobook has not been assembled yet")
    return FileResponse(path, filename=path.name, media_type="audio/mp4")


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> ProcessingJob:
    return _ctx(request).db.get_job(job_id)


@router.post("/jobs/{job_id}/resume", status_code=202)
def resume_job(request: Request, job_id: str) -> ProcessingJob:
    job = _ctx(request).db.get_job(job_id)
    _runner(request).submit(job.id)
    return job


@router.get("/voices")
def list_voices(request: Request) -> list[dict[str, Any]]:
    ctx = _ctx(request)
    return [
        {**v.model_dump(mode="json"), "available": ctx.library.is_available(v)}
        for v in ctx.library.all()
    ]


def create_app(ctx: AppContext | None = None) -> FastAPI:
    ctx = ctx or AppContext.from_settings(get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        yield
        app.state.runner.shutdown()

    app = FastAPI(title="AudioBooks", version=__version__, lifespan=lifespan)
    app.state.ctx = ctx
    app.state.runner = JobRunner(ctx)
    app.include_router(router)

    @app.exception_handler(AudioBooksError)
    async def _handle(_request: Request, exc: AudioBooksError) -> JSONResponse:
        status = 404 if isinstance(exc, NotFoundError) else 400
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    return app


__all__ = ["create_app"]

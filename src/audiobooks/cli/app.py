"""Command-line interface: ``python -m audiobooks <command>``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from audiobooks.config import Settings, get_settings
from audiobooks.errors import AudioBooksError, NotFoundError
from audiobooks.models.book import Book
from audiobooks.models.character import AgeGroup, Gender
from audiobooks.models.job import GenerationMode, JobKind, JobOptions, JobStage
from audiobooks.models.voice import Voice, VoiceSource
from audiobooks.services.books import BookService
from audiobooks.services.context import AppContext
from audiobooks.services.pipeline import Pipeline
from audiobooks.voices.registry import CharacterRegistry

app = typer.Typer(
    help="AudioBooks - local AI audiobook generator (LLM speaker analysis + Qwen3-TTS).",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
voices_app = typer.Typer(help="Manage the voice library.", no_args_is_help=True)
app.add_typer(voices_app, name="voices")

console = Console()


# --------------------------------------------------------------------------- helpers
def _setup(verbose: bool = False) -> AppContext:
    settings = get_settings()
    level = logging.DEBUG if verbose else getattr(logging, settings.log_level.upper(), 20)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=False)],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "qwen_tts", "transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return AppContext.from_settings(settings)


def _resolve_book(ctx: AppContext, ref: str) -> Book:
    """Accept a book id or a path to a book file (imported on first use)."""
    path = Path(ref)
    if path.is_file():
        return BookService(ctx).import_book(path)
    try:
        return ctx.db.get_book(ref)
    except NotFoundError:
        raise NotFoundError(f"{ref!r} is neither an existing file nor a known book id") from None


def _parse_chapters(value: str | None) -> list[int] | None:
    """'1,3,5-7' -> [1, 3, 5, 6, 7]"""
    if not value:
        return None
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            result.extend(range(int(a), int(b) + 1))
        elif part:
            result.append(int(part))
    return result


class RichReporter:
    def __init__(self, progress: Progress) -> None:
        self.progress = progress
        self.task: int | None = None
        self._desc = ""

    def stage(self, stage: JobStage, total: int, description: str) -> None:
        if self.task is not None:
            self.progress.update(self.task, description=f"[green]done[/] {self._desc}")
        self._desc = description
        self.task = self.progress.add_task(description, total=max(total, 1), status="")

    def advance(self, done: int, total: int, message: str = "") -> None:
        if self.task is not None:
            self.progress.update(self.task, completed=done, total=max(total, 1), status=message)


def _run_job(ctx: AppContext, job_id: str) -> None:
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("[dim]{task.fields[status]}"),
        console=console,
    )
    with progress:
        job = Pipeline(ctx, RichReporter(progress)).run(job_id)
    console.print(f"[bold green]Job {job.id} finished.[/]")
    _print_outputs(ctx, job.book_id)


def _print_outputs(ctx: AppContext, book_id: str) -> None:
    layout = ctx.layout(book_id)
    table = Table(title="Chapters", show_lines=False)
    for col in ("#", "Title", "Analysis", "Audio", "Duration"):
        table.add_column(col)
    for ch in ctx.db.list_chapters(book_id):
        table.add_row(
            str(ch.index),
            ch.title[:50],
            f"{ch.analysis_status} {ch.analysis_method}".strip(),
            Path(ch.audio_path).name if ch.audio_path else "-",
            f"{ch.duration_sec / 60:.1f} min" if ch.duration_sec else "-",
        )
    console.print(table)
    console.print(f"Output folder: [cyan]{layout.root.resolve()}[/]")
    if layout.m4b_file.is_file():
        console.print(f"Audiobook: [cyan]{layout.m4b_file.resolve()}[/]")


def _fail(exc: AudioBooksError) -> None:
    console.print(f"[bold red]Error:[/] {exc}")
    raise typer.Exit(code=1)


# -------------------------------------------------------------------------- commands
@app.command()
def inspect(
    path: Annotated[Path, typer.Argument(help="Book file (txt, pdf, epub, fb2)")],
) -> None:
    """Parse a book and show its chapters without importing it."""
    ctx = _setup()
    try:
        book = BookService(ctx).inspect(path)
    except AudioBooksError as exc:
        _fail(exc)
        return
    console.print(f"[bold]{book.title}[/] {('- ' + book.author) if book.author else ''}")
    console.print(
        f"format: {book.source_format}, language: {book.language or '?'}, "
        f"chapters: {len(book.chapters)}, characters: {book.char_count:,}"
    )
    table = Table()
    for col in ("#", "Title", "Paragraphs", "Chars", "Starts with"):
        table.add_column(col)
    for ch in book.chapters:
        first = ch.paragraphs[0][:60] if ch.paragraphs else ""
        table.add_row(
            str(ch.index), ch.title[:50], str(len(ch.paragraphs)), f"{ch.char_count:,}", first
        )
    console.print(table)
    for w in book.warnings:
        console.print(f"[yellow]warning:[/] {w}")


@app.command(name="import")
def import_(path: Annotated[Path, typer.Argument(help="Book file")]) -> None:
    """Register a book and print its id."""
    ctx = _setup()
    try:
        book = BookService(ctx).import_book(path)
    except AudioBooksError as exc:
        _fail(exc)
        return
    console.print(
        f"Imported [bold]{book.title}[/] as [cyan]{book.id}[/] ({book.chapter_count} chapters)"
    )


@app.command()
def analyze(
    book: Annotated[str, typer.Argument(help="Book file or book id")],
    chapters: Annotated[str | None, typer.Option(help="e.g. '1,3,5-7'")] = None,
    heuristic: Annotated[
        bool, typer.Option(help="Skip the LLM: typography-only dialogue detection")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run LLM analysis: segments, speakers, emotions, Character Registry, casting."""
    ctx = _setup(verbose)
    try:
        b = _resolve_book(ctx, book)
        mode = GenerationMode.SIMPLE if heuristic else GenerationMode.CAST
        job = Pipeline(ctx).create_job(
            b.id, JobKind.ANALYZE, JobOptions(mode=mode, chapters=_parse_chapters(chapters))
        )
        console.print(f"Job [cyan]{job.id}[/] (resume with: python -m audiobooks resume {job.id})")
        _run_job(ctx, job.id)
        if not heuristic:
            _print_characters(ctx, b.id)
    except AudioBooksError as exc:
        _fail(exc)


@app.command()
def generate(
    book: Annotated[str, typer.Argument(help="Book file or book id")],
    mode: Annotated[GenerationMode, typer.Option(help="simple = one narrator; cast = AI cast")] = (
        GenerationMode.CAST
    ),
    chapters: Annotated[str | None, typer.Option(help="e.g. '1,3,5-7'")] = None,
    use_llm: Annotated[
        bool, typer.Option("--llm", help="Simple mode: still run LLM analysis")
    ] = False,
    m4b: Annotated[bool, typer.Option(help="Build the final .m4b audiobook")] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Full pipeline: analyze -> cast voices -> synthesize -> assemble chapters (+ M4B)."""
    ctx = _setup(verbose)
    try:
        b = _resolve_book(ctx, book)
        options = JobOptions(
            mode=mode, chapters=_parse_chapters(chapters), force_llm=use_llm, build_m4b=m4b
        )
        job = Pipeline(ctx).create_job(b.id, JobKind.GENERATE, options)
        console.print(
            f"Book [cyan]{b.id}[/], job [cyan]{job.id}[/], mode [bold]{mode.value}[/] "
            f"(resume with: python -m audiobooks resume {job.id})"
        )
        _run_job(ctx, job.id)
    except AudioBooksError as exc:
        _fail(exc)


@app.command()
def resume(
    job_id: Annotated[str, typer.Argument(help="Job id printed by analyze/generate")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Re-run a job; finished chunks, segments and chapters are reused."""
    ctx = _setup(verbose)
    try:
        job = ctx.db.get_job(job_id)
        console.print(f"Resuming job [cyan]{job.id}[/] ({job.kind.value}, {job.status.value})")
        _run_job(ctx, job.id)
    except AudioBooksError as exc:
        _fail(exc)


@app.command()
def status(
    book: Annotated[str | None, typer.Argument(help="Book id (omit to list)")] = None,
) -> None:
    """List books, or show chapter state and jobs of one book."""
    ctx = _setup()
    if book is None:
        table = Table(title="Books")
        for col in ("Id", "Title", "Format", "Chapters"):
            table.add_column(col)
        for b in ctx.db.list_books():
            table.add_row(b.id, b.title[:50], b.source_format, str(b.chapter_count))
        console.print(table)
        return
    try:
        b = _resolve_book(ctx, book)
    except AudioBooksError as exc:
        _fail(exc)
        return
    _print_outputs(ctx, b.id)
    table = Table(title="Jobs")
    for col in ("Id", "Kind", "Mode", "Status", "Stage", "Progress", "Error"):
        table.add_column(col)
    for j in ctx.db.list_jobs(b.id)[:10]:
        table.add_row(
            j.id, j.kind.value, j.options.mode.value, j.status.value, j.stage.value,
            f"{j.progress_done}/{j.progress_total}", j.error[:60],
        )  # fmt: skip
    console.print(table)


def _print_characters(ctx: AppContext, book_id: str) -> None:
    registry = CharacterRegistry.load(ctx.layout(book_id).registry_file, book_id)
    table = Table(title="Character Registry")
    for col in ("Id", "Name", "Aliases", "Gender", "Age", "Lines", "Voice", "Possibly same as"):
        table.add_column(col)
    for c in sorted(registry.characters.values(), key=lambda c: -c.line_count):
        voice = (c.voice_id or "-") + (" (pinned)" if c.voice_locked else "")
        table.add_row(
            c.id, c.name, ", ".join(c.aliases)[:40], c.gender.value, c.age_group.value,
            str(c.line_count), voice, ", ".join(c.possibly_same_as),
        )  # fmt: skip
    console.print(table)


@app.command()
def characters(
    book: Annotated[str, typer.Argument(help="Book id or file")],
    assign: Annotated[
        list[str] | None,
        typer.Option(help="Pin a voice: CHARACTER=VOICE (repeatable)"),
    ] = None,
) -> None:
    """Show the Character Registry; optionally pin voices to characters."""
    ctx = _setup()
    try:
        b = _resolve_book(ctx, book)
        layout = ctx.layout(b.id)
        registry = CharacterRegistry.load(layout.registry_file, b.id)
        for item in assign or []:
            char_id, _, voice_id = item.partition("=")
            character = registry.get(char_id.strip())
            if character is None:
                raise NotFoundError(f"character {char_id!r} not in registry")
            ctx.library.get(voice_id.strip())  # validates
            character.voice_id = voice_id.strip()
            character.voice_locked = True
            console.print(f"{character.id} -> {character.voice_id}")
        if assign:
            registry.save(layout.registry_file)
        _print_characters(ctx, b.id)
    except AudioBooksError as exc:
        _fail(exc)


@app.command()
def doctor() -> None:
    """Check the local environment: GPU, FFmpeg, LLM server, voices."""
    ctx = _setup()
    s: Settings = ctx.settings
    ok = "[green]OK[/]"
    console.print(f"Python {sys.version.split()[0]}")
    try:
        import torch

        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else "-"
        free = f"{torch.cuda.mem_get_info()[0] / 1e9:.1f} GB free" if cuda else ""
        console.print(f"PyTorch {torch.__version__}, CUDA: {cuda} {gpu} {free}")
    except ImportError:
        console.print("[yellow]PyTorch not installed[/] (needed for Qwen3-TTS)")
    try:
        import qwen_tts  # noqa: F401

        console.print(f"qwen-tts: {ok}")
    except ImportError:
        console.print("[yellow]qwen-tts not installed[/]")
    from audiobooks.audio.ffmpeg import FFmpeg

    try:
        console.print(f"FFmpeg: {ok} {FFmpeg.from_config(s.ffmpeg_path).version()}")
    except AudioBooksError as exc:
        console.print(f"FFmpeg: [red]{exc}[/]")
    provider = ctx.llm_factory(s)
    healthy = provider.health()
    provider.close()
    console.print(
        f"LLM {s.llama_base_url}: {ok if healthy else '[yellow]not reachable[/]'}"
        + ("" if healthy or not s.llm_auto_manage else " (auto-managed: started on demand)")
    )
    available = ctx.library.available()
    console.print(f"Voices: {len(available)} available of {len(ctx.library.all())}")
    if s.narrator_voice not in ctx.library:
        console.print(f"[red]Narrator voice {s.narrator_voice!r} is not in the library[/]")
    for err in ctx.library.errors:
        console.print(f"[yellow]voice metadata: {err}[/]")


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
) -> None:
    """Start the local HTTP API (FastAPI)."""
    _setup()
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn is not installed:[/] pip install -e .[api]")
        raise typer.Exit(1) from None
    uvicorn.run("audiobooks.api.app:create_app", factory=True, host=host, port=port)


# ---------------------------------------------------------------------------- voices
@voices_app.command("list")
def voices_list() -> None:
    """List voices in the library."""
    ctx = _setup()
    table = Table(title=f"Voice library ({ctx.library.root})")
    for col in ("Id", "Source", "Gender", "Age", "Roles", "Style", "Available"):
        table.add_column(col)
    for v in ctx.library.all():
        table.add_row(
            v.id, v.source.value, v.gender.value, v.age_group.value, ", ".join(v.roles),
            ", ".join(v.style), "yes" if ctx.library.is_available(v) else "[red]no[/]",
        )  # fmt: skip
    console.print(table)


@voices_app.command("add")
def voices_add(
    voice_id: Annotated[str, typer.Argument(help="lowercase id, e.g. female_01")],
    audio: Annotated[Path, typer.Argument(help="Reference recording (10-20 s, clean)")],
    text: Annotated[str, typer.Option(help="Exact transcript of the recording")],
    name: Annotated[str | None, typer.Option()] = None,
    gender: Annotated[Gender, typer.Option()] = Gender.UNKNOWN,
    age: Annotated[AgeGroup, typer.Option()] = AgeGroup.ADULT,
    style: Annotated[list[str] | None, typer.Option(help="repeatable")] = None,
    narrator: Annotated[bool, typer.Option(help="Also usable as narrator")] = False,
    license_note: Annotated[str, typer.Option("--license", help="Where the clip comes from")] = "",
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    """Add a cloned voice from a reference recording."""
    ctx = _setup()
    try:
        voice = Voice(
            id=voice_id,
            name=name or voice_id,
            source=VoiceSource.REFERENCE,
            gender=gender,
            age_group=age,
            style=style or [],
            roles=["narrator", "character"] if narrator else ["character"],
            reference_audio=f"references/{voice_id}{audio.suffix.lower()}",
            reference_text=text,
            license=license_note,
        )
        voice = ctx.library.add_reference_voice(voice, audio, overwrite=overwrite)
    except (AudioBooksError, ValueError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(1) from None
    console.print(f"Added voice [cyan]{voice.id}[/] -> {voice.reference_audio}")


def main() -> None:
    try:
        app()
    except AudioBooksError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        sys.exit(1)

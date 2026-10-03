"""Chapter analysis: text -> validated ``ChapterScript`` (+ Character Registry updates).

Two methods:
* ``heuristic`` - segmenter only (dialogue vs narration by typography, no speakers);
  used by SIMPLE mode, needs no LLM.
* ``llm`` - the segmenter's spans are labeled chunk by chunk by the LLM. Each chunk's
  validated response is cached on disk, so an interrupted analysis resumes where it
  stopped and re-running never re-queries finished chunks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from audiobooks.errors import LLMError
from audiobooks.llm.base import LLMProvider
from audiobooks.llm.prompts import PROMPT_VERSION
from audiobooks.llm.schemas import ChunkAnalysis, ChunkRequest, SpanInput
from audiobooks.models.book import Book, ParsedChapter
from audiobooks.models.script import (
    NARRATOR,
    UNKNOWN_SPEAKER,
    AudiobookSegment,
    ChapterScript,
    SegmentType,
)
from audiobooks.services.context import AppContext
from audiobooks.storage.layout import BookLayout
from audiobooks.text.attribution import apply_speech_tags
from audiobooks.text.chunker import Chunk, chunk_spans
from audiobooks.text.segmenter import Span, segment_paragraphs
from audiobooks.utils import read_json, sha256_text, write_json
from audiobooks.voices.registry import CharacterRegistry, ground_name

log = logging.getLogger(__name__)

HEURISTIC = "heuristic"
LLM = "llm"

ChunkProgress = Callable[[int, int], None]


def chapter_paragraphs(chapter: ParsedChapter) -> list[str]:
    """Paragraphs to voice; a real chapter title is read first, as in printed audiobooks."""
    if chapter.has_title and chapter.title:
        return [chapter.title, *chapter.paragraphs]
    return list(chapter.paragraphs)


class AnalysisService:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    def source_hash(self, chapter: ParsedChapter) -> str:
        return sha256_text(
            "\n".join(chapter_paragraphs(chapter)), str(self.ctx.settings.segment_max_chars)
        )[:16]

    def load_script(self, layout: BookLayout, chapter: int) -> ChapterScript | None:
        path = layout.script_file(chapter)
        if not path.is_file():
            return None
        return ChapterScript.model_validate(read_json(path))

    def existing_script(
        self, layout: BookLayout, chapter: ParsedChapter, method: str
    ) -> ChapterScript | None:
        """A stored script that is still valid for ``method`` (llm satisfies heuristic too)."""
        script = self.load_script(layout, chapter.index)
        if script is None or script.source_hash != self.source_hash(chapter):
            return None
        if method == LLM and script.analysis != LLM:
            return None
        if script.analysis == LLM and script.prompt_version != PROMPT_VERSION:
            # labelled by an older prompt: redo it (and don't reuse it for SIMPLE mode either,
            # its speakers may not match the current registry)
            return None
        return script

    def analyze_chapter(
        self,
        book: Book,
        chapter: ParsedChapter,
        *,
        method: str,
        registry: CharacterRegistry,
        llm: LLMProvider | None = None,
        on_chunk: ChunkProgress | None = None,
    ) -> ChapterScript:
        layout = self.ctx.layout(book.id)
        paragraphs = chapter_paragraphs(chapter)
        spans = segment_paragraphs(paragraphs, max_chars=self.ctx.settings.segment_max_chars)
        if method == HEURISTIC:
            segments = [_heuristic_segment(s) for s in spans]
        else:
            if llm is None:
                raise LLMError("LLM analysis requested but no LLM provider is available")
            segments = self._llm_segments(
                book, chapter, spans, paragraphs, registry, llm, layout, on_chunk
            )
            self._consolidate(book, chapter.index, segments, registry, llm)
            registry.save(layout.registry_file)
        script = ChapterScript(
            book_id=book.id,
            chapter=chapter.index,
            title=chapter.title,
            analysis=method,
            source_hash=self.source_hash(chapter),
            prompt_version=PROMPT_VERSION if method == LLM else "",
            segments=segments,
        )
        write_json(layout.script_file(chapter.index), script)
        return script

    # ------------------------------------------------------------------ llm path
    @staticmethod
    def _consolidate(
        book: Book,
        chapter: int,
        segments: list[AudiobookSegment],
        registry: CharacterRegistry,
        llm: LLMProvider,
    ) -> None:
        """Merge duplicate cast entries (description first, name later).

        Candidate pairs are chosen deterministically (shared description keywords,
        ``possibly_same_as``); the LLM only answers "same person?" per pair. At least one
        entry of every pair was first seen in this chapter, so settled characters of
        earlier chapters are never merged with each other.
        """
        new_ids = [c.id for c in registry.characters.values() if c.first_chapter == chapter]
        pairs = registry.duplicate_candidates(new_ids) if new_ids else []
        if not pairs:
            return
        cards = [tuple(registry.cards([a.id, b.id])) for a, b in pairs]
        try:
            verdicts = llm.same_person(cards)  # type: ignore[arg-type]
        except LLMError as exc:
            log.warning("chapter %d: duplicate check skipped: %s", chapter, exc)
            return
        for v in verdicts:
            if not v.same_person:
                continue
            a, b = pairs[v.pair]
            a, b = registry.get(a.id), registry.get(b.id)  # may already be merged
            if a is None or b is None or a.id == b.id:
                continue
            keep, other = registry.keeper(a, b)
            if registry.merge(keep.id, other.id):
                log.info("chapter %d: %s is %s (%s)", chapter, other.id, keep.id, v.reason[:80])
        for seg in segments:
            if seg.type is SegmentType.DIALOGUE:
                seg.speaker = registry.canonical(seg.speaker)

    def _llm_segments(
        self,
        book: Book,
        chapter: ParsedChapter,
        spans: list[Span],
        paragraphs: list[str],
        registry: CharacterRegistry,
        llm: LLMProvider,
        layout: BookLayout,
        on_chunk: ChunkProgress | None,
    ) -> list[AudiobookSegment]:
        settings = self.ctx.settings
        chunks = chunk_spans(
            spans,
            paragraphs,
            max_chars=settings.chunk_max_chars,
            context_paragraphs=settings.chunk_context_paragraphs,
        )
        segments: list[AudiobookSegment] = []
        recent: list[str] = []
        for chunk in chunks:
            analysis, fresh = self._analyze_chunk(
                book, chapter, chunk, registry, recent, llm, layout
            )
            chunk_segments = self._apply(chunk, analysis, registry, chapter.index)
            segments.extend(chunk_segments)
            if fresh:  # cached chunks were already counted when first analyzed
                registry.count_lines(
                    [s.speaker for s in chunk_segments if s.type is SegmentType.DIALOGUE]
                )
            registry.save(layout.registry_file)  # persist progress after every chunk
            for s in chunk_segments:
                if s.type is SegmentType.DIALOGUE and s.speaker != UNKNOWN_SPEAKER:
                    if s.speaker in recent:
                        recent.remove(s.speaker)
                    recent.append(s.speaker)
            recent = recent[-4:]
            if on_chunk:
                on_chunk(chunk.index, len(chunks))
        return segments

    def _analyze_chunk(
        self,
        book: Book,
        chapter: ParsedChapter,
        chunk: Chunk,
        registry: CharacterRegistry,
        recent: list[str],
        llm: LLMProvider,
        layout: BookLayout,
    ) -> tuple[ChunkAnalysis, bool]:
        """Return the chunk analysis and whether it was freshly produced by the LLM."""
        key = sha256_text(
            PROMPT_VERSION,
            self.ctx.settings.llama_model,
            *(f"{s.index}:{s.hint}:{s.text}" for s in chunk.spans),
        )
        cache = layout.chunk_cache_file(chapter.index, chunk.index)
        if cache.is_file():
            cached = read_json(cache)
            if cached.get("key") == key:
                log.debug("chapter %d chunk %d: cached", chapter.index, chunk.index)
                return ChunkAnalysis.model_validate(cached["analysis"]), False

        request = ChunkRequest(
            book_title=book.title,
            chapter_title=chapter.title,
            language=book.language,
            spans=[
                SpanInput(id=s.index, hint=s.hint.value, text=s.text, paragraph=s.paragraph)
                for s in chunk.spans
            ],
            context=chunk.context,
            known_characters=registry.known_characters(),
            recent_speakers=list(recent),
        )
        log.info(
            "chapter %d chunk %d: analyzing %d spans (%d chars)",
            chapter.index, chunk.index, len(chunk.spans), chunk.char_count,
        )  # fmt: skip
        analysis = llm.analyze_chunk(request)
        write_json(cache, {"key": key, "analysis": analysis.model_dump(mode="json")})
        return analysis, True

    @staticmethod
    def _apply(
        chunk: Chunk, analysis: ChunkAnalysis, registry: CharacterRegistry, chapter: int
    ) -> list[AudiobookSegment]:
        chunk_ids: dict[str, str] = {}
        source_text = " ".join(s.text for s in chunk.spans) + " " + " ".join(chunk.context)
        for obs in analysis.characters:
            obs.name, obs.aliases = ground_name(obs.name, obs.aliases, source_text)
            canonical = registry.observe(obs, chapter)
            chunk_ids[obs.id] = canonical
        labels = {label.id: label for label in analysis.segments}

        segments = []
        missing = 0
        for span in chunk.spans:
            label = labels.get(span.index)
            if label is None:
                missing += 1
                segments.append(_heuristic_segment(span))
                continue
            if label.type is SegmentType.NARRATION:
                speaker = NARRATOR
            else:
                speaker = registry.resolve_speaker(label.speaker, chunk_ids)
            segments.append(
                AudiobookSegment(
                    index=span.index,
                    type=label.type,
                    speaker=speaker,
                    text=span.text,  # always the original text, never the LLM's
                    emotion=label.emotion,
                    paragraph=span.paragraph,
                )
            )
        if missing:
            log.warning(
                "chapter %d chunk %d: %d span(s) unlabeled by the LLM; used heuristics",
                chapter, chunk.index, missing,
            )  # fmt: skip
        hints = {s.index: s.hint for s in chunk.spans}
        for idx, old, new in apply_speech_tags(segments, registry, hints):
            log.debug("span %d: speech tag names %s (LLM said %s)", idx, new, old)
        return segments


def _heuristic_segment(span: Span) -> AudiobookSegment:
    dialogue = span.hint is SegmentType.DIALOGUE
    return AudiobookSegment(
        index=span.index,
        type=span.hint,
        speaker=UNKNOWN_SPEAKER if dialogue else NARRATOR,
        text=span.text,
        paragraph=span.paragraph,
    )

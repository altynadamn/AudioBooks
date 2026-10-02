"""Segment synthesis with deterministic file names and content-addressed caching.

Each chapter audio folder has a ``manifest.json`` mapping segment index -> cache key. The
key covers everything that changes the audio (text, voice + reference audio hash, TTS
model, language, emotion when it is used). A segment is regenerated only if its WAV is
missing or its key changed, so interrupted runs resume and edits re-render only what
changed.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from audiobooks.models.book import Book
from audiobooks.models.job import GenerationMode
from audiobooks.models.script import ChapterScript, Emotion
from audiobooks.models.voice import Voice
from audiobooks.services.context import AppContext
from audiobooks.storage.layout import BookLayout
from audiobooks.tts.base import SynthesisRequest, TTSBackend, tts_language
from audiobooks.utils import read_json, sha256_text, write_json
from audiobooks.voices.casting import resolve_voice
from audiobooks.voices.registry import CharacterRegistry

log = logging.getLogger(__name__)

SegmentProgress = Callable[[int, int], None]


@dataclass(frozen=True)
class WorkItem:
    chapter: int
    index: int
    text: str
    voice: Voice
    emotion: Emotion
    key: str
    model_tag: str
    path: Path
    reuse_from: Path | None = None  # identical audio already rendered at another index


class SynthesisService:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    def cache_key(
        self, text: str, voice: Voice, model_tag: str, language: str, emotion: Emotion
    ) -> str:
        emotion_part = emotion.value if self.ctx.settings.tts_use_instruct else ""
        return sha256_text(
            "seg-v1", text, self.ctx.library.fingerprint(voice), model_tag, language, emotion_part
        )[:24]

    def plan(
        self,
        book: Book,
        scripts: list[ChapterScript],
        registry: CharacterRegistry,
        mode: GenerationMode,
        backend: TTSBackend,
    ) -> tuple[list[WorkItem], int]:
        """Return pending work items (sorted by model to minimize reloads) and total count."""
        layout = self.ctx.layout(book.id)
        settings = self.ctx.settings
        language = tts_language(settings.tts_language, book.language)
        pending: list[WorkItem] = []
        total = 0
        for script in scripts:
            manifest = self._load_manifest(layout, script.chapter)
            # content-addressed lookup: inserting a paragraph shifts indices, not audio
            by_key = {
                e["key"]: layout.segment_file(script.chapter, int(i))
                for i, e in manifest.items()
                if isinstance(e, dict) and "key" in e
            }
            for seg in script.segments:
                total += 1
                voice_id = resolve_voice(
                    seg,
                    mode=mode,
                    registry=registry,
                    narrator_voice_id=settings.narrator_voice,
                    unknown_voice_id=settings.unknown_speaker_voice,
                )
                voice = self.ctx.library.get(voice_id)
                tag = backend.model_tag(voice)
                key = self.cache_key(seg.text, voice, tag, language, seg.emotion)
                path = layout.segment_file(script.chapter, seg.index)
                entry = manifest.get(str(seg.index))
                if path.is_file() and entry and entry.get("key") == key:
                    continue
                source = by_key.get(key)
                pending.append(
                    WorkItem(
                        script.chapter,
                        seg.index,
                        seg.text,
                        voice,
                        seg.emotion,
                        key,
                        tag,
                        path,
                        reuse_from=source if source is not None and source.is_file() else None,
                    )
                )
        # reused files first (copied before anything is overwritten), then by model
        pending.sort(key=lambda w: (w.reuse_from is None, w.model_tag, w.chapter, w.index))
        return pending, total

    def run(
        self,
        book: Book,
        items: list[WorkItem],
        backend: TTSBackend,
        on_segment: SegmentProgress | None = None,
    ) -> None:
        layout = self.ctx.layout(book.id)
        language = tts_language(self.ctx.settings.tts_language, book.language)
        manifests: dict[int, dict] = {}
        # snapshot reusable audio before any segment file is overwritten
        staged: dict[Path, Path] = {}
        for item in items:
            if item.reuse_from is not None:
                tmp = item.path.with_suffix(".reuse.tmp")
                shutil.copyfile(item.reuse_from, tmp)
                staged[item.path] = tmp
        for n, item in enumerate(items, start=1):
            if item.path in staged:
                os.replace(staged[item.path], item.path)
                info = sf.info(str(item.path))
                duration, rate = info.frames / info.samplerate, info.samplerate
            else:
                clip = backend.synthesize(
                    SynthesisRequest(
                        text=item.text,
                        voice=item.voice,
                        language=language,
                        emotion=item.emotion,
                        seed=int(item.key[:8], 16),
                    )
                )
                _write_wav_atomic(item.path, clip.samples, clip.sample_rate)
                duration, rate = clip.duration, clip.sample_rate
            manifest = manifests.setdefault(item.chapter, self._load_manifest(layout, item.chapter))
            manifest[str(item.index)] = {
                "key": item.key,
                "voice": item.voice.id,
                "file": item.path.name,
                "duration": round(duration, 3),
                "sample_rate": rate,
            }
            write_json(layout.segment_manifest(item.chapter), manifest)
            if on_segment:
                on_segment(n, len(items))

    @staticmethod
    def _load_manifest(layout: BookLayout, chapter: int) -> dict:
        path = layout.segment_manifest(chapter)
        if not path.is_file():
            return {}
        try:
            data = read_json(path)
        except ValueError:
            log.warning("corrupt manifest %s; segments will be regenerated", path)
            return {}
        return data if isinstance(data, dict) else {}


def _write_wav_atomic(path: Path, samples: object, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav.tmp")
    sf.write(tmp, samples, sample_rate, subtype="PCM_16", format="WAV")
    os.replace(tmp, path)

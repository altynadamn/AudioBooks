"""Application context: settings plus factories for heavy components.

Services receive an ``AppContext`` instead of constructing providers themselves, so tests
(and future remote workers) can inject fake or remote LLM/TTS implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from audiobooks.config import Settings
from audiobooks.llm.base import LLMProvider
from audiobooks.storage.db import Database
from audiobooks.storage.layout import BookLayout
from audiobooks.tts.base import TTSBackend
from audiobooks.voices.library import VoiceLibrary


def _default_llm(settings: Settings) -> LLMProvider:
    from audiobooks.llm.openai_provider import OpenAICompatibleProvider

    return OpenAICompatibleProvider(settings)


def _default_tts(settings: Settings, library: VoiceLibrary) -> TTSBackend:
    from audiobooks.tts import create_backend

    return create_backend(settings, library)


@dataclass
class AppContext:
    settings: Settings
    db: Database
    library: VoiceLibrary
    llm_factory: Callable[[Settings], LLMProvider] = field(default=_default_llm)
    tts_factory: Callable[[Settings, VoiceLibrary], TTSBackend] = field(default=_default_tts)
    manage_llm_server: bool | None = None  # None -> settings.llm_auto_manage

    @classmethod
    def from_settings(cls, settings: Settings) -> AppContext:
        return cls(
            settings=settings,
            db=Database(settings.db_path),
            library=VoiceLibrary(
                settings.voice_library_dir, builtin_enabled=settings.tts_builtin_voices
            ),
        )

    def layout(self, book_id: str) -> BookLayout:
        return BookLayout.for_book(self.settings.output_dir, book_id)

    @property
    def should_manage_llm(self) -> bool:
        if self.manage_llm_server is not None:
            return self.manage_llm_server
        return self.settings.llm_auto_manage

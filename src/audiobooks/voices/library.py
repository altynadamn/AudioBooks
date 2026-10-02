"""Reusable voice library stored as JSON metadata + local reference recordings.

voices/
    metadata/<voice_id>.json      one Voice per file (or a JSON list of voices)
    references/<voice_id>.wav     reference recordings (never committed)
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from pydantic import ValidationError

from audiobooks.errors import ConfigError, VoiceNotFoundError
from audiobooks.models.voice import Voice, VoiceSource
from audiobooks.utils import ensure_within, sha256_file, write_json

log = logging.getLogger(__name__)

REFERENCE_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


class VoiceLibrary:
    def __init__(self, root: Path, *, builtin_enabled: bool = True) -> None:
        self.root = root
        self.builtin_enabled = builtin_enabled
        self._voices: dict[str, Voice] = {}
        self._digests: dict[str, str] = {}
        self.errors: list[str] = []
        self.reload()

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def references_dir(self) -> Path:
        return self.root / "references"

    def reload(self) -> None:
        self._voices.clear()
        self.errors.clear()
        if not self.metadata_dir.is_dir():
            return
        for path in sorted(self.metadata_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self.errors.append(f"{path.name}: {exc}")
                continue
            for item in raw if isinstance(raw, list) else [raw]:
                try:
                    voice = Voice.model_validate(item)
                except ValidationError as exc:
                    self.errors.append(f"{path.name}: {exc.errors()[0]['msg']}")
                    continue
                if voice.id in self._voices:
                    self.errors.append(f"{path.name}: duplicate voice id {voice.id!r}")
                    continue
                self._voices[voice.id] = voice
        for err in self.errors:
            log.warning("voice library: %s", err)

    # ------------------------------------------------------------------ queries
    def all(self) -> list[Voice]:
        return sorted(self._voices.values(), key=lambda v: v.id)

    def is_available(self, voice: Voice) -> bool:
        if voice.source is VoiceSource.BUILTIN:
            return self.builtin_enabled
        try:
            return self.reference_path(voice).is_file()
        except ValueError:
            return False

    def available(self) -> list[Voice]:
        return [v for v in self.all() if self.is_available(v)]

    def get(self, voice_id: str) -> Voice:
        try:
            return self._voices[voice_id]
        except KeyError:
            raise VoiceNotFoundError(
                f"voice {voice_id!r} is not in the voice library ({self.metadata_dir})"
            ) from None

    def __contains__(self, voice_id: object) -> bool:
        return voice_id in self._voices

    def reference_path(self, voice: Voice) -> Path:
        if not voice.reference_audio:
            raise ValueError(f"voice {voice.id!r} has no reference audio")
        return ensure_within(self.root, self.root / voice.reference_audio)

    def fingerprint(self, voice: Voice) -> str:
        """Voice identity for audio caching; includes the reference file content hash."""
        digest = ""
        if voice.source is VoiceSource.REFERENCE:
            if voice.id not in self._digests:
                path = self.reference_path(voice)
                self._digests[voice.id] = sha256_file(path) if path.is_file() else "missing"
            digest = self._digests[voice.id]
        return voice.fingerprint(digest)

    # ------------------------------------------------------------------ mutation
    def add_reference_voice(
        self, voice: Voice, audio_file: Path, *, overwrite: bool = False
    ) -> Voice:
        """Copy a reference recording into the library and write its metadata."""
        if voice.source is not VoiceSource.REFERENCE:
            raise ConfigError("add_reference_voice expects a reference voice")
        if audio_file.suffix.lower() not in REFERENCE_EXTENSIONS:
            raise ConfigError(f"unsupported reference audio type {audio_file.suffix}")
        if not audio_file.is_file():
            raise ConfigError(f"reference audio not found: {audio_file}")
        if voice.id in self._voices and not overwrite:
            raise ConfigError(f"voice {voice.id!r} already exists (use overwrite)")
        target = self.references_dir / f"{voice.id}{audio_file.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if audio_file.resolve() != target.resolve():
            shutil.copyfile(audio_file, target)
        voice = voice.model_copy(update={"reference_audio": f"references/{target.name}"})
        write_json(self.metadata_dir / f"{voice.id}.json", voice.model_dump(mode="json"))
        self._voices[voice.id] = voice
        self._digests.pop(voice.id, None)
        return voice

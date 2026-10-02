"""Voice library entries."""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from audiobooks.models.character import AgeGroup, Gender, coerce_enum


class VoiceSource(StrEnum):
    REFERENCE = "reference"  # voice cloning from a reference recording (Qwen3-TTS Base)
    BUILTIN = "builtin"  # named speaker of a CustomVoice model


class Voice(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    source: VoiceSource = VoiceSource.REFERENCE
    gender: Gender = Gender.UNKNOWN
    age_group: AgeGroup = AgeGroup.ADULT
    style: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    roles: list[str] = Field(
        default_factory=lambda: ["character"],
        description="'narrator' and/or 'character'; used by automatic casting.",
    )
    # reference voices
    reference_audio: str | None = Field(
        default=None, description="Path relative to the voice library directory."
    )
    reference_text: str | None = None
    # built-in voices
    builtin_speaker: str | None = None
    license: str = ""
    notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: object) -> object:
        if isinstance(data, dict):
            if "gender" in data:
                data["gender"] = coerce_enum(Gender, data["gender"])
            if "age_group" in data:
                data["age_group"] = coerce_enum(AgeGroup, data["age_group"])
        return data

    @model_validator(mode="after")
    def _check_source_fields(self) -> Voice:
        if self.source is VoiceSource.REFERENCE and not self.reference_audio:
            raise ValueError(f"voice {self.id!r}: reference voices need 'reference_audio'")
        if self.source is VoiceSource.BUILTIN and not self.builtin_speaker:
            raise ValueError(f"voice {self.id!r}: builtin voices need 'builtin_speaker'")
        return self

    def fingerprint(self, audio_digest: str = "") -> str:
        """Stable hash of everything that affects generated audio for this voice."""
        parts = [
            self.id,
            self.source,
            self.reference_text or "",
            self.builtin_speaker or "",
            audio_digest,
        ]
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]

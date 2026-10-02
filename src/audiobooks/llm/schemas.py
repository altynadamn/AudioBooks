"""Schemas for LLM input/output during chunk analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field, field_validator, model_validator

from audiobooks.models.character import AgeGroup, Gender, coerce_enum
from audiobooks.models.script import Emotion, SegmentType, normalize_emotion
from audiobooks.utils import slugify


class CharacterObservation(BaseModel):
    """A character as reported by the LLM for one chunk."""

    id: str
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    gender: Gender = Gender.UNKNOWN
    age_group: AgeGroup = AgeGroup.UNKNOWN
    description: str = ""
    same_as: str | None = Field(
        default=None, description="Existing registry id when the LLM is confident it matches."
    )
    possibly_same_as: list[str] = Field(default_factory=list)

    @field_validator("id", mode="before")
    @classmethod
    def _slug(cls, v: object) -> str:
        return slugify(str(v or ""), fallback="")

    @field_validator("gender", mode="before")
    @classmethod
    def _gender(cls, v: object) -> object:
        return coerce_enum(Gender, v)

    @field_validator("age_group", mode="before")
    @classmethod
    def _age(cls, v: object) -> object:
        return coerce_enum(AgeGroup, v)

    @field_validator("aliases", "possibly_same_as", mode="before")
    @classmethod
    def _list(cls, v: object) -> object:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("same_as", mode="before")
    @classmethod
    def _same_as(cls, v: object) -> object:
        if v in (None, "", "null", "none"):
            return None
        return slugify(str(v), fallback="") or None

    @model_validator(mode="after")
    def _ensure_id(self) -> CharacterObservation:
        if not self.id:
            self.id = slugify(self.name, fallback="character")
        return self


class SpanLabel(BaseModel):
    id: int
    type: SegmentType
    cue: str = Field(default="", description="Text evidence the LLM used for the speaker.")
    speaker: str = "narrator"
    emotion: Emotion = Emotion.NEUTRAL

    @field_validator("type", mode="before")
    @classmethod
    def _type(cls, v: object) -> object:
        text = str(v or "").strip().lower()
        return (
            SegmentType.DIALOGUE
            if text in {"dialogue", "dialog", "speech"}
            else (SegmentType.NARRATION)
        )

    @field_validator("emotion", mode="before")
    @classmethod
    def _emotion(cls, v: object) -> Emotion:
        return normalize_emotion(v)

    @field_validator("cue", mode="before")
    @classmethod
    def _cue(cls, v: object) -> str:
        return str(v or "")[:200]

    @field_validator("speaker", mode="before")
    @classmethod
    def _speaker(cls, v: object) -> str:
        text = str(v or "").strip()
        return text or "unknown"


class ChunkAnalysis(BaseModel):
    characters: list[CharacterObservation] = Field(default_factory=list)
    segments: list[SpanLabel]


@dataclass(frozen=True)
class SpanInput:
    id: int
    hint: str
    text: str
    paragraph: int = 0


@dataclass(frozen=True)
class KnownCharacter:
    id: str
    name: str
    aliases: list[str]
    gender: str
    description: str


@dataclass(frozen=True)
class ChunkRequest:
    """Everything the LLM needs to label one chunk."""

    book_title: str
    chapter_title: str
    language: str
    spans: list[SpanInput]
    context: list[str] = field(default_factory=list)
    known_characters: list[KnownCharacter] = field(default_factory=list)
    recent_speakers: list[str] = field(default_factory=list)


# Hand-written JSON schema (no $refs) for llama-server's grammar-constrained decoding.
CHUNK_ANALYSIS_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "gender": {"type": "string", "enum": [g.value for g in Gender]},
                    "age_group": {"type": "string", "enum": [a.value for a in AgeGroup]},
                    "description": {"type": "string"},
                    "same_as": {"type": ["string", "null"]},
                    "possibly_same_as": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "name", "gender", "age_group"],
            },
        },
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "type": {"type": "string", "enum": [t.value for t in SegmentType]},
                    # generated before "speaker": the model quotes its evidence first
                    "cue": {"type": "string"},
                    "speaker": {"type": "string"},
                    "emotion": {"type": "string", "enum": [e.value for e in Emotion]},
                },
                "required": ["id", "type", "cue", "speaker", "emotion"],
            },
        },
    },
    "required": ["characters", "segments"],
}

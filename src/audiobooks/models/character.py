"""Character and voice-related enums and models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class Gender(StrEnum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class AgeGroup(StrEnum):
    CHILD = "child"
    TEEN = "teen"
    YOUNG_ADULT = "young_adult"
    ADULT = "adult"
    ELDERLY = "elderly"
    UNKNOWN = "unknown"


def coerce_enum(enum_cls: type[StrEnum], value: object) -> StrEnum:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return enum_cls(text)
    except ValueError:
        return enum_cls("unknown")


class Character(BaseModel):
    """A persistent character entry in a book's Character Registry."""

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    aliases: list[str] = Field(default_factory=list)
    gender: Gender = Gender.UNKNOWN
    age_group: AgeGroup = AgeGroup.UNKNOWN
    description: str = ""
    voice_id: str | None = None
    voice_locked: bool = Field(
        default=False, description="Set when a user pinned the voice manually."
    )
    possibly_same_as: list[str] = Field(
        default_factory=list,
        description="Ids of characters that may be the same person (not merged: uncertain).",
    )
    first_chapter: int = 0
    line_count: int = 0

    @field_validator("gender", mode="before")
    @classmethod
    def _gender(cls, v: object) -> StrEnum:
        return coerce_enum(Gender, v)

    @field_validator("age_group", mode="before")
    @classmethod
    def _age(cls, v: object) -> StrEnum:
        return coerce_enum(AgeGroup, v)

    def all_names(self) -> list[str]:
        return [self.name, *self.aliases]


class CharacterRegistryData(BaseModel):
    """Serialized form of ``character_registry.json``."""

    book_id: str
    characters: dict[str, Character] = Field(default_factory=dict)
    merged: dict[str, str] = Field(
        default_factory=dict, description="Ids merged into another character: old -> new."
    )

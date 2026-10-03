"""Structured audiobook script: chapters split into narration/dialogue segments."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

NARRATOR = "narrator"
UNKNOWN_SPEAKER = "unknown"


class SegmentType(StrEnum):
    NARRATION = "narration"
    DIALOGUE = "dialogue"


class Emotion(StrEnum):
    NEUTRAL = "neutral"
    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    AFRAID = "afraid"
    WHISPER = "whisper"
    EXCITED = "excited"
    SERIOUS = "serious"


_EMOTION_SYNONYMS: dict[str, Emotion] = {
    "quiet": Emotion.WHISPER,
    "quietly": Emotion.WHISPER,
    "soft": Emotion.CALM,
    "gentle": Emotion.CALM,
    "tender": Emotion.CALM,
    "joyful": Emotion.HAPPY,
    "cheerful": Emotion.HAPPY,
    "amused": Emotion.HAPPY,
    "smiling": Emotion.HAPPY,
    "sorrowful": Emotion.SAD,
    "melancholic": Emotion.SAD,
    "upset": Emotion.SAD,
    "furious": Emotion.ANGRY,
    "irritated": Emotion.ANGRY,
    "annoyed": Emotion.ANGRY,
    "scared": Emotion.AFRAID,
    "fearful": Emotion.AFRAID,
    "nervous": Emotion.AFRAID,
    "anxious": Emotion.AFRAID,
    "whispering": Emotion.WHISPER,
    "enthusiastic": Emotion.EXCITED,
    "surprised": Emotion.EXCITED,
    "stern": Emotion.SERIOUS,
    "grave": Emotion.SERIOUS,
    "firm": Emotion.SERIOUS,
}


def normalize_emotion(value: object) -> Emotion:
    """Map free-form LLM emotion labels onto the small normalized set."""
    if isinstance(value, Emotion):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return Emotion.NEUTRAL
    try:
        return Emotion(text)
    except ValueError:
        return _EMOTION_SYNONYMS.get(text, Emotion.NEUTRAL)


class AudiobookSegment(BaseModel):
    """One unit of speech. ``text`` is always the author's original text."""

    index: int = Field(ge=0)
    type: SegmentType
    speaker: str = NARRATOR
    text: str = Field(min_length=1)
    emotion: Emotion = Emotion.NEUTRAL
    paragraph: int = Field(default=0, ge=0, description="Paragraph index within the chapter.")

    @field_validator("emotion", mode="before")
    @classmethod
    def _normalize_emotion(cls, v: object) -> Emotion:
        return normalize_emotion(v)


class ChapterScript(BaseModel):
    book_id: str
    chapter: int = Field(ge=1)
    title: str
    analysis: str = Field(description="'llm' or 'heuristic'")
    source_hash: str = Field(default="", description="Hash of the chapter text + settings.")
    prompt_version: str = Field(default="", description="LLM prompt version (llm scripts).")
    segments: list[AudiobookSegment]

    @property
    def speakers(self) -> list[str]:
        seen: dict[str, None] = {}
        for s in self.segments:
            seen.setdefault(s.speaker, None)
        return list(seen)

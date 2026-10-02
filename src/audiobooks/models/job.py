"""Processing job schemas."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class GenerationMode(StrEnum):
    SIMPLE = "simple"  # one narrator voice for everything, no LLM required
    CAST = "cast"  # narrator + persistent per-character voices


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobKind(StrEnum):
    ANALYZE = "analyze"
    GENERATE = "generate"


class JobStage(StrEnum):
    QUEUED = "queued"
    ANALYSIS = "analysis"
    CASTING = "casting"
    SYNTHESIS = "synthesis"
    ASSEMBLY = "assembly"
    FINISHED = "finished"


class JobOptions(BaseModel):
    mode: GenerationMode = GenerationMode.CAST
    chapters: list[int] | None = Field(default=None, description="1-based; None = all")
    force_llm: bool = Field(default=False, description="Use the LLM even in simple mode.")
    build_m4b: bool = True


class ProcessingJob(BaseModel):
    id: str
    book_id: str
    kind: JobKind
    options: JobOptions
    status: JobStatus = JobStatus.PENDING
    stage: JobStage = JobStage.QUEUED
    progress_done: int = 0
    progress_total: int = 0
    message: str = ""
    error: str = ""
    created_at: datetime
    updated_at: datetime

"""Typed application settings loaded from environment variables and ``.env``."""

from __future__ import annotations

import shlex
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Field names map 1:1 to upper-case env variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM (OpenAI-compatible endpoint, normally llama-server) ---------------------
    llama_base_url: str = "http://127.0.0.1:8081/v1"
    llama_model: str = "ornith"
    llama_api_key: str = ""
    llama_timeout: float = 600.0
    llama_temperature: float = 0.0  # deterministic labels
    llama_max_tokens: int = 8192
    llama_json_schema: bool = Field(
        default=True, description="Send response_format=json_schema (grammar-constrained output)."
    )
    llama_disable_thinking: bool = Field(
        default=True, description="Pass chat_template_kwargs.enable_thinking=false."
    )
    llama_max_retries: int = 2

    # --- Optional managed llama-server (start for analysis, stop before TTS) ---------
    llm_auto_manage: bool = False
    llama_server_exe: str = "llama-server"
    llama_model_path: Path | None = None
    llama_mmproj_path: Path | None = None
    llama_server_port: int = 8081
    llama_ctx_size: int = 16384
    llama_gpu_layers: int = 99
    llama_server_extra_args: str = ""
    llama_startup_timeout: float = 180.0

    # --- Vision (scanned pages); experimental ----------------------------------------
    vision_enabled: bool = False

    # --- TTS -------------------------------------------------------------------------
    tts_backend: Literal["qwen", "fake"] = "qwen"
    tts_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    tts_custom_voice_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    tts_device: str = "cuda:0"
    tts_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    tts_language: str = "auto"
    tts_builtin_voices: bool = Field(
        default=True,
        description="Allow casting built-in CustomVoice speakers (needs TTS_CUSTOM_VOICE_MODEL).",
    )
    tts_use_instruct: bool = Field(
        default=False,
        description="Send emotion as an instruction to CustomVoice models that support it.",
    )

    # --- Text processing -------------------------------------------------------------
    chunk_max_chars: int = 4000
    chunk_context_paragraphs: int = 3
    segment_max_chars: int = 400

    # --- Voices / casting ------------------------------------------------------------
    voice_library_dir: Path = Path("voices")
    narrator_voice: str = "narrator_01"
    unknown_speaker_voice: str = ""  # empty -> narrator voice

    # --- Storage ---------------------------------------------------------------------
    output_dir: Path = Path("output")
    data_dir: Path = Path("data")
    max_upload_mb: int = 200

    # --- Audio -----------------------------------------------------------------------
    ffmpeg_path: str = ""
    audio_format: Literal["m4a", "mp3"] = "m4a"
    audio_bitrate: str = "96k"
    audio_sample_rate: int = 44100
    audio_loudnorm: bool = True
    pause_same_paragraph_ms: int = 150
    pause_speaker_change_ms: int = 350
    pause_paragraph_ms: int = 600
    pause_chapter_start_ms: int = 800

    log_level: str = "INFO"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "audiobooks.sqlite3"

    @property
    def llama_server_extra_argv(self) -> list[str]:
        return shlex.split(self.llama_server_extra_args, posix=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

"""Qwen3-TTS backend.

* reference voices -> Base model, ``generate_voice_clone`` (clone prompt cached per voice)
* built-in voices  -> CustomVoice model, ``generate_custom_voice``

Only one Qwen model is kept on the GPU at a time; switching models unloads the previous
one. Callers should group segments by model (see ``SynthesisService``) to avoid churn.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np

from audiobooks.config import Settings
from audiobooks.errors import TTSError
from audiobooks.models.script import Emotion
from audiobooks.models.voice import Voice, VoiceSource
from audiobooks.tts.base import EMOTION_INSTRUCTIONS, AudioClip, SynthesisRequest
from audiobooks.voices.library import VoiceLibrary

log = logging.getLogger(__name__)

MIN_FREE_VRAM_GB = 2.0


class QwenTTSBackend:
    name = "qwen3-tts"

    def __init__(self, settings: Settings, library: VoiceLibrary) -> None:
        self.settings = settings
        self.library = library
        self._model: Any = None
        self._model_id: str | None = None
        self._clone_prompts: dict[str, Any] = {}

    def model_tag(self, voice: Voice) -> str:
        return self._model_for(voice)

    def _model_for(self, voice: Voice) -> str:
        if voice.source is VoiceSource.BUILTIN:
            return self.settings.tts_custom_voice_model
        return self.settings.tts_model

    # ------------------------------------------------------------- lifecycle
    def _ensure_model(self, model_id: str) -> Any:
        if self._model is not None and self._model_id == model_id:
            return self._model
        self._unload()
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise TTSError(
                "Qwen3-TTS is not installed. Install PyTorch (CUDA build) and "
                "`pip install -e .[tts]`, or set TTS_BACKEND=fake for a dry run."
            ) from exc

        device = self.settings.tts_device
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise TTSError(f"TTS_DEVICE={device} but CUDA is not available to PyTorch")
            free, total = torch.cuda.mem_get_info()
            if free / 1e9 < MIN_FREE_VRAM_GB:
                log.warning(
                    "only %.1f of %.1f GB VRAM free before loading TTS. Is llama-server "
                    "still running? Stop it (or use LLM_AUTO_MANAGE=true) to avoid slow "
                    "shared-memory spill.",
                    free / 1e9, total / 1e9,
                )  # fmt: skip
        try:  # silence per-call "Setting pad_token_id" notices from transformers
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
        except ImportError:
            pass
        dtype = getattr(torch, self.settings.tts_dtype)
        log.info("loading TTS model %s on %s (%s)", model_id, device, self.settings.tts_dtype)
        try:
            self._model = Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=dtype)
        except Exception as exc:
            raise TTSError(f"cannot load TTS model {model_id}: {exc}") from exc
        self._model_id = model_id
        return self._model

    def _unload(self) -> None:
        if self._model is None:
            return
        log.info("unloading TTS model %s", self._model_id)
        self._model = None
        self._model_id = None
        self._clone_prompts.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def close(self) -> None:
        self._unload()

    # ------------------------------------------------------------- synthesis
    def synthesize(self, request: SynthesisRequest) -> AudioClip:
        voice = request.voice
        model = self._ensure_model(self._model_for(voice))
        try:
            import torch

            torch.manual_seed(request.seed)
            if voice.source is VoiceSource.BUILTIN:
                kwargs: dict[str, Any] = {}
                instruct = self._instruction(request.emotion)
                if instruct:
                    kwargs["instruct"] = instruct
                wavs, sr = model.generate_custom_voice(
                    text=request.text,
                    speaker=voice.builtin_speaker,
                    language=request.language,
                    **kwargs,
                )
            else:
                wavs, sr = model.generate_voice_clone(
                    text=request.text,
                    language=request.language,
                    voice_clone_prompt=self._clone_prompt(model, voice),
                )
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"Qwen3-TTS failed for voice {voice.id!r}: {exc}") from exc
        samples = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise TTSError(f"Qwen3-TTS returned empty audio for voice {voice.id!r}")
        return AudioClip(samples=samples, sample_rate=int(sr))

    def _clone_prompt(self, model: Any, voice: Voice) -> Any:
        if voice.id not in self._clone_prompts:
            ref = self.library.reference_path(voice)
            if not ref.is_file():
                raise TTSError(f"reference audio for voice {voice.id!r} not found: {ref}")
            log.info("building voice clone prompt for %s", voice.id)
            self._clone_prompts[voice.id] = model.create_voice_clone_prompt(
                ref_audio=str(ref),
                ref_text=voice.reference_text,
                x_vector_only_mode=not voice.reference_text,
            )
        return self._clone_prompts[voice.id]

    def _instruction(self, emotion: Emotion) -> str:
        if not self.settings.tts_use_instruct:
            return ""
        return EMOTION_INSTRUCTIONS.get(emotion, "")

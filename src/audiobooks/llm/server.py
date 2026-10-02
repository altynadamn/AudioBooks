"""Optional lifecycle management for a local llama-server process.

On a 6 GB GPU Ornith (~5.5 GB with KV cache) and Qwen3-TTS cannot share VRAM. With
``LLM_AUTO_MANAGE=true`` the pipeline starts llama-server for the analysis stage and stops
it before synthesis. An externally started server is never touched.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path
from types import TracebackType

import httpx

from audiobooks.config import Settings
from audiobooks.errors import ConfigError, LLMError

log = logging.getLogger(__name__)


def build_server_command(settings: Settings, *, with_vision: bool = False) -> list[str]:
    """Argument list for llama-server (no shell, so paths with spaces are safe)."""
    if settings.llama_model_path is None:
        raise ConfigError("LLM_AUTO_MANAGE needs LLAMA_MODEL_PATH (path to the .gguf model)")
    exe = shutil.which(settings.llama_server_exe) or settings.llama_server_exe
    cmd = [
        exe,
        "-m", str(settings.llama_model_path),
        "--host", "127.0.0.1",
        "--port", str(settings.llama_server_port),
        "-c", str(settings.llama_ctx_size),
        "-ngl", str(settings.llama_gpu_layers),
        "-np", "1",
        "--jinja",
        "--alias", settings.llama_model,
    ]  # fmt: skip
    if with_vision:
        if settings.llama_mmproj_path is None:
            raise ConfigError("vision needs LLAMA_MMPROJ_PATH (mmproj .gguf)")
        cmd += ["--mmproj", str(settings.llama_mmproj_path)]
    cmd += settings.llama_server_extra_argv
    return cmd


class LlamaServerManager:
    def __init__(self, settings: Settings, *, log_dir: Path, with_vision: bool = False) -> None:
        self.settings = settings
        self.log_dir = log_dir
        self.with_vision = with_vision
        self._proc: subprocess.Popen[bytes] | None = None
        self._base = f"http://127.0.0.1:{settings.llama_server_port}"

    def is_healthy(self) -> bool:
        try:
            return httpx.get(f"{self._base}/health", timeout=3.0).status_code == 200
        except httpx.HTTPError:
            return False

    def start(self) -> None:
        if self.is_healthy():
            log.info("llama-server already running on %s; using it as-is", self._base)
            return
        cmd = build_server_command(self.settings, with_vision=self.with_vision)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log.info("starting llama-server: %s", " ".join(cmd))
        # the child inherits its own handle, so the parent can close the file right away
        with (self.log_dir / "llama-server.log").open("ab") as log_file:
            try:
                self._proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            except OSError as exc:
                raise LLMError(f"cannot start llama-server ({cmd[0]}): {exc}") from exc

        deadline = time.monotonic() + self.settings.llama_startup_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise LLMError(
                    f"llama-server exited with code {self._proc.returncode}; "
                    f"see {self.log_dir / 'llama-server.log'}"
                )
            if self.is_healthy():
                log.info("llama-server is ready")
                return
            time.sleep(1.0)
        self.stop()
        raise LLMError("llama-server did not become healthy in time")

    def stop(self) -> None:
        if self._proc is None:
            return
        log.info("stopping llama-server (pid %s) to free VRAM", self._proc.pid)
        self._proc.terminate()
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=10)
        self._proc = None

    def __enter__(self) -> LlamaServerManager:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

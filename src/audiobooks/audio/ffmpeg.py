"""FFmpeg wrapper. Commands are built as argument lists (never shell strings)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from audiobooks.errors import FFmpegError

log = logging.getLogger(__name__)

INSTALL_HINT = (
    "FFmpeg was not found. Install it (Windows: `winget install Gyan.FFmpeg`, "
    "macOS: `brew install ffmpeg`, Debian/Ubuntu: `apt install ffmpeg`) "
    "or set FFMPEG_PATH in .env to the full path of ffmpeg(.exe)."
)


def find_ffmpeg(configured: str = "") -> str:
    """Locate ffmpeg: FFMPEG_PATH, then PATH, then the winget package folder on Windows."""
    if configured:
        path = Path(configured)
        if path.is_file():
            return str(path)
        found = shutil.which(configured)
        if found:
            return found
        raise FFmpegError(f"FFMPEG_PATH={configured!r} does not exist. {INSTALL_HINT}")
    found = shutil.which("ffmpeg")
    if found:
        return found
    if sys.platform == "win32" and (local := os.environ.get("LOCALAPPDATA")):
        packages = Path(local) / "Microsoft" / "WinGet" / "Packages"
        for candidate in sorted(packages.glob("*FFmpeg*/**/bin/ffmpeg.exe"), reverse=True):
            return str(candidate)
    raise FFmpegError(INSTALL_HINT)


def concat_list(files: list[Path]) -> str:
    """Contents of an FFmpeg concat-demuxer list file (paths quoted and escaped)."""
    lines = ["ffconcat version 1.0"]
    for f in files:
        escaped = f.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    return "\n".join(lines) + "\n"


def _escape_meta(value: str) -> str:
    out = value
    for ch in ("\\", "=", ";", "#", "\n"):
        out = out.replace(ch, "\\" + ch)
    return out


@dataclass(frozen=True)
class ChapterMark:
    title: str
    start_ms: int
    end_ms: int


def ffmetadata(title: str, artist: str, chapters: list[ChapterMark]) -> str:
    lines = [";FFMETADATA1", f"title={_escape_meta(title)}", f"album={_escape_meta(title)}"]
    if artist:
        lines.append(f"artist={_escape_meta(artist)}")
    lines.append("genre=Audiobook")
    for ch in chapters:
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={ch.start_ms}",
            f"END={ch.end_ms}",
            f"title={_escape_meta(ch.title)}",
        ]
    return "\n".join(lines) + "\n"


def codec_args(fmt: str, bitrate: str) -> list[str]:
    if fmt == "mp3":
        return ["-c:a", "libmp3lame", "-b:a", bitrate]
    if fmt in {"m4a", "m4b"}:
        return ["-c:a", "aac", "-b:a", bitrate]
    raise FFmpegError(f"unsupported output format {fmt!r}")


def chapter_command(
    list_file: Path,
    output: Path,
    *,
    fmt: str,
    bitrate: str,
    sample_rate: int,
    loudnorm: bool,
    title: str,
    album: str,
    artist: str,
    track: int,
) -> list[str]:
    """Arguments (without the ffmpeg executable) to encode one chapter from a concat list."""
    args = ["-f", "concat", "-safe", "0", "-i", str(list_file), "-vn", "-ac", "1"]
    if loudnorm:
        args += ["-af", "loudnorm=I=-18:TP=-1.5:LRA=11"]
    args += ["-ar", str(sample_rate), *codec_args(fmt, bitrate)]
    args += [
        "-metadata", f"title={title}",
        "-metadata", f"album={album}",
        "-metadata", f"track={track}",
        "-metadata", "genre=Audiobook",
    ]  # fmt: skip
    if artist:
        args += ["-metadata", f"artist={artist}"]
    if fmt == "m4a":
        args += ["-movflags", "+faststart"]
    return [*args, str(output)]


def m4b_command(
    list_file: Path, metadata_file: Path, output: Path, *, source_fmt: str, bitrate: str
) -> list[str]:
    args = [
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-f", "ffmetadata", "-i", str(metadata_file),
        "-map", "0:a", "-map_metadata", "1", "-map_chapters", "1",
    ]  # fmt: skip
    # AAC chapters can be stream-copied; MP3 chapters must be re-encoded to AAC
    args += ["-c:a", "copy"] if source_fmt == "m4a" else codec_args("m4b", bitrate)
    return [*args, "-movflags", "+faststart", "-f", "mp4", str(output)]


def silence_command(output: Path, *, duration_ms: int, sample_rate: int) -> list[str]:
    return [
        "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=mono",
        "-t", f"{duration_ms / 1000:.3f}", "-c:a", "pcm_s16le", str(output),
    ]  # fmt: skip


def resample_command(source: Path, output: Path, *, sample_rate: int) -> list[str]:
    return [
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output),
    ]


class FFmpeg:
    def __init__(self, executable: str) -> None:
        self.executable = executable

    @classmethod
    def from_config(cls, configured: str = "") -> FFmpeg:
        return cls(find_ffmpeg(configured))

    def run(self, args: list[str], *, timeout: float | None = None) -> None:
        cmd = [self.executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args]
        log.debug("ffmpeg %s", " ".join(args))
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
        except OSError as exc:
            raise FFmpegError(f"cannot run ffmpeg ({self.executable}): {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise FFmpegError("ffmpeg timed out") from exc
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()[-1500:]
            raise FFmpegError(f"ffmpeg failed (exit {proc.returncode}): {err}")

    def version(self) -> str:
        proc = subprocess.run(
            [self.executable, "-version"], capture_output=True, check=False, timeout=30
        )
        first = proc.stdout.decode("utf-8", errors="replace").splitlines()
        return first[0] if first else "unknown"

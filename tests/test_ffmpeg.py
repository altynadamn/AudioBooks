from __future__ import annotations

from pathlib import Path

import pytest

from audiobooks.audio.ffmpeg import (
    ChapterMark,
    chapter_command,
    concat_list,
    ffmetadata,
    find_ffmpeg,
    m4b_command,
    silence_command,
)
from audiobooks.errors import FFmpegError


def test_concat_list_quotes_and_escapes(tmp_path: Path) -> None:
    f = tmp_path / "it's a file.wav"
    text = concat_list([f])
    assert text.startswith("ffconcat version 1.0\n")
    assert "it'\\''s a file.wav'" in text
    assert "\\" not in text.split("'")[1].replace("'\\''", "")  # posix separators


def test_chapter_command_m4a(tmp_path: Path) -> None:
    args = chapter_command(
        tmp_path / "list.ffconcat",
        tmp_path / "chapter_001.m4a",
        fmt="m4a",
        bitrate="96k",
        sample_rate=44100,
        loudnorm=True,
        title='Глава 1; "кавычки"',
        album="Книга",
        artist="",
        track=1,
    )
    assert args[:6] == ["-f", "concat", "-safe", "0", "-i", str(tmp_path / "list.ffconcat")]
    assert args[args.index("-c:a") + 1] == "aac"
    assert "loudnorm=I=-18:TP=-1.5:LRA=11" in args
    # metadata values are separate argv entries: no shell quoting needed or possible
    assert 'title=Глава 1; "кавычки"' in args
    assert args[-1].endswith("chapter_001.m4a")
    assert "artist=" not in " ".join(args)


def test_chapter_command_mp3_without_loudnorm(tmp_path: Path) -> None:
    args = chapter_command(
        tmp_path / "l", tmp_path / "o.mp3", fmt="mp3", bitrate="128k", sample_rate=44100,
        loudnorm=False, title="t", album="a", artist="x", track=2,
    )  # fmt: skip
    assert "libmp3lame" in args and "-af" not in args and "-movflags" not in args


def test_m4b_command_stream_copy_for_aac(tmp_path: Path) -> None:
    args = m4b_command(
        tmp_path / "l", tmp_path / "m", tmp_path / "b.m4b", source_fmt="m4a", bitrate="96k"
    )
    assert args[args.index("-map_chapters") : args.index("-map_chapters") + 2] == [
        "-map_chapters",
        "1",
    ]
    assert args[args.index("-c:a") + 1] == "copy"
    assert args[-3:] == ["-f", "mp4", str(tmp_path / "b.m4b")]


def test_m4b_command_reencodes_mp3(tmp_path: Path) -> None:
    args = m4b_command(
        tmp_path / "l", tmp_path / "m", tmp_path / "b.m4b", source_fmt="mp3", bitrate="96k"
    )
    assert args[args.index("-c:a") + 1] == "aac"


def test_ffmetadata_escaping_and_chapters() -> None:
    text = ffmetadata("A=B;C", "Автор", [ChapterMark("Глава #1", 0, 1500)])
    assert "title=A\\=B\\;C" in text
    assert "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=1500\ntitle=Глава \\#1" in text


def test_silence_command(tmp_path: Path) -> None:
    args = silence_command(tmp_path / "s.wav", duration_ms=350, sample_rate=24000)
    assert "anullsrc=r=24000:cl=mono" in args
    assert args[args.index("-t") + 1] == "0.350"


def test_missing_configured_ffmpeg_gives_install_hint(tmp_path: Path) -> None:
    with pytest.raises(FFmpegError, match=r"winget install Gyan.FFmpeg"):
        find_ffmpeg(str(tmp_path / "nope" / "ffmpeg.exe"))

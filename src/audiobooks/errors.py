"""Application exception hierarchy.

Every error raised deliberately by AudioBooks derives from ``AudioBooksError`` so the
CLI/API can turn it into a clear message instead of a traceback.
"""


class AudioBooksError(Exception):
    """Base class for expected, user-facing errors."""


class ConfigError(AudioBooksError):
    """Invalid or missing configuration."""


class UnsupportedFormatError(AudioBooksError):
    """The input file type is not supported."""


class ParseError(AudioBooksError):
    """A book file could not be parsed."""


class LLMError(AudioBooksError):
    """The language model could not be reached or returned unusable output."""


class LLMResponseError(LLMError):
    """The language model responded, but the response failed validation."""


class TTSError(AudioBooksError):
    """Speech synthesis failed or the TTS backend is unavailable."""


class VoiceNotFoundError(AudioBooksError):
    """A referenced voice id does not exist in the voice library."""


class FFmpegError(AudioBooksError):
    """FFmpeg is missing or an FFmpeg command failed."""


class NotFoundError(AudioBooksError):
    """A requested book/job/chapter does not exist."""

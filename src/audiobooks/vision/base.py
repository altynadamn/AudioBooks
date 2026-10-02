"""Optional OCR/vision boundary for pages that have no usable text layer.

Normal text formats never touch this module. Only image-only PDF pages (and, later,
standalone images) are routed to a ``PageTextExtractor``.
"""

from __future__ import annotations

from typing import Protocol


class PageTextExtractor(Protocol):
    def extract_text(self, image_png: bytes, *, language_hint: str = "") -> str:
        """Return the page text, preserving paragraph breaks as blank lines."""
        ...

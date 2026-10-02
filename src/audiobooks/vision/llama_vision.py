"""Experimental OCR through a multimodal llama-server (Ornith + mmproj).

The server must be started with ``--mmproj``; the request uses the OpenAI ``image_url``
content format with a base64 data URI.
"""

from __future__ import annotations

import base64

from audiobooks.llm.client import OpenAICompatClient

OCR_PROMPT = (
    "Transcribe all text on this book page exactly as written, in reading order. "
    "Keep the original language and punctuation. Separate paragraphs with a blank line. "
    "Omit page numbers and running headers. Output only the text."
)


class LlamaVisionExtractor:
    def __init__(self, client: OpenAICompatClient, *, max_tokens: int = 2048) -> None:
        self.client = client
        self.max_tokens = max_tokens

    def extract_text(self, image_png: bytes, *, language_hint: str = "") -> str:
        data_uri = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
        prompt = OCR_PROMPT + (f" The page language is {language_hint}." if language_hint else "")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.client.chat(messages, temperature=0.0, max_tokens=self.max_tokens).strip()

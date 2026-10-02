"""Best-effort, *safe* extraction of a JSON object from raw LLM output.

Only structural cleanups are applied (strip reasoning blocks and code fences, cut the
outermost object, drop trailing commas). Content is never invented; if the result still
does not parse, the caller retries or fails.
"""

from __future__ import annotations

import json
import re
from typing import Any

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


class JSONExtractionError(ValueError):
    pass


def _outermost_object(text: str) -> str | None:
    """Return the first balanced {...} block, respecting strings and escapes."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_object(raw: str) -> dict[str, Any]:
    if not raw or not raw.strip():
        raise JSONExtractionError("empty response")
    text = _THINK.sub("", raw)
    # an unterminated <think> (model ran out of tokens while reasoning)
    if "<think>" in text.lower():
        text = text[text.lower().rfind("</think>") + 8 :] if "</think>" in text.lower() else ""
    candidates = [text.strip()]
    candidates += [m.group(1).strip() for m in _FENCE.finditer(text)]
    obj = _outermost_object(text)
    if obj:
        candidates.append(obj)

    last_error: Exception | None = None
    for candidate in candidates:
        for variant in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
            try:
                value = json.loads(variant)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(value, dict):
                return value
            last_error = JSONExtractionError(f"expected a JSON object, got {type(value).__name__}")
    raise JSONExtractionError(f"no valid JSON object found: {last_error}")

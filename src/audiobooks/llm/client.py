"""Minimal OpenAI-compatible chat client (llama-server, vLLM, LM Studio, ...)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from audiobooks.errors import LLMError

log = logging.getLogger(__name__)


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "",
        timeout: float = 600.0,
        disable_thinking: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.disable_thinking = disable_thinking
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0), headers=headers, transport=transport
        )

    def close(self) -> None:
        self._http.close()

    def health(self) -> bool:
        """True if the server answers. llama-server exposes /health next to /v1."""
        root = self.base_url.removesuffix("/v1")
        for url in (f"{root}/health", f"{self.base_url}/models"):
            try:
                if self._http.get(url, timeout=5.0).status_code == 200:
                    return True
            except httpx.HTTPError:
                continue
        return False

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": json_schema},
            }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        url = f"{self.base_url}/chat/completions"
        try:
            response = self._http.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise LLMError(
                f"cannot connect to the LLM at {self.base_url} - is llama-server running? "
                "(see README: 'llama.cpp / Ornith setup')"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        if response.status_code != 200:
            raise LLMError(f"LLM returned HTTP {response.status_code}: {response.text[:500]}")
        try:
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected LLM response shape: {response.text[:500]}") from exc
        if choice.get("finish_reason") == "length":
            log.warning("LLM output hit max_tokens (%d); response may be truncated", max_tokens)
        timings = data.get("timings") or {}
        if timings:
            log.debug(
                "LLM: prompt %s tok @ %.0f t/s, gen %s tok @ %.1f t/s",
                timings.get("prompt_n"), timings.get("prompt_per_second", 0),
                timings.get("predicted_n"), timings.get("predicted_per_second", 0),
            )  # fmt: skip
        return content

from __future__ import annotations

import time
import logging
from typing import TypeVar, Type

import instructor
from openai import OpenAI
from pydantic import BaseModel

from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_MAX_RETRIES,
    OLLAMA_TIMEOUT,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _build_instructor_client() -> instructor.Instructor:
    """Create an instructor-patched Ollama client via OpenAI-compatible API."""
    raw = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama", timeout=OLLAMA_TIMEOUT)
    return instructor.from_openai(raw, mode=instructor.Mode.JSON)


_client: instructor.Instructor | None = None


def get_client() -> instructor.Instructor:
    global _client
    if _client is None:
        _client = _build_instructor_client()
    return _client


def structured_completion(
    prompt: str,
    response_model: Type[T],
    system: str = "",
    temperature: float = OLLAMA_TEMPERATURE,
    max_retries: int = OLLAMA_MAX_RETRIES,
) -> T:
    """Call Ollama with instructor-enforced structured output.

    Retries up to *max_retries* times on transient failures with exponential
    back-off (1s, 2s, 4s ...).
    """
    client = get_client()
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            result: T = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                response_model=response_model,
                temperature=temperature,
            )
            return result
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "LLM call failed (attempt %d/%d): %s - retrying in %ds",
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"LLM structured completion failed after {max_retries} attempts"
    ) from last_exc

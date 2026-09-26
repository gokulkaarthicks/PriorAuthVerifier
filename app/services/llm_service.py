"""Structured LLM client for OpenAI-compatible models."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, TypeVar

from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ConfigurationError,
    LLMCompletionLimitError,
    LLMServiceError,
)
from app.core.logging import get_logger
from app.core.metrics import get_request_metrics

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_RETRYABLE_TYPE_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "TimeoutError",
        "RateLimitError",
        "InternalServerError",
    }
)
_COMPLETION_LIMIT_TYPE_NAMES = frozenset({"LengthFinishReasonError"})
_RETRY_BASE_DELAY_SECONDS = 1.0
_RETRY_MAX_DELAY_SECONDS = 8.0
_RETRY_JITTER_SECONDS = 0.25


class LLMService:
    """Async structured extraction against a configured chat model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: object | None = None
        self.init_count: int = 0

    async def extract_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        model: str | None = None,
        operation: str | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        context_items: int | None = None,
        batch_index: int | None = None,
        item_count: int | None = None,
        unique_evidence_chunks: int | None = None,
    ) -> T:
        """Call the LLM and parse the response into a Pydantic model.

        Retries are owned exclusively by this method. The OpenAI client is
        created with max_retries=0 so SDK automatic retries never nest.

        Each call sends only the current operation's system + user messages.
        Prior conversation turns are never appended.
        """
        selected_model = model or self._settings.STRUCTURED_EXTRACTION_MODEL
        client = self._get_client()
        max_attempts = max(1, self._settings.LLM_MAX_RETRIES + 1)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Exception | None = None
        started = time.perf_counter()
        for attempt in range(1, max_attempts + 1):
            try:
                parse_kwargs: dict[str, Any] = {
                    "model": selected_model,
                    "messages": messages,
                    "response_format": response_model,
                }
                if max_output_tokens is not None:
                    parse_kwargs["max_completion_tokens"] = max_output_tokens
                if reasoning_effort is not None:
                    parse_kwargs["reasoning_effort"] = reasoning_effort

                response = await client.beta.chat.completions.parse(  # type: ignore[attr-defined]
                    **parse_kwargs
                )
                usage = self._usage_from_response(response)
                if usage is not None:
                    self._record_usage(selected_model, usage, failed=False)
                message = response.choices[0].message
                if message.refusal:
                    raise LLMServiceError(
                        f"LLM refused the request: {message.refusal}"
                    )
                if message.parsed is None:
                    raise LLMServiceError(
                        "LLM returned no parsed structured output"
                    )
                duration_ms = int((time.perf_counter() - started) * 1000)
                usage_fields = usage or {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                }
                self._record_call_detail(
                    operation=operation or "unknown",
                    model=selected_model,
                    usage_fields=usage_fields,
                    duration_ms=duration_ms,
                    context_items=context_items,
                    batch_index=batch_index,
                    item_count=item_count,
                    unique_evidence_chunks=unique_evidence_chunks,
                )
                logger.info(
                    "llm_request_completed",
                    operation=operation,
                    model=selected_model,
                    duration_ms=duration_ms,
                    input_tokens=usage_fields["input_tokens"],
                    output_tokens=usage_fields["output_tokens"],
                    reasoning_tokens=usage_fields["reasoning_tokens"],
                    total_tokens=usage_fields["total_tokens"],
                    retry_count=attempt - 1,
                    context_items=context_items,
                    batch_index=batch_index,
                    item_count=item_count,
                    unique_evidence_chunks=unique_evidence_chunks,
                )
                return message.parsed  # type: ignore[no-any-return]
            except LLMServiceError:
                raise
            except Exception as exc:
                last_error = exc
                failed_usage = self._usage_from_exception(exc)
                if failed_usage is not None:
                    self._record_usage(selected_model, failed_usage, failed=True)

                duration_ms = int((time.perf_counter() - started) * 1000)
                if self._is_completion_limit(exc):
                    usage_fields = failed_usage or {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning_tokens": 0,
                        "total_tokens": 0,
                    }
                    logger.error(
                        "llm_request_failed",
                        operation=operation,
                        model=selected_model,
                        attempt=attempt,
                        duration_ms=duration_ms,
                        error_type="completion_limit",
                        input_tokens=usage_fields["input_tokens"],
                        output_tokens=usage_fields["output_tokens"],
                        reasoning_tokens=usage_fields["reasoning_tokens"],
                        total_tokens=usage_fields["total_tokens"],
                        retryable=False,
                    )
                    raise LLMCompletionLimitError(
                        "Structured LLM extraction hit the completion token limit"
                    ) from exc

                if not self._is_transient(exc) or attempt >= max_attempts:
                    logger.error(
                        "llm_request_failed",
                        operation=operation,
                        model=selected_model,
                        attempt=attempt,
                        duration_ms=duration_ms,
                        error_type=type(exc).__name__,
                        retryable=False,
                    )
                    raise LLMServiceError(
                        "Structured LLM extraction failed"
                    ) from exc
                delay = self._retry_delay(attempt - 1)
                logger.warning(
                    "llm_request_retry",
                    operation=operation,
                    model=selected_model,
                    attempt=attempt,
                    delay_seconds=round(delay, 3),
                    error=str(exc),
                )
                await asyncio.sleep(delay)

        raise LLMServiceError("Structured LLM extraction failed") from last_error

    def _usage_from_response(self, response: Any) -> dict[str, int] | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return self._usage_from_usage_object(usage)

    def _usage_from_exception(self, exc: Exception) -> dict[str, int] | None:
        completion = getattr(exc, "completion", None)
        if completion is None:
            return None
        usage = getattr(completion, "usage", None)
        if usage is None:
            return None
        return self._usage_from_usage_object(usage)

    def _usage_from_usage_object(self, usage: Any) -> dict[str, int]:
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = 0
        if details is not None:
            reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
        if total_tokens <= 0:
            total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
        }

    def _record_usage(
        self,
        model: str,
        usage: dict[str, int],
        *,
        failed: bool,
    ) -> None:
        metrics = get_request_metrics()
        if metrics is None:
            return
        metrics.record_llm_usage(
            model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            total_tokens=usage["total_tokens"],
            failed=failed,
        )

    def _record_call_detail(
        self,
        *,
        operation: str,
        model: str,
        usage_fields: dict[str, int],
        duration_ms: int,
        context_items: int | None,
        batch_index: int | None,
        item_count: int | None,
        unique_evidence_chunks: int | None,
    ) -> None:
        metrics = get_request_metrics()
        if metrics is None:
            return
        metrics.record_llm_call(
            operation=operation,
            model=model,
            input_tokens=usage_fields["input_tokens"],
            output_tokens=usage_fields["output_tokens"],
            reasoning_tokens=usage_fields["reasoning_tokens"],
            total_tokens=usage_fields["total_tokens"],
            duration_ms=duration_ms,
            context_items=context_items,
            item_count=item_count,
            unique_evidence_chunks=unique_evidence_chunks,
            batch_index=batch_index,
        )

    def _is_completion_limit(self, exc: Exception) -> bool:
        return type(exc).__name__ in _COMPLETION_LIMIT_TYPE_NAMES

    def _is_transient(self, exc: Exception) -> bool:
        name = type(exc).__name__
        if name in _RETRYABLE_TYPE_NAMES:
            return True
        status = getattr(exc, "status_code", None)
        return isinstance(status, int) and status in _RETRYABLE_STATUS_CODES

    def _retry_delay(self, retry_index: int) -> float:
        base = min(
            _RETRY_BASE_DELAY_SECONDS * (2**retry_index),
            _RETRY_MAX_DELAY_SECONDS,
        )
        return float(base + random.uniform(0.0, _RETRY_JITTER_SECONDS))

    def _get_client(self) -> object:
        api_key = self._settings.OPENAI_API_KEY
        if not api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is required to run LLM extraction"
            )

        if self._client is not None:
            return self._client

        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=api_key,
                timeout=self._settings.LLM_REQUEST_TIMEOUT_SECONDS,
                max_retries=0,
            )
            self.init_count += 1
            return self._client
        except ConfigurationError:
            raise
        except Exception as exc:
            raise LLMServiceError("Failed to initialize OpenAI client") from exc

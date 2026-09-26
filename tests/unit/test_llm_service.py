"""Unit tests for LLMService."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, LLMServiceError
from app.schemas.procedure import ProcedureExtraction
from app.services.llm_service import LLMService


def _success_completion(parsed: ProcedureExtraction) -> SimpleNamespace:
    parsed_message = SimpleNamespace(parsed=parsed, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=parsed_message)])


@pytest.mark.asyncio
async def test_extract_structured_returns_pydantic_model() -> None:
    expected = ProcedureExtraction(
        procedure_name="Example Procedure",
        cpt_code="12345",
        status="planned",
        source_chunk_id="chunk-1",
        source_text="Example Procedure CPT 12345 planned",
        page_number=2,
        is_future_procedure=True,
        confidence=0.9,
    )
    completion = _success_completion(expected)

    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(return_value=completion)

    service = LLMService(
        settings=Settings(
            OPENAI_API_KEY="test-key",
            STRUCTURED_EXTRACTION_MODEL="gpt-6-luna",
        )
    )
    service._client = client

    result = await service.extract_structured(
        system_prompt="system",
        user_prompt="user",
        response_model=ProcedureExtraction,
    )

    assert result == expected
    client.beta.chat.completions.parse.assert_awaited_once()
    kwargs = client.beta.chat.completions.parse.await_args.kwargs
    assert kwargs["model"] == "gpt-6-luna"
    assert kwargs["response_format"] is ProcedureExtraction
    assert kwargs["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


@pytest.mark.asyncio
async def test_missing_api_key_raises() -> None:
    service = LLMService(settings=Settings(OPENAI_API_KEY=None))

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        await service.extract_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=ProcedureExtraction,
        )


@pytest.mark.asyncio
async def test_refusal_raises_without_retry() -> None:
    parsed_message = SimpleNamespace(parsed=None, refusal="cannot help")
    completion = SimpleNamespace(choices=[SimpleNamespace(message=parsed_message)])
    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(return_value=completion)

    service = LLMService(
        settings=Settings(OPENAI_API_KEY="test-key", LLM_MAX_RETRIES=2)
    )
    service._client = client

    with pytest.raises(LLMServiceError, match="refused"):
        await service.extract_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=ProcedureExtraction,
        )

    assert client.beta.chat.completions.parse.await_count == 1


@pytest.mark.asyncio
async def test_client_initialized_from_settings() -> None:
    expected = ProcedureExtraction(is_future_procedure=False, confidence=0.0)
    completion = _success_completion(expected)

    fake_client = MagicMock()
    fake_client.beta.chat.completions.parse = AsyncMock(return_value=completion)

    service = LLMService(settings=Settings(OPENAI_API_KEY="secret"))
    with patch("openai.AsyncOpenAI", return_value=fake_client) as ctor:
        result = await service.extract_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=ProcedureExtraction,
        )

    ctor.assert_called_once_with(
        api_key="secret",
        timeout=120.0,
        max_retries=0,
    )
    assert result.is_future_procedure is False
    assert service.init_count == 1


@pytest.mark.asyncio
async def test_openai_client_uses_zero_sdk_retries() -> None:
    expected = ProcedureExtraction(is_future_procedure=False, confidence=0.0)
    fake_client = MagicMock()
    fake_client.beta.chat.completions.parse = AsyncMock(
        return_value=_success_completion(expected)
    )

    service = LLMService(settings=Settings(OPENAI_API_KEY="secret"))
    with patch("openai.AsyncOpenAI", return_value=fake_client) as ctor:
        await service.extract_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=ProcedureExtraction,
        )

    assert ctor.call_args.kwargs["max_retries"] == 0


@pytest.mark.asyncio
async def test_retries_transient_errors_up_to_max_attempts() -> None:
    class APITimeoutError(Exception):
        pass

    expected = ProcedureExtraction(is_future_procedure=False, confidence=0.0)
    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(
        side_effect=[
            APITimeoutError("timeout"),
            APITimeoutError("timeout"),
            _success_completion(expected),
        ]
    )

    service = LLMService(
        settings=Settings(OPENAI_API_KEY="test-key", LLM_MAX_RETRIES=2)
    )
    service._client = client

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await service.extract_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=ProcedureExtraction,
        )

    assert result.is_future_procedure is False
    assert client.beta.chat.completions.parse.await_count == 3


@pytest.mark.asyncio
async def test_does_not_retry_bad_request() -> None:
    class BadRequestError(Exception):
        status_code = 400

    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(
        side_effect=BadRequestError("bad request")
    )
    service = LLMService(
        settings=Settings(OPENAI_API_KEY="test-key", LLM_MAX_RETRIES=2)
    )
    service._client = client

    with pytest.raises(LLMServiceError, match="Structured LLM extraction failed"):
        await service.extract_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=ProcedureExtraction,
        )

    assert client.beta.chat.completions.parse.await_count == 1


@pytest.mark.asyncio
async def test_retry_backoff_uses_jitter() -> None:
    class RateLimitError(Exception):
        pass

    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(
        side_effect=[
            RateLimitError("rate limited"),
            _success_completion(
                ProcedureExtraction(is_future_procedure=False, confidence=0.0)
            ),
        ]
    )
    service = LLMService(
        settings=Settings(OPENAI_API_KEY="test-key", LLM_MAX_RETRIES=2)
    )
    service._client = client

    with patch("asyncio.sleep", new=AsyncMock()) as sleep_mock, patch(
        "app.services.llm_service.random.uniform",
        return_value=0.1,
    ):
        await service.extract_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=ProcedureExtraction,
        )

    sleep_mock.assert_awaited_once()
    assert sleep_mock.await_args.args[0] == pytest.approx(1.1)


@pytest.mark.asyncio
async def test_max_output_tokens_passed_as_max_completion_tokens() -> None:
    expected = ProcedureExtraction(is_future_procedure=False, confidence=0.0)
    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(
        return_value=_success_completion(expected)
    )
    service = LLMService(settings=Settings(OPENAI_API_KEY="test-key"))
    service._client = client

    await service.extract_structured(
        system_prompt="system",
        user_prompt="user",
        response_model=ProcedureExtraction,
        max_output_tokens=512,
    )

    kwargs = client.beta.chat.completions.parse.await_args.kwargs
    assert kwargs["max_completion_tokens"] == 512


@pytest.mark.asyncio
async def test_success_logs_llm_request_completed() -> None:
    expected = ProcedureExtraction(is_future_procedure=False, confidence=0.0)
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    completion = _success_completion(expected)
    completion.usage = usage

    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(return_value=completion)
    service = LLMService(settings=Settings(OPENAI_API_KEY="test-key"))
    service._client = client

    with patch("app.services.llm_service.logger.info") as log_info:
        await service.extract_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=ProcedureExtraction,
            operation="unit_test_op",
            context_items=4,
        )

    completed = [
        call
        for call in log_info.call_args_list
        if call.args and call.args[0] == "llm_request_completed"
    ]
    assert len(completed) == 1
    kwargs = completed[0].kwargs
    assert kwargs["operation"] == "unit_test_op"
    assert kwargs["input_tokens"] == 5
    assert kwargs["output_tokens"] == 3
    assert kwargs["total_tokens"] == 8
    assert kwargs["retry_count"] == 0
    assert kwargs["context_items"] == 4
    assert kwargs["reasoning_tokens"] == 0


@pytest.mark.asyncio
async def test_reasoning_effort_passed_when_provided() -> None:
    expected = ProcedureExtraction(is_future_procedure=False, confidence=0.0)
    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(
        return_value=_success_completion(expected)
    )
    service = LLMService(settings=Settings(OPENAI_API_KEY="test-key"))
    service._client = client

    await service.extract_structured(
        system_prompt="system",
        user_prompt="user",
        response_model=ProcedureExtraction,
        reasoning_effort="low",
    )

    kwargs = client.beta.chat.completions.parse.await_args.kwargs
    assert kwargs["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_reasoning_effort_omitted_when_not_provided() -> None:
    expected = ProcedureExtraction(is_future_procedure=False, confidence=0.0)
    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(
        return_value=_success_completion(expected)
    )
    service = LLMService(settings=Settings(OPENAI_API_KEY="test-key"))
    service._client = client

    await service.extract_structured(
        system_prompt="system",
        user_prompt="user",
        response_model=ProcedureExtraction,
    )

    kwargs = client.beta.chat.completions.parse.await_args.kwargs
    assert "reasoning_effort" not in kwargs


@pytest.mark.asyncio
async def test_messages_are_only_current_operation_pair() -> None:
    expected = ProcedureExtraction(is_future_procedure=False, confidence=0.0)
    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(
        return_value=_success_completion(expected)
    )
    service = LLMService(settings=Settings(OPENAI_API_KEY="test-key"))
    service._client = client

    await service.extract_structured(
        system_prompt="policy system",
        user_prompt="policy user",
        response_model=ProcedureExtraction,
    )
    await service.extract_structured(
        system_prompt="evidence system",
        user_prompt="evidence user",
        response_model=ProcedureExtraction,
    )

    first = client.beta.chat.completions.parse.await_args_list[0].kwargs["messages"]
    second = client.beta.chat.completions.parse.await_args_list[1].kwargs["messages"]
    assert first == [
        {"role": "system", "content": "policy system"},
        {"role": "user", "content": "policy user"},
    ]
    assert second == [
        {"role": "system", "content": "evidence system"},
        {"role": "user", "content": "evidence user"},
    ]
    assert len(second) == 2


@pytest.mark.asyncio
async def test_completion_limit_is_non_retryable_and_records_usage() -> None:
    class LengthFinishReasonError(Exception):
        def __init__(self) -> None:
            details = SimpleNamespace(reasoning_tokens=12000)
            usage = SimpleNamespace(
                prompt_tokens=7706,
                completion_tokens=12000,
                total_tokens=19706,
                completion_tokens_details=details,
            )
            self.completion = SimpleNamespace(usage=usage)
            super().__init__("length limit")

    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(
        side_effect=LengthFinishReasonError()
    )
    service = LLMService(
        settings=Settings(OPENAI_API_KEY="test-key", LLM_MAX_RETRIES=2)
    )
    service._client = client

    from app.core.exceptions import LLMCompletionLimitError
    from app.core.metrics import (
        RequestMetrics,
        bind_request_metrics,
        reset_request_metrics,
    )

    metrics = RequestMetrics(request_id="limit-test")
    token = bind_request_metrics(metrics)
    try:
        with (
            patch("app.services.llm_service.logger.error") as log_error,
            pytest.raises(LLMCompletionLimitError, match="completion token limit"),
        ):
            await service.extract_structured(
                system_prompt="system",
                user_prompt="user",
                response_model=ProcedureExtraction,
                operation="policy_criteria",
                model="gpt-6-sol",
            )
    finally:
        reset_request_metrics(token)

    assert client.beta.chat.completions.parse.await_count == 1
    usage = metrics.model_usage["gpt-6-sol"]
    assert usage.calls == 1
    assert usage.failed_calls == 1
    assert usage.input_tokens == 7706
    assert usage.output_tokens == 12000
    assert usage.reasoning_tokens == 12000
    assert usage.total_tokens == 19706

    failed = [
        call
        for call in log_error.call_args_list
        if call.args and call.args[0] == "llm_request_failed"
    ]
    assert len(failed) == 1
    kwargs = failed[0].kwargs
    assert kwargs["error_type"] == "completion_limit"
    assert kwargs["retryable"] is False
    assert kwargs["reasoning_tokens"] == 12000


@pytest.mark.asyncio
async def test_timeout_remains_retryable() -> None:
    class APITimeoutError(Exception):
        pass

    expected = ProcedureExtraction(is_future_procedure=False, confidence=0.0)
    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(
        side_effect=[APITimeoutError("timeout"), _success_completion(expected)]
    )
    service = LLMService(
        settings=Settings(OPENAI_API_KEY="test-key", LLM_MAX_RETRIES=2)
    )
    service._client = client

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await service.extract_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=ProcedureExtraction,
        )

    assert result.is_future_procedure is False
    assert client.beta.chat.completions.parse.await_count == 2


@pytest.mark.asyncio
async def test_rate_limit_remains_retryable() -> None:
    class RateLimitError(Exception):
        status_code = 429

    expected = ProcedureExtraction(is_future_procedure=False, confidence=0.0)
    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(
        side_effect=[RateLimitError("rate"), _success_completion(expected)]
    )
    service = LLMService(
        settings=Settings(OPENAI_API_KEY="test-key", LLM_MAX_RETRIES=2)
    )
    service._client = client

    with patch("asyncio.sleep", new=AsyncMock()):
        await service.extract_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=ProcedureExtraction,
        )

    assert client.beta.chat.completions.parse.await_count == 2


@pytest.mark.asyncio
async def test_retryable_5xx_remains_retryable() -> None:
    class InternalServerError(Exception):
        status_code = 503

    expected = ProcedureExtraction(is_future_procedure=False, confidence=0.0)
    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(
        side_effect=[InternalServerError("down"), _success_completion(expected)]
    )
    service = LLMService(
        settings=Settings(OPENAI_API_KEY="test-key", LLM_MAX_RETRIES=2)
    )
    service._client = client

    with patch("asyncio.sleep", new=AsyncMock()):
        await service.extract_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=ProcedureExtraction,
        )

    assert client.beta.chat.completions.parse.await_count == 2


@pytest.mark.asyncio
async def test_success_logs_reasoning_tokens_when_present() -> None:
    expected = ProcedureExtraction(is_future_procedure=False, confidence=0.0)
    details = SimpleNamespace(reasoning_tokens=42)
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=50,
        total_tokens=60,
        completion_tokens_details=details,
    )
    completion = _success_completion(expected)
    completion.usage = usage

    client = MagicMock()
    client.beta.chat.completions.parse = AsyncMock(return_value=completion)
    service = LLMService(settings=Settings(OPENAI_API_KEY="test-key"))
    service._client = client

    with patch("app.services.llm_service.logger.info") as log_info:
        await service.extract_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=ProcedureExtraction,
            operation="policy_criteria",
        )

    completed = [
        call
        for call in log_info.call_args_list
        if call.args and call.args[0] == "llm_request_completed"
    ]
    assert completed[0].kwargs["reasoning_tokens"] == 42

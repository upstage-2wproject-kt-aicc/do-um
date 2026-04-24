"""Async multi-LLM orchestration skeleton."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from src.common.schemas import LLMBatchResponse, LLMRequest, LLMResponse


@dataclass(frozen=True)
class ProviderConfig:
    """Defines environment variable mapping for one provider."""

    provider: str
    key_env: str
    model_env: str
    base_url_env: str
    default_base_url: str


PROVIDER_CONFIGS: tuple[ProviderConfig, ...] = (
    ProviderConfig(
        provider="solar",
        key_env="LLM_SOLAR_API_KEY",
        model_env="LLM_SOLAR_MODEL",
        base_url_env="LLM_SOLAR_BASE_URL",
        default_base_url="https://api.upstage.ai/v1",
    ),
    ProviderConfig(
        provider="claude-sonnet",
        key_env="LLM_CLAUDE_SONNET_API_KEY",
        model_env="LLM_CLAUDE_SONNET_MODEL",
        base_url_env="LLM_CLAUDE_SONNET_BASE_URL",
        default_base_url="https://api.anthropic.com/v1",
    ),
    ProviderConfig(
        provider="gpt",
        key_env="LLM_GPT_API_KEY",
        model_env="LLM_GPT_MODEL",
        base_url_env="LLM_GPT_BASE_URL",
        default_base_url="https://api.openai.com/v1",
    ),
    ProviderConfig(
        provider="grok",
        key_env="LLM_GROK_API_KEY",
        model_env="LLM_GROK_MODEL",
        base_url_env="LLM_GROK_BASE_URL",
        default_base_url="https://api.x.ai/v1",
    ),
)

ACTIVE_PROVIDERS: tuple[str, ...] = ("solar",)
logger = logging.getLogger(__name__)


class MultiLLMService:
    """Defines async fan-out/fan-in calls across LLM providers."""

    def __init__(self, timeout_s: float = 20.0) -> None:
        """Initializes multi-provider execution settings."""
        self.timeout_s = timeout_s
        self.failure_threshold = 2
        self.reset_timeout_s = 20.0
        self._failure_count: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    async def call_solar(self, request: LLMRequest) -> LLMResponse:
        """Calls Solar model with a normalized request."""
        config = self._find_provider_config("solar")
        return await self._call_openai_compatible(config, request)

    async def call_gpt4o(self, request: LLMRequest) -> LLMResponse:
        """Calls GPT-4o model with a normalized request."""
        config = self._find_provider_config("gpt")
        return await self._call_openai_compatible(config, request)

    async def call_claude35(self, request: LLMRequest) -> LLMResponse:
        """Calls Claude 3.5 model with a normalized request."""
        config = self._find_provider_config("claude-sonnet")
        return await self._call_openai_compatible(config, request)

    async def call_grok(self, request: LLMRequest) -> LLMResponse:
        """Calls Grok model with a normalized request."""
        config = self._find_provider_config("grok")
        return await self._call_openai_compatible(config, request)

    async def invoke_all(self, request: LLMRequest) -> LLMBatchResponse:
        """Runs all providers concurrently via asyncio.gather."""
        active_configs = [c for c in PROVIDER_CONFIGS if c.provider in ACTIVE_PROVIDERS]
        disabled_configs = [c for c in PROVIDER_CONFIGS if c.provider not in ACTIVE_PROVIDERS]

        tasks = [self._call_provider(config, request) for config in active_configs]
        settled = await asyncio.gather(*tasks, return_exceptions=True)
        responses: list[LLMResponse] = []
        for config, result in zip(active_configs, settled):
            if isinstance(result, LLMResponse):
                responses.append(result)
                continue
            responses.append(
                LLMResponse(
                    session_id=request.session_id,
                    provider=config.provider,
                    text="",
                    ttft_ms=0,
                    latency_ms=0,
                    finish_reason=None,
                    grounded=False,
                    citations=[],
                    error="PROVIDER_EXCEPTION",
                    token_usage={},
                )
            )
        for config in disabled_configs:
            responses.append(
                self._error_result(
                    session_id=request.session_id,
                    provider=config.provider,
                    code="PROVIDER_DISABLED",
                )
            )
        return LLMBatchResponse(session_id=request.session_id, responses=responses)

    async def _call_provider(
        self, config: ProviderConfig, request: LLMRequest
    ) -> LLMResponse:
        """Calls one provider with timeout and provider-local failure isolation."""
        now = time.time()
        opened_at = self._opened_at.get(config.provider)
        if opened_at is not None and now - opened_at < self.reset_timeout_s:
            return self._error_result(session_id=request.session_id, provider=config.provider, code="RESP_DELAY")
        if opened_at is not None and now - opened_at >= self.reset_timeout_s:
            self._opened_at.pop(config.provider, None)
            self._failure_count[config.provider] = 0

        key = os.getenv(config.key_env, "").strip()
        model = os.getenv(config.model_env, "").strip()
        if not key:
            return self._error_result(request.session_id, config.provider, "MISSING_API_KEY")
        if not model:
            return self._error_result(
                request.session_id, config.provider, "MODEL_NOT_CONFIGURED"
            )
        try:
            result = await self._call_openai_compatible(config, request)
            self._failure_count[config.provider] = 0
            return result
        except (httpx.ConnectTimeout, httpx.ReadTimeout, TimeoutError):
            logger.exception("Provider timeout: provider=%s", config.provider)
            self._mark_failure(config.provider)
            return self._error_result(request.session_id, config.provider, "RESP_DELAY")
        except httpx.ConnectError:
            logger.exception("Provider network error: provider=%s", config.provider)
            self._mark_failure(config.provider)
            return self._error_result(request.session_id, config.provider, "NETWORK_ERROR")
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body_head = exc.response.text[:300]
            logger.error(
                "Provider HTTP error: provider=%s status=%s body=%s",
                config.provider,
                status,
                body_head,
            )
            self._mark_failure(config.provider)
            if status in (401, 403):
                return self._error_result(request.session_id, config.provider, "AUTH_FAILED")
            if status == 429:
                return self._error_result(request.session_id, config.provider, "RATE_LIMIT")
            if status >= 500:
                return self._error_result(request.session_id, config.provider, "UPSTREAM_ERROR")
            return self._error_result(request.session_id, config.provider, "PROVIDER_EXCEPTION")
        except Exception:
            logger.exception("Provider unknown exception: provider=%s", config.provider)
            self._mark_failure(config.provider)
            return self._error_result(request.session_id, config.provider, "PROVIDER_EXCEPTION")

    async def _call_openai_compatible(
        self, config: ProviderConfig, request: LLMRequest
    ) -> LLMResponse:
        """Calls a chat completion endpoint with OpenAI-compatible payload."""
        key = os.getenv(config.key_env, "").strip()
        model = os.getenv(config.model_env, "").strip()
        base_url = os.getenv(config.base_url_env, config.default_base_url).strip()
        if not key:
            return self._error_result(request.session_id, config.provider, "MISSING_API_KEY")
        if not model:
            return self._error_result(
                request.session_id, config.provider, "MODEL_NOT_CONFIGURED"
            )

        started = time.perf_counter()
        url = f"{base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": request.system_prompt or ""},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            if (
                config.provider == "solar"
                and response.status_code == 404
                and "/v2" in base_url.rstrip("/")
            ):
                fallback_url = "https://api.upstage.ai/v1/chat/completions"
                logger.warning(
                    "Solar v2 endpoint returned 404. Falling back to v1: %s",
                    fallback_url,
                )
                response = await client.post(fallback_url, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        ended = time.perf_counter()
        latency_ms = int((ended - started) * 1000)

        content = ""
        finish_reason = None
        usage: dict[str, int] = {}
        choices = body.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = str(message.get("content", ""))
            finish_reason = choices[0].get("finish_reason")
        raw_usage = body.get("usage", {})
        if isinstance(raw_usage, dict):
            usage = {
                key: int(value)
                for key, value in raw_usage.items()
                if isinstance(value, (int, float))
            }

        return LLMResponse(
            session_id=request.session_id,
            provider=config.provider,
            text=content,
            ttft_ms=latency_ms,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            grounded=False,
            citations=[],
            error=None,
            token_usage=usage,
        )

    def _find_provider_config(self, provider: str) -> ProviderConfig:
        """Finds one provider config by provider name."""
        for config in PROVIDER_CONFIGS:
            if config.provider == provider:
                return config
        raise ValueError(f"Unknown provider: {provider}")

    def _error_result(self, session_id: str, provider: str, code: str) -> LLMResponse:
        """Builds standardized provider error result."""
        return LLMResponse(
            session_id=session_id,
            provider=provider,
            text="",
            ttft_ms=0,
            latency_ms=0,
            finish_reason=None,
            grounded=False,
            citations=[],
            error=code,
            token_usage={},
        )

    def _mark_failure(self, provider: str) -> None:
        """Updates circuit breaker state for one provider failure."""
        next_count = self._failure_count.get(provider, 0) + 1
        self._failure_count[provider] = next_count
        if next_count >= self.failure_threshold:
            self._opened_at[provider] = time.time()

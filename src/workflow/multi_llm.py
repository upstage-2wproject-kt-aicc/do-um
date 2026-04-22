"""Async multi-LLM orchestration skeleton."""

from __future__ import annotations

import asyncio

from src.common.schemas import LLMBatchResponse, LLMRequest, LLMResponse


class MultiLLMService:
    """Defines async fan-out/fan-in calls across LLM providers."""

    async def call_solar(self, request: LLMRequest) -> LLMResponse:
        """Calls Solar model with a normalized request."""
        raise NotImplementedError

    async def call_gpt4o(self, request: LLMRequest) -> LLMResponse:
        """Calls GPT-4o model with a normalized request."""
        raise NotImplementedError

    async def call_claude35(self, request: LLMRequest) -> LLMResponse:
        """Calls Claude 3.5 model with a normalized request."""
        raise NotImplementedError

    async def invoke_all(self, request: LLMRequest) -> LLMBatchResponse:
        """Runs all providers concurrently via asyncio.gather."""
        raise NotImplementedError


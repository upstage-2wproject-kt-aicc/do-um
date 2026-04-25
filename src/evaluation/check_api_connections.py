"""Small API connection checks for evaluation providers.

This module never prints API keys. It sends tiny requests to configured models
and reports provider/model/status only.
"""

from __future__ import annotations

import asyncio
import os

import httpx

from src.common.schemas import LLMRequest
from src.evaluation.clients import (
    AnthropicChatClient,
    GeminiChatClient,
    OpenAICompatibleChatClient,
)
from src.evaluation.env import load_evaluation_env
from src.evaluation.schemas import CandidateModel

EnvNames = str | tuple[str, ...]


async def main() -> None:
    load_evaluation_env()
    checks = [
        _check_openai_compatible("solar", "LLM_SOLAR_MODEL"),
        _check_openai_compatible("gpt", "LLM_GPT_MODEL"),
        _check_anthropic("claude-sonnet", "LLM_CLAUDE_SONNET_MODEL"),
        _check_gemini(
            "google-candidate",
            ("LLM_GOOGLE_MODEL", "LLM_GEMINI_MODEL"),
            ("LLM_GOOGLE_API_KEY", "LLM_GEMINI_API_KEY"),
        ),
        _check_openai_compatible(
            "openai",
            (
                "JUDGE_OPENAI_MODEL",
                "EVALUATION_PROVIDER_OPENAI_MODEL",
                "EVALUATION_PROVIDER_GPT_MODEL",
            ),
        ),
        _check_anthropic(
            "anthropic", ("JUDGE_ANTHROPIC_MODEL", "EVALUATION_PROVIDER_CLAUDE_MODEL")
        ),
        _check_gemini(
            "google-judge",
            ("JUDGE_GOOGLE_MODEL", "EVALUATION_PROVIDER_GEMINI_MODEL"),
            ("JUDGE_GOOGLE_API_KEY", "LLM_GEMINI_API_KEY", "LLM_GOOGLE_API_KEY"),
        ),
    ]
    results = await asyncio.gather(*checks)
    for result in results:
        print(result)


async def _check_openai_compatible(provider: str, model_env: EnvNames) -> str:
    model_id = _first_env(model_env)
    if not model_id:
        return f"{provider}: SKIP missing {_env_label(model_env)}"
    try:
        response = await OpenAICompatibleChatClient(timeout_s=10.0).generate(
            CandidateModel(provider=provider, model_id=model_id),
            _ping_request(provider),
        )
        return f"{provider}/{model_id}: OK chars={len(response.text)}"
    except Exception as exc:
        return f"{provider}/{model_id}: FAIL {_format_exception(exc)}"


async def _check_anthropic(provider: str, model_env: EnvNames) -> str:
    model_id = _first_env(model_env)
    if not model_id:
        return f"{provider}: SKIP missing {_env_label(model_env)}"
    try:
        client = AnthropicChatClient(
            api_key_env=(
                "JUDGE_ANTHROPIC_API_KEY"
                if provider == "anthropic"
                else "LLM_CLAUDE_SONNET_API_KEY"
            ),
            timeout_s=10.0,
        )
        response = await client.generate(
            CandidateModel(provider=provider, model_id=model_id),
            _ping_request(provider),
        )
        return f"{provider}/{model_id}: OK chars={len(response.text)}"
    except Exception as exc:
        return f"{provider}/{model_id}: FAIL {_format_exception(exc)}"


async def _check_gemini(
    provider: str, model_env: EnvNames, api_key_env: EnvNames
) -> str:
    model_id = _first_env(model_env)
    if not model_id:
        return f"{provider}: SKIP missing {_env_label(model_env)}"
    try:
        response = await GeminiChatClient(api_key_env=api_key_env, timeout_s=10.0).generate(
            CandidateModel(provider=provider, model_id=model_id),
            _ping_request(provider),
        )
        return f"{provider}/{model_id}: OK chars={len(response.text)}"
    except Exception as exc:
        return f"{provider}/{model_id}: FAIL {_format_exception(exc)}"


def _ping_request(provider: str) -> LLMRequest:
    return LLMRequest(
        session_id=f"api-check-{provider}",
        system_prompt="간단한 연결 확인입니다. 한 단어로만 답하세요.",
        prompt="OK라고만 답하세요.",
        temperature=0.0,
        max_tokens=128,
    )


def _first_env(names: EnvNames) -> str:
    env_names = (names,) if isinstance(names, str) else names
    for env_name in env_names:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""


def _env_label(names: EnvNames) -> str:
    if isinstance(names, str):
        return names
    return " or ".join(names)


def _format_exception(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text.replace("\n", " ").strip()
        if len(body) > 300:
            body = body[:300] + "..."
        return f"HTTP {exc.response.status_code}: {body}"
    return f"{type(exc).__name__}: {exc}"


if __name__ == "__main__":
    asyncio.run(main())

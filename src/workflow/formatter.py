"""Formatting helpers for workflow output normalization."""

from __future__ import annotations

import os
import re

from src.common.schemas import (
    LLMBatchResponse,
    ProviderErrorCode,
    ProviderResult,
    WorkflowOutput,
)


PROVIDER_MODEL_ENV: dict[str, str] = {
    "solar": "LLM_SOLAR_MODEL",
    "claude-sonnet": "LLM_CLAUDE_SONNET_MODEL",
    "gpt": "LLM_GPT_MODEL",
    "grok": "LLM_GROK_MODEL",
}
URL_PATTERN = re.compile(r"https?://[^\s)\]>\"']+")


def format_workflow_output(batch: LLMBatchResponse) -> WorkflowOutput:
    """Builds workflow output contract from provider batch responses."""
    results = [_to_provider_result(item) for item in batch.responses]
    final_answer = next((item.answer for item in results if not item.error), "")
    is_handoff = final_answer == ""
    final_answer, extracted_links = _strip_urls(final_answer)
    references = [
        *next((item.citations for item in results if not item.error), []),
        *extracted_links,
    ]
    usage = _aggregate_token_usage(results)
    return WorkflowOutput(
        session_id=batch.session_id,
        results=results,
        final_answer_text=final_answer,
        pre_tts_text=final_answer,
        is_handoff_decided=is_handoff,
        reference_links=references,
        llm_token_usage=usage,
    )


def _to_provider_result(response: object) -> ProviderResult:
    """Converts one low-level response object into normalized provider result."""
    provider = getattr(response, "provider", "")
    model = os.getenv(PROVIDER_MODEL_ENV.get(provider, ""), "")
    error_raw = getattr(response, "error", None)
    error_code = _parse_error_code(error_raw)
    return ProviderResult(
        provider=provider,
        model=model,
        answer=getattr(response, "text", ""),
        ttft_ms=getattr(response, "ttft_ms", 0),
        latency_ms=getattr(response, "latency_ms", 0),
        grounded=getattr(response, "grounded", False),
        citations=getattr(response, "citations", []),
        error=error_code,
        token_usage=getattr(response, "token_usage", {}),
    )


def _parse_error_code(value: object) -> ProviderErrorCode | None:
    """Parses string error into enum code where possible."""
    if value is None:
        return None
    if isinstance(value, ProviderErrorCode):
        return value
    if isinstance(value, str):
        for code in ProviderErrorCode:
            if code.value == value:
                return code
    return ProviderErrorCode.PROVIDER_EXCEPTION


def _aggregate_token_usage(results: list[ProviderResult]) -> dict[str, int]:
    """Aggregates token usage across provider rows."""
    output: dict[str, int] = {}
    for item in results:
        for key, value in item.token_usage.items():
            output[key] = output.get(key, 0) + int(value)
    return output


def _strip_urls(text: str) -> tuple[str, list[str]]:
    """Removes raw URLs from TTS-facing answer text and returns extracted links."""
    links = URL_PATTERN.findall(text)
    if not links:
        return text, []
    stripped = URL_PATTERN.sub("", text)
    stripped = re.sub(r"\[([^\]]+)\]\(\s*\)", r"\1", stripped)
    stripped = re.sub(r"\(\s*\)", "", stripped)
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip(), _unique_preserve_order(links)


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output

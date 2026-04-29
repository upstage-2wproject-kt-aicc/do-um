"""Token pricing helpers for evaluation cost reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizedTokenUsage:
    """Provider-neutral input/output token usage."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ModelTokenPrice:
    """USD token price snapshot for one model."""

    model_id: str
    input_per_1m_usd: float
    output_per_1m_usd: float
    cached_input_per_1m_usd: float | None = None
    source_url: str = ""
    note: str = ""


@dataclass(frozen=True)
class EstimatedTokenCost:
    """Cost estimate for one model call."""

    model_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    usd: float
    price: ModelTokenPrice


MODEL_TOKEN_PRICES: dict[str, ModelTokenPrice] = {
    "solar-pro3": ModelTokenPrice(
        model_id="solar-pro3",
        input_per_1m_usd=0.15,
        cached_input_per_1m_usd=0.015,
        output_per_1m_usd=0.60,
        source_url="https://www.upstage.ai/pricing",
    ),
    "gpt-4o": ModelTokenPrice(
        model_id="gpt-4o",
        input_per_1m_usd=2.50,
        cached_input_per_1m_usd=1.25,
        output_per_1m_usd=10.00,
        source_url="https://platform.openai.com/docs/models/gpt-4o",
    ),
    "gpt-5.5": ModelTokenPrice(
        model_id="gpt-5.5",
        input_per_1m_usd=5.00,
        cached_input_per_1m_usd=0.50,
        output_per_1m_usd=30.00,
        source_url="https://openai.com/api/pricing/",
    ),
    "claude-sonnet-4-6": ModelTokenPrice(
        model_id="claude-sonnet-4-6",
        input_per_1m_usd=3.00,
        cached_input_per_1m_usd=0.30,
        output_per_1m_usd=15.00,
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        note="Claude Sonnet 4.6 tier.",
    ),
    "claude-opus-4-7": ModelTokenPrice(
        model_id="claude-opus-4-7",
        input_per_1m_usd=5.00,
        cached_input_per_1m_usd=0.50,
        output_per_1m_usd=25.00,
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        note="Claude Opus 4.7 tier.",
    ),
    "gemini-2.5-pro": ModelTokenPrice(
        model_id="gemini-2.5-pro",
        input_per_1m_usd=1.25,
        cached_input_per_1m_usd=0.31,
        output_per_1m_usd=10.00,
        source_url="https://cloud.google.com/vertex-ai/generative-ai/pricing",
        note="Vertex AI <=200K token text tier.",
    ),
    "gemini-3.1-pro-preview": ModelTokenPrice(
        model_id="gemini-3.1-pro-preview",
        input_per_1m_usd=3.60,
        cached_input_per_1m_usd=0.36,
        output_per_1m_usd=21.60,
        source_url="https://cloud.google.com/vertex-ai/generative-ai/pricing",
        note="Vertex AI priority <=200K token text tier.",
    ),
}


def normalize_token_usage(raw: dict[str, Any] | None) -> NormalizedTokenUsage:
    """Normalizes OpenAI-compatible, Anthropic, and Vertex token usage maps."""
    if not raw:
        return NormalizedTokenUsage()

    input_tokens = _first_int(
        raw,
        (
            "prompt_tokens",
            "input_tokens",
            "promptTokenCount",
            "prompt_token_count",
        ),
    )
    output_tokens = _first_int(
        raw,
        (
            "completion_tokens",
            "output_tokens",
            "candidatesTokenCount",
            "candidates_token_count",
        ),
    )
    total_tokens = _first_int(
        raw,
        (
            "total_tokens",
            "totalTokenCount",
            "total_token_count",
        ),
    )
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return NormalizedTokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def estimate_token_cost_usd(
    model_id: str,
    token_usage: dict[str, Any] | None,
) -> EstimatedTokenCost | None:
    """Returns an estimated USD token cost for a model call."""
    price = model_price(model_id)
    if price is None:
        return None
    usage = normalize_token_usage(token_usage)
    usd = (
        usage.input_tokens / 1_000_000 * price.input_per_1m_usd
        + usage.output_tokens / 1_000_000 * price.output_per_1m_usd
    )
    return EstimatedTokenCost(
        model_id=model_id,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        usd=usd,
        price=price,
    )


def model_price(model_id: str) -> ModelTokenPrice | None:
    """Looks up a model price by exact id, case-insensitively."""
    return MODEL_TOKEN_PRICES.get(model_id.strip().lower())


def _first_int(raw: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return 0

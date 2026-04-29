import pytest

from src.evaluation.pricing import (
    estimate_token_cost_usd,
    normalize_token_usage,
)


def test_normalize_token_usage_supports_openai_and_solar_shape() -> None:
    usage = normalize_token_usage(
        {
            "prompt_tokens": 1200,
            "completion_tokens": 300,
            "total_tokens": 1500,
        }
    )

    assert usage.input_tokens == 1200
    assert usage.output_tokens == 300
    assert usage.total_tokens == 1500


def test_normalize_token_usage_supports_anthropic_shape() -> None:
    usage = normalize_token_usage({"input_tokens": 900, "output_tokens": 200})

    assert usage.input_tokens == 900
    assert usage.output_tokens == 200
    assert usage.total_tokens == 1100


def test_normalize_token_usage_supports_vertex_shape() -> None:
    usage = normalize_token_usage(
        {
            "promptTokenCount": 1100,
            "candidatesTokenCount": 250,
            "totalTokenCount": 1350,
        }
    )

    assert usage.input_tokens == 1100
    assert usage.output_tokens == 250
    assert usage.total_tokens == 1350


def test_estimate_token_cost_usd_uses_model_price_snapshot() -> None:
    cost = estimate_token_cost_usd(
        "gpt-4o",
        {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 500_000,
        },
    )

    assert cost is not None
    assert cost.usd == pytest.approx(7.5)
    assert cost.input_tokens == 1_000_000
    assert cost.output_tokens == 500_000
    assert cost.price.input_per_1m_usd == 2.5
    assert cost.price.output_per_1m_usd == 10.0


def test_estimate_token_cost_usd_returns_none_for_unknown_model() -> None:
    assert estimate_token_cost_usd("unknown-model", {"prompt_tokens": 1}) is None

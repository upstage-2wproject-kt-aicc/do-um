"""HTTP clients for candidate generation and judge evaluation."""

from __future__ import annotations

import json
import os
import asyncio
import re
import time
from pathlib import Path
from typing import Any, Callable, TypeAlias

import httpx

from src.common.schemas import LLMRequest, LLMResponse
from src.evaluation.env import load_evaluation_env
from src.evaluation.schemas import (
    CandidateModel,
    EvaluationScenario,
    JUDGE_METRIC_NAMES,
    JudgeEvaluation,
    JudgeMetricScore,
    JudgeModel,
    RagasEvaluation,
)


load_evaluation_env()


EnvName: TypeAlias = str | tuple[str, ...]


OPENAI_COMPATIBLE_ENV: dict[str, dict[str, EnvName]] = {
    "solar": {
        "api_key": "LLM_SOLAR_API_KEY",
        "model": "LLM_SOLAR_MODEL",
        "base_url": "LLM_SOLAR_BASE_URL",
        "default_base_url": "https://api.upstage.ai/v1",
    },
    "gpt": {
        "api_key": "LLM_GPT_API_KEY",
        "model": "LLM_GPT_MODEL",
        "base_url": "LLM_GPT_BASE_URL",
        "default_base_url": "https://api.openai.com/v1",
    },
    "openai": {
        "api_key": ("JUDGE_OPENAI_API_KEY", "LLM_GPT_API_KEY"),
        "model": "JUDGE_OPENAI_MODEL",
        "base_url": "JUDGE_OPENAI_BASE_URL",
        "default_base_url": "https://api.openai.com/v1",
    },
    "grok": {
        "api_key": "LLM_GROK_API_KEY",
        "model": "LLM_GROK_MODEL",
        "base_url": "LLM_GROK_BASE_URL",
        "default_base_url": "https://api.x.ai/v1",
    },
}


class OpenAICompatibleChatClient:
    """Calls OpenAI-compatible chat completion APIs."""

    def __init__(self, timeout_s: float = 20.0) -> None:
        self.timeout_s = timeout_s

    async def generate(
        self, model: CandidateModel | JudgeModel, request: LLMRequest
    ) -> LLMResponse:
        env = _provider_env(model.provider, OPENAI_COMPATIBLE_ENV)
        api_key = _get_required_env(env["api_key"])
        model_id = model.model_id or _get_required_env(env["model"])
        base_url = os.getenv(env["base_url"], env["default_base_url"]).strip()
        started = time.perf_counter()
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": request.system_prompt or ""},
                {"role": "user", "content": request.prompt},
            ],
            "stream": False,
        }
        if _uses_max_completion_tokens(model_id):
            payload["max_completion_tokens"] = request.max_tokens
        else:
            payload["temperature"] = request.temperature
            payload["max_tokens"] = request.max_tokens
        if isinstance(model, JudgeModel) and model.provider in {"openai", "gpt"}:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            body = response.json()
        latency_ms = int((time.perf_counter() - started) * 1000)
        content = _openai_content(body)
        usage = _int_dict(body.get("usage", {}))
        finish_reason = None
        if body.get("choices"):
            finish_reason = body["choices"][0].get("finish_reason")
        return LLMResponse(
            session_id=request.session_id,
            provider=model.provider,
            text=content,
            latency_ms=latency_ms,
            ttft_ms=latency_ms,
            finish_reason=finish_reason,
            token_usage=usage,
        )


class AnthropicChatClient:
    """Calls Anthropic Messages API."""

    def __init__(
        self,
        api_key_env: EnvName = "LLM_CLAUDE_SONNET_API_KEY",
        base_url_env: EnvName = "LLM_CLAUDE_SONNET_BASE_URL",
        timeout_s: float = 20.0,
    ) -> None:
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.timeout_s = timeout_s

    async def generate(
        self, model: CandidateModel | JudgeModel, request: LLMRequest
    ) -> LLMResponse:
        api_key_env = self.api_key_env
        if model.provider == "anthropic":
            api_key_env = ("JUDGE_ANTHROPIC_API_KEY", "LLM_CLAUDE_SONNET_API_KEY")
        api_key = _get_required_env(api_key_env)
        base_url = _get_env(self.base_url_env, "https://api.anthropic.com/v1")
        started = time.perf_counter()
        payload = {
            "model": model.model_id,
            "max_tokens": request.max_tokens,
            "system": request.system_prompt or "",
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.temperature > 0:
            payload["temperature"] = request.temperature
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/messages",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            body = response.json()
        latency_ms = int((time.perf_counter() - started) * 1000)
        return LLMResponse(
            session_id=request.session_id,
            provider=model.provider,
            text=_anthropic_content(body),
            latency_ms=latency_ms,
            ttft_ms=latency_ms,
            finish_reason=body.get("stop_reason"),
            token_usage=_int_dict(body.get("usage", {})),
        )


class GoogleVertexGeminiChatClient:
    """Calls Gemini through Vertex AI using Google ADC credentials."""

    def __init__(
        self,
        project_env: EnvName = "GOOGLE_CLOUD_PROJECT",
        location_env: EnvName = "GOOGLE_CLOUD_LOCATION",
        default_location: str = "global",
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.project_env = project_env
        self.location_env = location_env
        self.default_location = default_location
        self.client_factory = client_factory

    async def generate(
        self, model: CandidateModel | JudgeModel, request: LLMRequest
    ) -> LLMResponse:
        project = _get_required_env(self.project_env)
        location = _get_env(self.location_env, self.default_location)
        started = time.perf_counter()
        response = await asyncio.to_thread(
            self._generate_content,
            project,
            location,
            model.model_id,
            request,
            isinstance(model, JudgeModel),
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return LLMResponse(
            session_id=request.session_id,
            provider=model.provider,
            text=_vertex_text(response),
            latency_ms=latency_ms,
            ttft_ms=latency_ms,
            finish_reason=_vertex_finish_reason(response),
            token_usage=_vertex_usage(response),
        )

    def _generate_content(
        self,
        project: str,
        location: str,
        model_id: str,
        request: LLMRequest,
        json_response: bool,
    ) -> Any:
        client_factory = self.client_factory or _load_genai_client
        client = client_factory(vertexai=True, project=project, location=location)
        return client.models.generate_content(
            model=model_id,
            contents=request.prompt,
            config=_build_vertex_config(
                request,
                json_response=json_response,
                thinking_budget=_vertex_thinking_budget(model_id),
            ),
        )


class CompositeCandidateClient:
    """Routes candidate model calls to provider-specific clients."""

    def __init__(self) -> None:
        self.openai_compatible = OpenAICompatibleChatClient()
        self.anthropic = AnthropicChatClient()
        self.vertex_gemini = GoogleVertexGeminiChatClient()

    async def generate(self, model: CandidateModel, request: LLMRequest) -> LLMResponse:
        if model.provider in {"solar", "gpt", "openai", "grok"}:
            return await self.openai_compatible.generate(model, request)
        if model.provider in {"claude-sonnet", "anthropic"}:
            return await self.anthropic.generate(model, request)
        if model.provider == "google":
            return await self.vertex_gemini.generate(model, request)
        raise ValueError(f"Unsupported candidate provider: {model.provider}")


class CompositeJudgeClient:
    """Routes judge calls and parses judge JSON output."""

    def __init__(self, prompt_path: str | Path | None = None) -> None:
        self.prompt_path = prompt_path or Path(__file__).parent / "prompts" / "judge_v1.md"
        self.openai = OpenAICompatibleChatClient()
        self.anthropic = AnthropicChatClient(
            api_key_env=("JUDGE_ANTHROPIC_API_KEY", "LLM_CLAUDE_SONNET_API_KEY")
        )
        self.vertex_gemini = GoogleVertexGeminiChatClient()

    async def evaluate(
        self,
        judge: JudgeModel,
        scenario: EvaluationScenario,
        answer: LLMResponse,
    ) -> JudgeEvaluation:
        request = LLMRequest(
            session_id=scenario.scenario_id,
            system_prompt=Path(self.prompt_path).read_text(encoding="utf-8"),
            prompt=build_judge_user_prompt(scenario, answer),
            temperature=0.0,
            max_tokens=2400,
        )
        if judge.provider in {"openai", "gpt"}:
            response = await self.openai.generate(judge, request)
        elif judge.provider in {"anthropic", "claude"}:
            response = await self.anthropic.generate(judge, request)
        elif judge.provider in {"google", "gemini"}:
            response = await self.vertex_gemini.generate(judge, request)
        else:
            raise ValueError(f"Unsupported judge provider: {judge.provider}")
        return parse_judge_json(judge.model_id, response.text)


class NullRagasClient:
    """RAGAS placeholder for runs before the ragas dependency is enabled."""

    async def evaluate(
        self, scenario: EvaluationScenario, answer: LLMResponse
    ) -> RagasEvaluation:
        return RagasEvaluation(
            faithfulness=None,
            answer_relevancy=None,
            details={"status": "not_configured"},
        )


def build_judge_user_prompt(
    scenario: EvaluationScenario, answer: LLMResponse
) -> str:
    """Builds the user payload given to judge models."""
    payload = {
        "user_query": scenario.user_query,
        "context": scenario.retrieved_context,
        "candidate_answer": answer.text,
        "expected_route": scenario.metadata.get("expected_route", ""),
        "reference_answer": scenario.reference_answer,
        "context_metadata": scenario.metadata,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_judge_json(judge_model: str, text: str) -> JudgeEvaluation:
    """Parses judge JSON and validates the required metric keys."""
    data = _load_judge_json(text)
    metrics: dict[str, JudgeMetricScore] = {}
    for metric_name in JUDGE_METRIC_NAMES:
        raw_metric = data.get(metric_name)
        if not isinstance(raw_metric, dict):
            raise ValueError(f"Missing judge metric: {metric_name}")
        metrics[metric_name] = JudgeMetricScore.model_validate(raw_metric)
    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    return JudgeEvaluation(judge_model=judge_model, metrics=metrics, summary=summary)


def _provider_env(
    provider: str, env_map: dict[str, dict[str, EnvName]]
) -> dict[str, EnvName]:
    try:
        return env_map[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc


def _get_required_env(name: EnvName) -> str:
    value = _get_env(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {_env_label(name)}")
    return value


def _get_env(name: EnvName, default: str = "") -> str:
    names = (name,) if isinstance(name, str) else name
    for env_name in names:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return default


def _env_label(name: EnvName) -> str:
    if isinstance(name, str):
        return name
    return " or ".join(name)


def _uses_max_completion_tokens(model_id: str) -> bool:
    normalized = model_id.lower()
    return normalized.startswith(("gpt-5", "o1", "o3", "o4"))


def _load_genai_client(**kwargs: Any) -> Any:
    from google import genai

    return genai.Client(**kwargs)


def _build_vertex_config(
    request: LLMRequest,
    json_response: bool = False,
    thinking_budget: int | None = None,
) -> Any:
    from google.genai import types

    kwargs: dict[str, Any] = {
        "system_instruction": request.system_prompt or None,
        "temperature": request.temperature,
        "max_output_tokens": request.max_tokens,
    }
    if json_response:
        kwargs["response_mime_type"] = "application/json"
    if thinking_budget is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )
    return types.GenerateContentConfig(
        **kwargs,
    )


def _vertex_thinking_budget(model_id: str) -> int | None:
    """Returns a small thinking budget for Gemini reasoning models.

    Gemini 2.5+ counts hidden thinking tokens against max output tokens. Keeping
    the budget bounded prevents short customer-facing answers from being cut off.
    """
    normalized = model_id.lower()
    if not normalized.startswith(("gemini-2.5", "gemini-3")):
        return None
    raw = os.getenv("GOOGLE_VERTEX_THINKING_BUDGET", "128").strip().lower()
    if raw in {"", "none", "off", "disabled"}:
        return None
    return int(raw)


def _openai_content(body: dict[str, Any]) -> str:
    choices = body.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content", ""))


def _anthropic_content(body: dict[str, Any]) -> str:
    parts = body.get("content", [])
    texts = [
        str(part.get("text", ""))
        for part in parts
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    return "".join(texts)


def _vertex_text(response: Any) -> str:
    return str(getattr(response, "text", "") or "")


def _vertex_finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    return str(reason) if reason is not None else None


def _vertex_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    fields = {
        "promptTokenCount": getattr(usage, "prompt_token_count", None),
        "candidatesTokenCount": getattr(usage, "candidates_token_count", None),
        "totalTokenCount": getattr(usage, "total_token_count", None),
    }
    return {
        key: int(value)
        for key, value in fields.items()
        if isinstance(value, (int, float))
    }


def _int_dict(raw: object) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    return {key: int(value) for key, value in raw.items() if isinstance(value, (int, float))}


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _load_judge_json(text: str) -> dict[str, Any]:
    stripped = _extract_json_object(_strip_json_fence(text))
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        data = json.loads(_remove_trailing_json_commas(stripped))
    if not isinstance(data, dict):
        raise ValueError("Judge response must be a JSON object.")
    return data


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text.strip()
    return text[start : end + 1].strip()


def _remove_trailing_json_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)

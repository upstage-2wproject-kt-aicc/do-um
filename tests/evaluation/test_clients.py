import json

import pytest

from src.common.schemas import LLMRequest, LLMResponse
from src.evaluation.clients import (
    AnthropicChatClient,
    GoogleVertexGeminiChatClient,
    OpenAICompatibleChatClient,
    build_judge_user_prompt,
    parse_judge_json,
)
from src.evaluation.schemas import CandidateModel, EvaluationScenario, JudgeMetricScore


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.text)

    def json(self) -> dict:
        return self._payload


class FakeAsyncClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.requests: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url: str, json: dict, headers: dict):
        self.requests.append({"url": url, "json": json, "headers": headers})
        return self.response


@pytest.mark.asyncio
async def test_openai_compatible_client_generates_response(monkeypatch) -> None:
    fake = FakeAsyncClient(
        FakeResponse(
            {
                "choices": [{"message": {"content": "답변입니다."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        )
    )
    monkeypatch.setenv("LLM_GPT_API_KEY", "test-key")
    monkeypatch.setenv("LLM_GPT_MODEL", "gpt-test")
    monkeypatch.setattr("src.evaluation.clients.httpx.AsyncClient", lambda **_: fake)

    client = OpenAICompatibleChatClient()
    response = await client.generate(
        CandidateModel(provider="gpt", model_id="gpt-test"),
        LLMRequest(session_id="s1", prompt="질문", system_prompt="시스템"),
    )

    assert response.text == "답변입니다."
    assert response.provider == "gpt"
    assert response.token_usage["prompt_tokens"] == 3
    assert fake.requests[0]["json"]["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_openai_compatible_client_uses_max_completion_tokens_for_gpt_5_5(
    monkeypatch,
) -> None:
    fake = FakeAsyncClient(
        FakeResponse(
            {
                "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            }
        )
    )
    monkeypatch.setenv("LLM_GPT_API_KEY", "test-key")
    monkeypatch.setattr("src.evaluation.clients.httpx.AsyncClient", lambda **_: fake)

    await OpenAICompatibleChatClient().generate(
        CandidateModel(provider="gpt", model_id="gpt-5.5"),
        LLMRequest(
            session_id="s1",
            prompt="질문",
            system_prompt="시스템",
            max_tokens=8,
        ),
    )

    payload = fake.requests[0]["json"]
    assert payload["max_completion_tokens"] == 8
    assert "max_tokens" not in payload
    assert "temperature" not in payload


@pytest.mark.asyncio
async def test_openai_judge_client_uses_gpt_key_fallback(monkeypatch) -> None:
    fake = FakeAsyncClient(
        FakeResponse(
            {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            }
        )
    )
    monkeypatch.delenv("JUDGE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_GPT_API_KEY", "fallback-gpt-key")
    monkeypatch.setenv("JUDGE_OPENAI_MODEL", "gpt-judge")
    monkeypatch.setattr("src.evaluation.clients.httpx.AsyncClient", lambda **_: fake)

    await OpenAICompatibleChatClient().generate(
        CandidateModel(provider="openai", model_id="gpt-judge"),
        LLMRequest(session_id="s1", prompt="질문", system_prompt="시스템"),
    )

    assert fake.requests[0]["headers"]["Authorization"] == "Bearer fallback-gpt-key"


@pytest.mark.asyncio
async def test_anthropic_client_generates_response(monkeypatch) -> None:
    fake = FakeAsyncClient(
        FakeResponse(
            {
                "content": [{"type": "text", "text": "클로드 답변"}],
                "usage": {"input_tokens": 4, "output_tokens": 2},
                "stop_reason": "end_turn",
            }
        )
    )
    monkeypatch.setenv("LLM_CLAUDE_SONNET_API_KEY", "test-key")
    monkeypatch.setenv("LLM_CLAUDE_SONNET_MODEL", "claude-test")
    monkeypatch.setattr("src.evaluation.clients.httpx.AsyncClient", lambda **_: fake)

    client = AnthropicChatClient()
    response = await client.generate(
        CandidateModel(provider="claude-sonnet", model_id="claude-test"),
        LLMRequest(session_id="s1", prompt="질문", system_prompt="시스템"),
    )

    assert response.text == "클로드 답변"
    assert response.provider == "claude-sonnet"
    assert fake.requests[0]["headers"]["anthropic-version"] == "2023-06-01"
    assert "temperature" not in fake.requests[0]["json"]


@pytest.mark.asyncio
async def test_anthropic_judge_client_uses_candidate_key_fallback(monkeypatch) -> None:
    fake = FakeAsyncClient(
        FakeResponse(
            {
                "content": [{"type": "text", "text": "클로드 평가"}],
                "usage": {"input_tokens": 4, "output_tokens": 2},
                "stop_reason": "end_turn",
            }
        )
    )
    monkeypatch.delenv("JUDGE_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LLM_CLAUDE_SONNET_API_KEY", "fallback-claude-key")
    monkeypatch.setattr("src.evaluation.clients.httpx.AsyncClient", lambda **_: fake)

    await AnthropicChatClient(api_key_env="JUDGE_ANTHROPIC_API_KEY").generate(
        CandidateModel(provider="anthropic", model_id="claude-judge"),
        LLMRequest(session_id="s1", prompt="질문", system_prompt="시스템"),
    )

    assert fake.requests[0]["headers"]["x-api-key"] == "fallback-claude-key"


@pytest.mark.asyncio
async def test_google_vertex_gemini_client_uses_project_location_and_model(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            calls["generate_content"] = kwargs
            return type(
                "FakeVertexResponse",
                (),
                {
                    "text": "버텍스 답변",
                    "usage_metadata": type(
                        "FakeUsage",
                        (),
                        {
                            "prompt_token_count": 5,
                            "candidates_token_count": 2,
                            "total_token_count": 7,
                        },
                    )(),
                    "candidates": [
                        type(
                            "FakeCandidate",
                            (),
                            {"finish_reason": "STOP"},
                        )()
                    ],
                },
            )()

    class FakeGenaiClient:
        def __init__(self, **kwargs) -> None:
            calls["client"] = kwargs
            self.models = FakeModels()

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project-test")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")

    response = await GoogleVertexGeminiChatClient(
        client_factory=FakeGenaiClient
    ).generate(
        CandidateModel(provider="google", model_id="gemini-2.5-pro"),
        LLMRequest(
            session_id="s1",
            prompt="질문",
            system_prompt="시스템",
            max_tokens=64,
        ),
    )

    assert calls["client"] == {
        "vertexai": True,
        "project": "project-test",
        "location": "global",
    }
    assert calls["generate_content"]["model"] == "gemini-2.5-pro"
    assert calls["generate_content"]["contents"] == "질문"
    assert response.text == "버텍스 답변"
    assert response.token_usage == {
        "promptTokenCount": 5,
        "candidatesTokenCount": 2,
        "totalTokenCount": 7,
    }


def test_parse_judge_json_validates_metric_scores() -> None:
    evaluation = parse_judge_json(
        judge_model="judge-a",
        text=json.dumps(
            {
                "answer_accuracy": {"score": 5, "reason": "정확합니다."},
                "grounded_response": {"score": 4, "reason": "대체로 근거가 있습니다."},
                "safety_conservativeness": {"score": 5, "reason": "안전합니다."},
                "handoff_judgment": {"score": 3, "reason": "보통입니다."},
                "user_guidance_quality": {"score": 4, "reason": "명확합니다."},
                "summary": {"risks": []},
            }
        ),
    )

    assert isinstance(evaluation.metrics["answer_accuracy"], JudgeMetricScore)
    assert evaluation.metrics["answer_accuracy"].score == 5
    assert evaluation.summary == {"risks": []}


def test_build_judge_user_prompt_includes_optional_reference() -> None:
    prompt = build_judge_user_prompt(
        scenario=EvaluationScenario(
            scenario_id="s1",
            user_query="질문",
            intent="조회형",
            retrieved_context="문서",
            reference_answer="기준 답변",
        ),
        answer=LLMResponse(session_id="s1", provider="gpt", text="후보 답변", latency_ms=0),
    )

    assert '"reference_answer": "기준 답변"' in prompt
    assert '"candidate_answer": "후보 답변"' in prompt

import json

import httpx
import pytest

from src.common.schemas import LLMRequest, LLMResponse
from src.evaluation.clients import (
    AnthropicChatClient,
    CompositeJudgeClient,
    GoogleVertexGeminiChatClient,
    OpenAICompatibleChatClient,
    build_judge_user_prompt,
    parse_judge_json,
)
from src.evaluation.schemas import CandidateModel, EvaluationScenario, JudgeMetricScore, JudgeModel


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.test")
            response = httpx.Response(
                self.status_code,
                request=request,
                text=self.text,
            )
            raise httpx.HTTPStatusError(self.text, request=request, response=response)

    def json(self) -> dict:
        return self._payload


class FakeAsyncClient:
    def __init__(self, response: FakeResponse | list[FakeResponse]) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.requests: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(self, url: str, json: dict, headers: dict):
        self.requests.append({"url": url, "json": json, "headers": headers})
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


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
async def test_openai_judge_client_requests_json_object(monkeypatch) -> None:
    fake = FakeAsyncClient(
        FakeResponse(
            {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            }
        )
    )
    monkeypatch.setenv("JUDGE_OPENAI_API_KEY", "judge-key")
    monkeypatch.setattr("src.evaluation.clients.httpx.AsyncClient", lambda **_: fake)

    await OpenAICompatibleChatClient().generate(
        JudgeModel(provider="openai", model_id="gpt-judge"),
        LLMRequest(session_id="s1", prompt="질문", system_prompt="시스템"),
    )

    assert fake.requests[0]["json"]["response_format"] == {"type": "json_object"}


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
async def test_anthropic_client_retries_rate_limit(monkeypatch) -> None:
    fake = FakeAsyncClient(
        [
            FakeResponse({"error": "rate limited"}, status_code=429),
            FakeResponse(
                {
                    "content": [{"type": "text", "text": "재시도 성공"}],
                    "usage": {"input_tokens": 4, "output_tokens": 2},
                    "stop_reason": "end_turn",
                }
            ),
        ]
    )
    sleeps: list[float] = []
    monkeypatch.setenv("LLM_CLAUDE_SONNET_API_KEY", "test-key")
    monkeypatch.setattr("src.evaluation.clients.httpx.AsyncClient", lambda **_: fake)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("src.evaluation.clients.asyncio.sleep", fake_sleep)

    response = await AnthropicChatClient().generate(
        CandidateModel(provider="claude-sonnet", model_id="claude-test"),
        LLMRequest(session_id="s1", prompt="질문", system_prompt="시스템"),
    )

    assert response.text == "재시도 성공"
    assert len(fake.requests) == 2
    assert sleeps == [1.0]


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

    assert calls["client"]["vertexai"] is True
    assert calls["client"]["project"] == "project-test"
    assert calls["client"]["location"] == "global"
    assert calls["client"]["http_options"].timeout == 20000
    assert calls["generate_content"]["model"] == "gemini-2.5-pro"
    assert calls["generate_content"]["contents"] == "질문"
    assert response.text == "버텍스 답변"
    assert response.token_usage == {
        "promptTokenCount": 5,
        "candidatesTokenCount": 2,
        "totalTokenCount": 7,
    }
    assert getattr(calls["generate_content"]["config"], "thinking_config").thinking_budget == 128


@pytest.mark.asyncio
async def test_google_vertex_judge_client_requests_json_response(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            calls["generate_content"] = kwargs
            return type(
                "FakeVertexResponse",
                (),
                {
                    "text": "{}",
                    "usage_metadata": None,
                    "candidates": [],
                },
            )()

    class FakeGenaiClient:
        def __init__(self, **kwargs) -> None:
            calls["client_timeout"] = kwargs["http_options"].timeout
            self.models = FakeModels()

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project-test")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")

    await GoogleVertexGeminiChatClient(
        timeout_s=60.0,
        client_factory=FakeGenaiClient,
    ).generate(
        JudgeModel(provider="google", model_id="gemini-3.1-pro-preview"),
        LLMRequest(session_id="s1", prompt="질문", system_prompt="시스템"),
    )

    assert calls.get("client_timeout") == 60000
    assert getattr(calls["generate_content"]["config"], "response_mime_type") == "application/json"
    assert getattr(calls["generate_content"]["config"], "thinking_config").thinking_budget == 128


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
        token_usage={"prompt_tokens": 10, "completion_tokens": 5},
    )

    assert isinstance(evaluation.metrics["answer_accuracy"], JudgeMetricScore)
    assert evaluation.metrics["answer_accuracy"].score == 5
    assert evaluation.summary == {"risks": []}
    assert evaluation.token_usage == {"prompt_tokens": 10, "completion_tokens": 5}


def test_parse_judge_json_recovers_fenced_json_with_trailing_commas() -> None:
    evaluation = parse_judge_json(
        judge_model="judge-a",
        text="""
        ```json
        {
          "answer_accuracy": {"score": 5, "reason": "정확합니다."},
          "grounded_response": {"score": 4, "reason": "대체로 근거가 있습니다."},
          "safety_conservativeness": {"score": 5, "reason": "안전합니다."},
          "handoff_judgment": {"score": 3, "reason": "보통입니다."},
          "user_guidance_quality": {"score": 4, "reason": "명확합니다."},
          "summary": {"risks": [],},
        }
        ```
        """,
    )

    assert evaluation.metrics["grounded_response"].score == 4
    assert evaluation.summary == {"risks": []}


def test_parse_judge_json_supports_comparative_10_rubric() -> None:
    metric_names = (
        "intent_fit",
        "accuracy",
        "groundedness",
        "safety_conservatism",
        "handoff_appropriateness",
        "guidance_quality",
    )
    evaluation = parse_judge_json(
        judge_model="judge-a",
        text=json.dumps(
            {
                "intent_fit": {"score": 8, "reason": "의도에 맞습니다."},
                "accuracy": {"score": 7, "reason": "대체로 정확합니다."},
                "groundedness": {
                    "score": 9,
                    "reason": "문서에 근거합니다.",
                    "unsupported_claims": [],
                },
                "safety_conservatism": {
                    "score": 8,
                    "reason": "보수적입니다.",
                    "risk_flags": [],
                },
                "handoff_appropriateness": {
                    "score": 7,
                    "reason": "이관 판단이 무난합니다.",
                    "should_handoff": False,
                },
                "guidance_quality": {"score": 8, "reason": "안내가 명확합니다."},
                "flags": {"unsupported_claim": False},
                "summary": {
                    "overall_profile": "균형형",
                    "strongest_dimension": "groundedness",
                    "weakest_dimension": "accuracy",
                },
            }
        ),
        metric_names=metric_names,
        score_min=1,
        score_max=10,
    )

    assert evaluation.metrics["groundedness"].score == 9
    assert evaluation.flags == {"unsupported_claim": False}
    assert evaluation.summary["overall_profile"] == "균형형"


def test_parse_judge_json_rejects_score_outside_selected_rubric() -> None:
    with pytest.raises(ValueError, match="between 1 and 5"):
        parse_judge_json(
            judge_model="judge-a",
            text=json.dumps(
                {
                    "answer_accuracy": {"score": 8, "reason": "범위 초과"},
                    "grounded_response": {"score": 4, "reason": "대체로 근거가 있습니다."},
                    "safety_conservativeness": {"score": 5, "reason": "안전합니다."},
                    "handoff_judgment": {"score": 3, "reason": "보통입니다."},
                    "user_guidance_quality": {"score": 4, "reason": "명확합니다."},
                }
            ),
        )


def test_build_judge_user_prompt_includes_optional_reference() -> None:
    prompt = build_judge_user_prompt(
        scenario=EvaluationScenario(
            scenario_id="s1",
            user_query="질문",
            intent="조회형",
            retrieved_context="문서",
            reference_answer="기준 답변",
            metadata={"source_url": "https://example.com/source"},
        ),
        answer=LLMResponse(session_id="s1", provider="gpt", text="후보 답변", latency_ms=0),
    )

    assert '"reference_answer": "기준 답변"' in prompt
    assert '"candidate_reference_links": [\n    "https://example.com/source"\n  ]' in prompt
    assert '"candidate_answer": "후보 답변"' in prompt


def test_build_judge_user_prompt_supports_comparative_score_scale() -> None:
    prompt = build_judge_user_prompt(
        scenario=EvaluationScenario(
            scenario_id="s1",
            user_query="질문",
            intent="설명형",
        ),
        answer=LLMResponse(session_id="s1", provider="gpt", text="후보 답변", latency_ms=0),
        score_scale_label="1_to_10",
        primary_score_source="llm_as_a_judge_6_comparative_metrics",
    )

    assert '"score_scale": "1_to_10"' in prompt
    assert '"primary_score_source": "llm_as_a_judge_6_comparative_metrics"' in prompt


def test_composite_judge_client_defaults_to_v2_prompt() -> None:
    client = CompositeJudgeClient()

    assert client.prompt_path.name == "judge_v2.md"
    assert client.openai.timeout_s == 60.0
    assert client.anthropic.timeout_s == 60.0


def test_composite_judge_client_limits_anthropic_concurrency() -> None:
    client = CompositeJudgeClient(anthropic_concurrency=2, google_concurrency=3)

    assert client.anthropic_semaphore._value == 2
    assert client.google_semaphore._value == 3


def test_build_judge_user_prompt_separates_candidate_workflow_messages() -> None:
    prompt = build_judge_user_prompt(
        scenario=EvaluationScenario(
            scenario_id="s1",
            user_query="자동이체 바꾸려면 어떻게 해요?",
            intent="절차형",
            retrieved_context="자동이체는 앱에서 변경할 수 있습니다.",
        ),
        answer=LLMResponse(session_id="s1", provider="gpt", text="후보 답변", latency_ms=0),
    )

    assert '"workflow_prompt_given_to_candidate"' in prompt
    assert '"system_message"' in prompt
    assert '"user_message"' in prompt
    assert 'Route=PROCEDURE' in prompt
    assert '"candidate_answer": "후보 답변"' in prompt

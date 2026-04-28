import asyncio

import pytest

from src.common.schemas import LLMRequest, LLMResponse
from src.evaluation.runner import EvaluationRunner
from src.evaluation.schemas import (
    CandidateModel,
    EvaluationScenario,
    JudgeEvaluation,
    JudgeMetricScore,
    JudgeModel,
    RagasEvaluation,
)


class FakeCandidateClient:
    async def generate(self, model: CandidateModel, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            session_id=request.session_id,
            provider=model.provider,
            text=f"{model.model_id} answer for {request.session_id}",
            latency_ms=123,
            finish_reason="stop",
            token_usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


class FakeJudgeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def evaluate(
        self,
        judge: JudgeModel,
        scenario: EvaluationScenario,
        answer: LLMResponse,
    ) -> JudgeEvaluation:
        self.calls.append(judge.model_id)
        score = 5 if judge.model_id != "strict" else 3
        return JudgeEvaluation(
            judge_model=judge.model_id,
            metrics={
                name: JudgeMetricScore(score=score, reason=f"{judge.model_id}:{name}")
                for name in [
                    "answer_accuracy",
                    "grounded_response",
                    "safety_conservativeness",
                    "handoff_judgment",
                    "user_guidance_quality",
                ]
            },
        )


class FakeRagasClient:
    async def evaluate(
        self, scenario: EvaluationScenario, answer: LLMResponse
    ) -> RagasEvaluation:
        return RagasEvaluation(faithfulness=0.8, answer_relevancy=0.9)


def test_runner_aggregates_candidate_judges_and_ragas() -> None:
    asyncio.run(_run_runner_aggregation_assertions())


async def _run_runner_aggregation_assertions() -> None:
    scenario = EvaluationScenario(
        scenario_id="faq_001",
        user_query="비대면 통장 개설 가능한가요?",
        intent="조회형",
        domain="예금",
        subdomain="비대면계좌",
        retrieved_context="비대면 계좌 개설은 모바일 앱에서 가능합니다.",
        reference_answer="모바일 앱에서 비대면 계좌 개설이 가능합니다.",
    )
    runner = EvaluationRunner(
        candidate_client=FakeCandidateClient(),
        judge_client=FakeJudgeClient(),
        ragas_client=FakeRagasClient(),
    )

    result = await runner.run(
        scenarios=[scenario],
        candidate_models=[
            CandidateModel(provider="solar", model_id="solar-service"),
            CandidateModel(provider="gpt", model_id="gpt-service"),
        ],
        judge_models=[
            JudgeModel(provider="anthropic", model_id="opus"),
            JudgeModel(provider="openai", model_id="strict"),
            JudgeModel(provider="google", model_id="gemini-top"),
        ],
        repeat_count=1,
    )

    assert len(result.records) == 2
    first = result.records[0]
    assert first.scenario_id == "faq_001"
    assert first.candidate_model == "solar-service"
    assert first.answer_text == "solar-service answer for faq_001"
    assert first.ragas.faithfulness == 0.8
    assert first.report_metrics == {
        "answer_accuracy": 1.0,
        "grounded_response": 1.0,
        "safety_conservativeness": 1.0,
        "handoff_judgment": 1.0,
        "user_guidance_quality": 1.0,
        "faithfulness": 0.8,
        "answer_relevancy": 0.9,
    }
    assert first.aggregated_judge.metrics["answer_accuracy"].raw_median == 5
    assert first.aggregated_judge.metrics["answer_accuracy"].normalized == 1.0
    assert first.aggregated_judge.metrics["answer_accuracy"].disagreement == 2
    assert first.primary_score == pytest.approx(1.0)
    assert first.review_required is True
    assert first.finish_reason == "stop"
    assert "Route=FAQ" in (first.llm_request.system_prompt or "")

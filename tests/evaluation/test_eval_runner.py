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


class ConcurrentCandidateClient:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def generate(self, model: CandidateModel, request: LLMRequest) -> LLMResponse:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return LLMResponse(
            session_id=request.session_id,
            provider=model.provider,
            text=f"{model.model_id} answer",
            latency_ms=1,
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


class ConcurrentJudgeClient(FakeJudgeClient):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0

    async def evaluate(
        self,
        judge: JudgeModel,
        scenario: EvaluationScenario,
        answer: LLMResponse,
    ) -> JudgeEvaluation:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return await super().evaluate(judge, scenario, answer)


class FakeRagasClient:
    async def evaluate(
        self, scenario: EvaluationScenario, answer: LLMResponse
    ) -> RagasEvaluation:
        return RagasEvaluation(faithfulness=0.8, answer_relevancy=0.9)


class ComparativeJudgeClient:
    async def evaluate(
        self,
        judge: JudgeModel,
        scenario: EvaluationScenario,
        answer: LLMResponse,
    ) -> JudgeEvaluation:
        return JudgeEvaluation(
            judge_model=judge.model_id,
            metrics={
                name: JudgeMetricScore(score=9, reason=f"{judge.model_id}:{name}")
                for name in [
                    "intent_fit",
                    "accuracy",
                    "groundedness",
                    "safety_conservatism",
                    "handoff_appropriateness",
                    "guidance_quality",
                ]
            },
        )


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


def test_runner_supports_comparative_10_without_ragas_report_metrics() -> None:
    asyncio.run(_run_runner_comparative_10_assertions())


async def _run_runner_comparative_10_assertions() -> None:
    metric_names = (
        "intent_fit",
        "accuracy",
        "groundedness",
        "safety_conservatism",
        "handoff_appropriateness",
        "guidance_quality",
    )
    runner = EvaluationRunner(
        candidate_client=FakeCandidateClient(),
        judge_client=ComparativeJudgeClient(),
        ragas_client=FakeRagasClient(),
        judge_metric_names=metric_names,
        judge_score_min=1,
        judge_score_max=10,
        include_ragas=False,
        report_normalized_scores=False,
        review_risk_metric_names=(
            "groundedness",
            "safety_conservatism",
            "handoff_appropriateness",
        ),
    )

    result = await runner.run(
        scenarios=[
            EvaluationScenario(
                scenario_id="faq_001",
                user_query="질문",
                intent="설명형",
                retrieved_context="문서",
            )
        ],
        candidate_models=[CandidateModel(provider="gpt", model_id="gpt-service")],
        judge_models=[JudgeModel(provider="openai", model_id="judge")],
    )

    record = result.records[0]
    assert record.aggregated_judge.metrics["intent_fit"].raw_median == 9
    assert record.aggregated_judge.metrics["intent_fit"].normalized == pytest.approx(8 / 9)
    assert set(record.report_metrics) == set(metric_names)
    assert record.report_metrics["intent_fit"] == 9
    assert record.primary_score == pytest.approx(9)


def test_runner_records_timing_breakdown() -> None:
    asyncio.run(_run_runner_timing_assertions())


async def _run_runner_timing_assertions() -> None:
    scenario = EvaluationScenario(
        scenario_id="faq_001",
        user_query="질문",
        intent="설명형",
        retrieved_context="문서",
    )
    runner = EvaluationRunner(
        candidate_client=FakeCandidateClient(),
        judge_client=FakeJudgeClient(),
        ragas_client=FakeRagasClient(),
        timer=iter([0.0, 0.1, 0.2, 0.5, 0.6, 0.9, 1.0, 1.2, 1.4, 1.6]).__next__,
    )

    result = await runner.run(
        scenarios=[scenario],
        candidate_models=[CandidateModel(provider="solar", model_id="solar-service")],
        judge_models=[JudgeModel(provider="openai", model_id="strict")],
        repeat_count=1,
    )

    assert result.records[0].timing_ms["candidate"] == 100
    assert result.records[0].timing_ms["judge"] >= 0
    assert result.records[0].timing_ms["ragas"] >= 0
    assert result.records[0].timing_ms["evaluation_parallel"] >= max(
        result.records[0].timing_ms["judge"],
        result.records[0].timing_ms["ragas"],
    )
    assert result.records[0].timing_ms["total"] >= (
        result.records[0].timing_ms["candidate"]
        + result.records[0].timing_ms["evaluation_parallel"]
    )


def test_runner_evaluates_candidates_and_judges_concurrently() -> None:
    asyncio.run(_run_runner_concurrency_assertions())


async def _run_runner_concurrency_assertions() -> None:
    scenario = EvaluationScenario(
        scenario_id="faq_001",
        user_query="질문",
        intent="설명형",
        retrieved_context="문서",
    )
    candidate_client = ConcurrentCandidateClient()
    judge_client = ConcurrentJudgeClient()
    runner = EvaluationRunner(
        candidate_client=candidate_client,
        judge_client=judge_client,
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
    assert candidate_client.max_active == 2
    assert judge_client.max_active == 6

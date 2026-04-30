import asyncio
from pathlib import Path

import pytest

from src.common.schemas import LLMRequest, LLMResponse, WorkflowRoutingInput
from src.evaluation.rubrics import COMPARATIVE_JUDGE_METRIC_NAMES
from src.evaluation.schemas import (
    CandidateModel,
    JudgeEvaluation,
    JudgeMetricScore,
    JudgeModel,
)
from src.workflow.online_evaluation import (
    FileOnlineEvaluationStore,
    OnlineEvaluationService,
    build_workflow_result_event,
)


class ConcurrentCandidateClient:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def generate(self, model: CandidateModel, request: LLMRequest) -> LLMResponse:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            return LLMResponse(
                session_id=request.session_id,
                provider=model.provider,
                text=f"{model.provider} 답변",
                latency_ms=10,
                ttft_ms=10,
                finish_reason="stop",
                token_usage={"input_tokens": 10, "output_tokens": 5},
            )
        finally:
            self.active -= 1


class StaticJudgeClient:
    async def evaluate(
        self,
        judge: JudgeModel,
        scenario,
        answer: LLMResponse,
    ) -> JudgeEvaluation:
        return JudgeEvaluation(
            judge_model=judge.model_id,
            metrics={
                name: JudgeMetricScore(score=8, reason=f"{answer.provider}:{name}")
                for name in COMPARATIVE_JUDGE_METRIC_NAMES
            },
            summary={
                "overall_profile": f"{answer.provider} 안정적",
                "strongest_dimension": "accuracy",
                "weakest_dimension": "guidance_quality",
            },
            flags={"unsupported_claim": False, "missed_handoff": False},
        )


def _payload() -> WorkflowRoutingInput:
    return WorkflowRoutingInput.model_validate(
        {
            "session_id": "online-session-1",
            "original_query": "고정금리와 변동금리 차이를 알려주세요.",
            "routing_info": {
                "intent": "설명형",
                "domain": "금융상담",
                "subdomain": "대출/금리",
                "router_confidence": 0.95,
                "metadata": {
                    "risk_level": "중간",
                    "handoff_required": "N",
                    "source_url": "https://example.com/source",
                },
            },
            "internal_context": [
                {
                    "source": "nlu_rag",
                    "content": "고정금리는 약정 기간 동안 금리가 동일하고 변동금리는 시장금리에 따라 변동됩니다.",
                    "metadata": {"source_url": "https://example.com/source"},
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_online_service_returns_solar_fast_path_and_evaluates_all_models(tmp_path: Path):
    candidate_client = ConcurrentCandidateClient()
    service = OnlineEvaluationService(
        candidate_client=candidate_client,
        judge_client=StaticJudgeClient(),
        store=FileOnlineEvaluationStore(tmp_path),
        candidate_models=[
            CandidateModel(provider="solar", model_id="solar-pro3"),
            CandidateModel(provider="gpt", model_id="gpt-4o"),
            CandidateModel(provider="claude-sonnet", model_id="claude-sonnet"),
            CandidateModel(provider="google", model_id="gemini-2.5-pro"),
        ],
        judge_models=[JudgeModel(provider="openai", model_id="gpt-judge")],
    )

    run = await service.start(_payload())

    assert run.workflow_output.final_answer_text == "solar 답변"
    assert run.workflow_output.results[0].provider == "solar"

    evaluation = await run.evaluation_task

    assert candidate_client.max_active == 4
    assert [panel["provider"] for panel in evaluation["model_panels"]] == [
        "solar",
        "gpt",
        "claude-sonnet",
        "google",
    ]
    assert evaluation["model_panels"][0]["is_customer_answer"] is True
    assert evaluation["model_panels"][0]["score"] == 8.0
    assert evaluation["model_panels"][0]["quality_badge"] == "좋음"
    assert (tmp_path / "online-session-1.json").exists()


def test_build_workflow_result_event_keeps_existing_websocket_shape():
    event = build_workflow_result_event(
        session_id="online-session-1",
        transcript_text="고정금리와 변동금리 차이를 알려주세요.",
        nlu_payload={"intent": "설명형"},
        workflow_output=None,
        action=None,
        evaluation_status="skipped",
    )

    assert event["type"] == "workflow_result"
    assert event["session_id"] == "online-session-1"
    assert event["is_final"] is True
    assert event["nlu_analysis"] == {"intent": "설명형"}
    assert event["workflow"] is None
    assert event["evaluation_status"] == "skipped"

"""Offline evaluation runner with injectable model, judge, and RAGAS clients."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from src.common.schemas import LLMRequest, LLMResponse
from src.evaluation.graph import build_llm_request
from src.evaluation.schemas import (
    CandidateModel,
    EvaluationRunResult,
    EvaluationScenario,
    JudgeEvaluation,
    JudgeModel,
    ModelEvaluationRecord,
    RagasEvaluation,
)
from src.evaluation.scoring import (
    aggregate_judge_evaluations,
    build_report_metrics,
    compute_primary_score,
    requires_review,
)


class CandidateLLMClient(Protocol):
    """Client protocol for generating AICC candidate answers."""

    async def generate(
        self, model: CandidateModel, request: LLMRequest
    ) -> LLMResponse:
        """Generates one candidate answer."""


class JudgeClient(Protocol):
    """Client protocol for scoring one candidate answer."""

    async def evaluate(
        self,
        judge: JudgeModel,
        scenario: EvaluationScenario,
        answer: LLMResponse,
    ) -> JudgeEvaluation:
        """Scores one candidate answer with one judge model."""


class RagasClient(Protocol):
    """Client protocol for RAGAS diagnostic scoring."""

    async def evaluate(
        self, scenario: EvaluationScenario, answer: LLMResponse
    ) -> RagasEvaluation:
        """Scores one candidate answer with RAGAS metrics."""


class EvaluationRunner:
    """Runs scenario x candidate model x repeat evaluation jobs."""

    def __init__(
        self,
        candidate_client: CandidateLLMClient,
        judge_client: JudgeClient,
        ragas_client: RagasClient,
    ) -> None:
        self.candidate_client = candidate_client
        self.judge_client = judge_client
        self.ragas_client = ragas_client

    async def run(
        self,
        scenarios: list[EvaluationScenario],
        candidate_models: list[CandidateModel],
        judge_models: list[JudgeModel],
        repeat_count: int = 1,
    ) -> EvaluationRunResult:
        """Runs the full offline evaluation loop without owning API details."""
        records: list[ModelEvaluationRecord] = []
        for scenario in scenarios:
            request = build_llm_request(scenario)
            for candidate_model in candidate_models:
                for repeat_index in range(repeat_count):
                    answer = await self.candidate_client.generate(
                        candidate_model, request
                    )
                    judge_evaluations = [
                        await self.judge_client.evaluate(judge, scenario, answer)
                        for judge in judge_models
                    ]
                    aggregated = aggregate_judge_evaluations(judge_evaluations)
                    ragas = await self.ragas_client.evaluate(scenario, answer)
                    records.append(
                        ModelEvaluationRecord(
                            scenario_id=scenario.scenario_id,
                            repeat_index=repeat_index,
                            candidate_provider=candidate_model.provider,
                            candidate_model=candidate_model.model_id,
                            answer_text=answer.text,
                            llm_request=request,
                            judge_evaluations=judge_evaluations,
                            aggregated_judge=aggregated,
                            ragas=ragas,
                            report_metrics=build_report_metrics(aggregated, ragas),
                            primary_score=compute_primary_score(aggregated),
                            review_required=requires_review(aggregated),
                            latency_ms=answer.latency_ms,
                            token_usage=answer.token_usage,
                            error=answer.error,
                        )
                    )
        return EvaluationRunResult(records=records)


def save_run_result(result: EvaluationRunResult, path: str | Path) -> None:
    """Saves a run result as UTF-8 JSON for later reporting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

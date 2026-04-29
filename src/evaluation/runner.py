"""Offline evaluation runner with injectable model, judge, and RAGAS clients."""

from __future__ import annotations

import json
import asyncio
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from src.common.schemas import LLMRequest, LLMResponse
from src.evaluation.graph import build_llm_request
from src.evaluation.rubrics import LEGACY_JUDGE_METRIC_NAMES
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
        judge_metric_names: tuple[str, ...] = LEGACY_JUDGE_METRIC_NAMES,
        judge_score_min: int = 1,
        judge_score_max: int = 5,
        include_ragas: bool = True,
        report_normalized_scores: bool = True,
        review_risk_metric_names: tuple[str, ...] = (
            "grounded_response",
            "safety_conservativeness",
            "handoff_judgment",
        ),
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.candidate_client = candidate_client
        self.judge_client = judge_client
        self.ragas_client = ragas_client
        self.judge_metric_names = judge_metric_names
        self.judge_score_min = judge_score_min
        self.judge_score_max = judge_score_max
        self.include_ragas = include_ragas
        self.report_normalized_scores = report_normalized_scores
        self.review_risk_metric_names = review_risk_metric_names
        self.timer = timer

    async def run(
        self,
        scenarios: list[EvaluationScenario],
        candidate_models: list[CandidateModel],
        judge_models: list[JudgeModel],
        repeat_count: int = 1,
    ) -> EvaluationRunResult:
        """Runs the full offline evaluation loop without owning API details."""
        jobs = []
        for scenario in scenarios:
            request = build_llm_request(scenario)
            for candidate_model in candidate_models:
                for repeat_index in range(repeat_count):
                    jobs.append(
                        self._run_record(
                            scenario=scenario,
                            request=request,
                            candidate_model=candidate_model,
                            judge_models=judge_models,
                            repeat_index=repeat_index,
                        )
                    )
        records = await asyncio.gather(*jobs)
        return EvaluationRunResult(records=list(records))

    async def _run_record(
        self,
        *,
        scenario: EvaluationScenario,
        request: LLMRequest,
        candidate_model: CandidateModel,
        judge_models: list[JudgeModel],
        repeat_index: int,
    ) -> ModelEvaluationRecord:
        total_started = self.timer()
        candidate_started = self.timer()
        answer = await self.candidate_client.generate(candidate_model, request)
        candidate_ms = _elapsed_ms(self.timer, candidate_started)
        evaluation_started = self.timer()
        judge_evaluations_result, ragas = await asyncio.gather(
            _timed_task(
                self.timer,
                self._evaluate_judges(judge_models, scenario, answer),
            ),
            _timed_task(
                self.timer,
                self.ragas_client.evaluate(scenario, answer),
            ),
        )
        judge_evaluations, judge_ms = judge_evaluations_result
        ragas, ragas_ms = ragas
        evaluation_parallel_ms = _elapsed_ms(self.timer, evaluation_started)
        total_ms = _elapsed_ms(self.timer, total_started)
        judge_evaluations = list(judge_evaluations)
        aggregated = aggregate_judge_evaluations(
            judge_evaluations,
            metric_names=self.judge_metric_names,
            score_min=self.judge_score_min,
            score_max=self.judge_score_max,
        )
        return ModelEvaluationRecord(
            scenario_id=scenario.scenario_id,
            repeat_index=repeat_index,
            candidate_provider=candidate_model.provider,
            candidate_model=candidate_model.model_id,
            answer_text=answer.text,
            llm_request=request,
            judge_evaluations=judge_evaluations,
            aggregated_judge=aggregated,
            ragas=ragas,
            report_metrics=build_report_metrics(
                aggregated,
                ragas,
                metric_names=self.judge_metric_names,
                include_ragas=self.include_ragas,
                use_normalized=self.report_normalized_scores,
            ),
            primary_score=compute_primary_score(
                aggregated,
                metric_names=self.judge_metric_names,
                use_normalized=self.report_normalized_scores,
            ),
            review_required=requires_review(
                aggregated,
                risk_metric_names=self.review_risk_metric_names,
            ),
            latency_ms=answer.latency_ms,
            token_usage=answer.token_usage,
            finish_reason=answer.finish_reason,
            timing_ms={
                "total": total_ms,
                "candidate": candidate_ms,
                "judge": judge_ms,
                "ragas": ragas_ms,
                "evaluation_parallel": evaluation_parallel_ms,
            },
            error=answer.error,
        )

    async def _evaluate_judges(
        self,
        judge_models: list[JudgeModel],
        scenario: EvaluationScenario,
        answer: LLMResponse,
    ) -> list[JudgeEvaluation]:
        return list(
            await asyncio.gather(
                *[
                    self.judge_client.evaluate(judge, scenario, answer)
                    for judge in judge_models
                ]
            )
        )


def save_run_result(result: EvaluationRunResult, path: str | Path) -> None:
    """Saves a run result as UTF-8 JSON for later reporting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def _timed_task(
    timer: Callable[[], float],
    task: Awaitable[Any],
) -> tuple[Any, int]:
    started = timer()
    result = await task
    return result, _elapsed_ms(timer, started)


def _elapsed_ms(timer: Callable[[], float], started: float) -> int:
    return int((timer() - started) * 1000)

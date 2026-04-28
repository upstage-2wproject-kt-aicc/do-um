"""Pydantic contracts for offline model evaluation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.common.schemas import LLMRequest


JUDGE_METRIC_NAMES: tuple[str, ...] = (
    "answer_accuracy",
    "grounded_response",
    "safety_conservativeness",
    "handoff_judgment",
    "user_guidance_quality",
)


class EvaluationScenario(BaseModel):
    """One fixed benchmark scenario for model comparison."""

    scenario_id: str = Field(..., description="Stable scenario identifier.")
    user_query: str = Field(..., description="Customer question for the scenario.")
    intent: str = Field(..., description="Fixed NLU intent label.")
    domain: str = Field("", description="Fixed NLU domain label.")
    subdomain: str = Field("", description="Fixed NLU subdomain label.")
    router_confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="Fixed routing confidence."
    )
    retrieved_context: str = Field(
        "", description="Fixed retrieved context shared across candidate models."
    )
    reference_answer: str = Field("", description="Reference answer or expected behavior.")
    policy_rules: list[dict[str, Any]] = Field(
        default_factory=list, description="Fixed policy rules for this scenario."
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata.")


class CandidateModel(BaseModel):
    """A service candidate model that generates AICC answers."""

    provider: str = Field(..., description="Provider identifier.")
    model_id: str = Field(..., description="Concrete model identifier.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata.")


class JudgeModel(BaseModel):
    """A stronger judge model used to score candidate answers."""

    provider: str = Field(..., description="Judge provider identifier.")
    model_id: str = Field(..., description="Concrete judge model identifier.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata.")


class JudgeMetricScore(BaseModel):
    """One judge metric score before aggregation."""

    score: int = Field(..., ge=1, le=5, description="Raw judge score.")
    reason: str = Field("", description="Concise Korean judge reason.")


class JudgeEvaluation(BaseModel):
    """All judge metric scores from one judge model."""

    judge_model: str = Field(..., description="Judge model identifier.")
    metrics: dict[str, JudgeMetricScore] = Field(
        ..., description="Metric scores keyed by judge metric name."
    )
    summary: dict[str, Any] = Field(default_factory=dict, description="Judge summary.")


class AggregatedMetricScore(BaseModel):
    """Aggregated score for one judge metric across judge models."""

    raw_median: float = Field(..., ge=1.0, le=5.0, description="Median raw score.")
    normalized: float = Field(..., ge=0.0, le=1.0, description="0-1 normalized score.")
    disagreement: int = Field(..., ge=0, description="Max raw score minus min raw score.")
    judge_scores: dict[str, int] = Field(
        default_factory=dict, description="Raw scores by judge model."
    )
    judge_reasons: dict[str, str] = Field(
        default_factory=dict, description="Reasons by judge model."
    )


class AggregatedJudgeEvaluation(BaseModel):
    """Aggregated judge result across all judge models."""

    metrics: dict[str, AggregatedMetricScore] = Field(
        ..., description="Aggregated metric scores."
    )


class RagasEvaluation(BaseModel):
    """RAGAS diagnostic metrics."""

    faithfulness: float | None = Field(None, ge=0.0, le=1.0)
    answer_relevancy: float | None = Field(None, ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)


class ModelEvaluationRecord(BaseModel):
    """One candidate answer and all evaluation results for one scenario run."""

    scenario_id: str
    repeat_index: int
    candidate_provider: str
    candidate_model: str
    answer_text: str
    llm_request: LLMRequest
    judge_evaluations: list[JudgeEvaluation]
    aggregated_judge: AggregatedJudgeEvaluation
    ragas: RagasEvaluation
    report_metrics: dict[str, float | None] = Field(default_factory=dict)
    primary_score: float = Field(..., ge=0.0, le=1.0)
    review_required: bool = Field(False)
    latency_ms: int = Field(0, ge=0)
    token_usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: str | None = None
    error: str | None = None


class EvaluationRunResult(BaseModel):
    """All records produced by one evaluation run."""

    records: list[ModelEvaluationRecord] = Field(default_factory=list)

"""Scoring helpers for model evaluation."""

from __future__ import annotations

from statistics import median

from src.evaluation.schemas import (
    AggregatedJudgeEvaluation,
    AggregatedMetricScore,
    JUDGE_METRIC_NAMES,
    JudgeEvaluation,
    RagasEvaluation,
)


def normalize_judge_score(score: float) -> float:
    """Normalizes a 1-5 judge score into a 0-1 score."""
    return max(0.0, min(1.0, (score - 1.0) / 4.0))


def aggregate_judge_evaluations(
    evaluations: list[JudgeEvaluation],
) -> AggregatedJudgeEvaluation:
    """Aggregates multiple judge outputs using median raw scores."""
    metrics: dict[str, AggregatedMetricScore] = {}
    for metric_name in JUDGE_METRIC_NAMES:
        score_by_judge: dict[str, int] = {}
        reason_by_judge: dict[str, str] = {}
        for evaluation in evaluations:
            metric = evaluation.metrics.get(metric_name)
            if metric is None:
                continue
            score_by_judge[evaluation.judge_model] = metric.score
            reason_by_judge[evaluation.judge_model] = metric.reason
        if not score_by_judge:
            raise ValueError(f"Missing judge metric: {metric_name}")
        raw_scores = list(score_by_judge.values())
        raw_median = float(median(raw_scores))
        metrics[metric_name] = AggregatedMetricScore(
            raw_median=raw_median,
            normalized=normalize_judge_score(raw_median),
            disagreement=max(raw_scores) - min(raw_scores),
            judge_scores=score_by_judge,
            judge_reasons=reason_by_judge,
        )
    return AggregatedJudgeEvaluation(metrics=metrics)


def compute_primary_score(
    aggregated: AggregatedJudgeEvaluation,
    weights: dict[str, float] | None = None,
) -> float:
    """Computes an optional summary score from normalized judge metrics.

    By default this is an unweighted average. Custom weights are supported only
    for explicit downstream reports that need a stakeholder-specific view.
    """
    if weights is None:
        values = [
            aggregated.metrics[metric_name].normalized
            for metric_name in JUDGE_METRIC_NAMES
        ]
        if not values:
            raise ValueError("At least one judge metric is required.")
        return sum(values) / len(values)

    weighted_sum = 0.0
    total_weight = 0.0
    for metric_name, weight in weights.items():
        metric = aggregated.metrics[metric_name]
        weighted_sum += metric.normalized * weight
        total_weight += weight
    if total_weight == 0:
        raise ValueError("Total judge weight must be greater than zero.")
    return weighted_sum / total_weight


def build_report_metrics(
    aggregated: AggregatedJudgeEvaluation,
    ragas: RagasEvaluation,
) -> dict[str, float | None]:
    """Builds the flat 7-metric view used by reports and charts."""
    report = {
        metric_name: aggregated.metrics[metric_name].normalized
        for metric_name in JUDGE_METRIC_NAMES
    }
    report["faithfulness"] = ragas.faithfulness
    report["answer_relevancy"] = ragas.answer_relevancy
    return report


def requires_review(
    aggregated: AggregatedJudgeEvaluation,
    disagreement_threshold: int = 2,
    low_score_threshold: float = 0.25,
) -> bool:
    """Flags records that need human review due to risk or judge disagreement."""
    for metric in aggregated.metrics.values():
        if metric.disagreement >= disagreement_threshold:
            return True
    for metric_name in (
        "grounded_response",
        "safety_conservativeness",
        "handoff_judgment",
    ):
        if aggregated.metrics[metric_name].normalized <= low_score_threshold:
            return True
    return False

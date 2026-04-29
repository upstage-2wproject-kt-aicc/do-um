"""Scoring helpers for model evaluation."""

from __future__ import annotations

from statistics import median

from src.evaluation.rubrics import LEGACY_JUDGE_METRIC_NAMES
from src.evaluation.schemas import (
    AggregatedJudgeEvaluation,
    AggregatedMetricScore,
    JudgeEvaluation,
    RagasEvaluation,
)


def normalize_judge_score(score: float, score_min: int = 1, score_max: int = 5) -> float:
    """Normalizes a judge score into a 0-1 score."""
    if score_max <= score_min:
        raise ValueError("score_max must be greater than score_min.")
    return max(0.0, min(1.0, (score - score_min) / (score_max - score_min)))


def aggregate_judge_evaluations(
    evaluations: list[JudgeEvaluation],
    metric_names: tuple[str, ...] = LEGACY_JUDGE_METRIC_NAMES,
    score_min: int = 1,
    score_max: int = 5,
) -> AggregatedJudgeEvaluation:
    """Aggregates multiple judge outputs using median raw scores."""
    metrics: dict[str, AggregatedMetricScore] = {}
    for metric_name in metric_names:
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
            normalized=normalize_judge_score(raw_median, score_min, score_max),
            disagreement=max(raw_scores) - min(raw_scores),
            judge_scores=score_by_judge,
            judge_reasons=reason_by_judge,
        )
    return AggregatedJudgeEvaluation(metrics=metrics)


def compute_primary_score(
    aggregated: AggregatedJudgeEvaluation,
    metric_names: tuple[str, ...] = LEGACY_JUDGE_METRIC_NAMES,
    use_normalized: bool = True,
    weights: dict[str, float] | None = None,
) -> float:
    """Computes an optional summary score from judge metrics.

    By default this is an unweighted average. Custom weights are supported only
    for explicit downstream reports that need a stakeholder-specific view.
    """
    if weights is None:
        values = [
            _metric_value(aggregated.metrics[metric_name], use_normalized)
            for metric_name in metric_names
        ]
        if not values:
            raise ValueError("At least one judge metric is required.")
        return sum(values) / len(values)

    weighted_sum = 0.0
    total_weight = 0.0
    for metric_name, weight in weights.items():
        metric = aggregated.metrics[metric_name]
        weighted_sum += _metric_value(metric, use_normalized) * weight
        total_weight += weight
    if total_weight == 0:
        raise ValueError("Total judge weight must be greater than zero.")
    return weighted_sum / total_weight


def build_report_metrics(
    aggregated: AggregatedJudgeEvaluation,
    ragas: RagasEvaluation,
    metric_names: tuple[str, ...] = LEGACY_JUDGE_METRIC_NAMES,
    include_ragas: bool = True,
    use_normalized: bool = True,
) -> dict[str, float | None]:
    """Builds the flat metric view used by reports and charts."""
    report = {
        metric_name: _metric_value(aggregated.metrics[metric_name], use_normalized)
        for metric_name in metric_names
    }
    if not include_ragas:
        return report
    report["faithfulness"] = ragas.faithfulness
    report["answer_relevancy"] = ragas.answer_relevancy
    return report


def requires_review(
    aggregated: AggregatedJudgeEvaluation,
    disagreement_threshold: int = 2,
    low_score_threshold: float = 0.25,
    risk_metric_names: tuple[str, ...] = (
        "grounded_response",
        "safety_conservativeness",
        "handoff_judgment",
    ),
) -> bool:
    """Flags records that need human review due to risk or judge disagreement."""
    for metric in aggregated.metrics.values():
        if metric.disagreement >= disagreement_threshold:
            return True
    for metric_name in risk_metric_names:
        metric = aggregated.metrics.get(metric_name)
        if metric and metric.normalized <= low_score_threshold:
            return True
    return False


def _metric_value(metric: AggregatedMetricScore, use_normalized: bool) -> float:
    return metric.normalized if use_normalized else metric.raw_median

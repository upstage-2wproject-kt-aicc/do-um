import pytest

from src.evaluation.schemas import (
    AggregatedJudgeEvaluation,
    AggregatedMetricScore,
    RagasEvaluation,
)
from src.evaluation.scoring import build_report_metrics, compute_primary_score


def test_primary_score_is_unweighted_average_by_default() -> None:
    aggregated = AggregatedJudgeEvaluation(
        metrics={
            "answer_accuracy": AggregatedMetricScore(
                raw_median=5, normalized=1.0, disagreement=0
            ),
            "grounded_response": AggregatedMetricScore(
                raw_median=4, normalized=0.75, disagreement=0
            ),
            "safety_conservativeness": AggregatedMetricScore(
                raw_median=3, normalized=0.5, disagreement=0
            ),
            "handoff_judgment": AggregatedMetricScore(
                raw_median=2, normalized=0.25, disagreement=0
            ),
            "user_guidance_quality": AggregatedMetricScore(
                raw_median=1, normalized=0.0, disagreement=0
            ),
        }
    )

    assert compute_primary_score(aggregated) == pytest.approx(0.5)


def test_build_report_metrics_flattens_judge_median_and_ragas_scores() -> None:
    aggregated = AggregatedJudgeEvaluation(
        metrics={
            "answer_accuracy": AggregatedMetricScore(
                raw_median=5, normalized=1.0, disagreement=0
            ),
            "grounded_response": AggregatedMetricScore(
                raw_median=4, normalized=0.75, disagreement=0
            ),
            "safety_conservativeness": AggregatedMetricScore(
                raw_median=3, normalized=0.5, disagreement=0
            ),
            "handoff_judgment": AggregatedMetricScore(
                raw_median=2, normalized=0.25, disagreement=0
            ),
            "user_guidance_quality": AggregatedMetricScore(
                raw_median=1, normalized=0.0, disagreement=0
            ),
        }
    )

    report_metrics = build_report_metrics(
        aggregated,
        RagasEvaluation(faithfulness=0.88, answer_relevancy=0.77),
    )

    assert report_metrics == {
        "answer_accuracy": 1.0,
        "grounded_response": 0.75,
        "safety_conservativeness": 0.5,
        "handoff_judgment": 0.25,
        "user_guidance_quality": 0.0,
        "faithfulness": 0.88,
        "answer_relevancy": 0.77,
    }

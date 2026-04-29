import pytest

from src.evaluation.schemas import (
    AggregatedJudgeEvaluation,
    AggregatedMetricScore,
    JudgeEvaluation,
    JudgeMetricScore,
    RagasEvaluation,
)
from src.evaluation.scoring import (
    aggregate_judge_evaluations,
    build_report_metrics,
    compute_primary_score,
)


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


def test_comparative_10_metrics_normalize_without_ragas() -> None:
    metric_names = (
        "intent_fit",
        "accuracy",
        "groundedness",
        "safety_conservatism",
        "handoff_appropriateness",
        "guidance_quality",
    )
    aggregated = aggregate_judge_evaluations(
        [
            JudgeEvaluation(
                judge_model="judge-a",
                metrics={
                    name: JudgeMetricScore(score=10, reason="우수")
                    for name in metric_names
                },
            ),
            JudgeEvaluation(
                judge_model="judge-b",
                metrics={
                    name: JudgeMetricScore(score=8, reason="양호")
                    for name in metric_names
                },
            ),
        ],
        metric_names=metric_names,
        score_min=1,
        score_max=10,
    )

    report_metrics = build_report_metrics(
        aggregated,
        RagasEvaluation(),
        metric_names=metric_names,
        include_ragas=False,
    )

    assert aggregated.metrics["intent_fit"].raw_median == 9
    assert aggregated.metrics["intent_fit"].normalized == pytest.approx(8 / 9)
    assert compute_primary_score(aggregated, metric_names=metric_names) == pytest.approx(8 / 9)
    assert set(report_metrics) == set(metric_names)


def test_comparative_10_report_can_use_raw_scores_without_ragas() -> None:
    metric_names = (
        "intent_fit",
        "accuracy",
        "groundedness",
        "safety_conservatism",
        "handoff_appropriateness",
        "guidance_quality",
    )
    aggregated = aggregate_judge_evaluations(
        [
            JudgeEvaluation(
                judge_model="judge-a",
                metrics={
                    name: JudgeMetricScore(score=10, reason="우수")
                    for name in metric_names
                },
            ),
            JudgeEvaluation(
                judge_model="judge-b",
                metrics={
                    name: JudgeMetricScore(score=8, reason="양호")
                    for name in metric_names
                },
            ),
        ],
        metric_names=metric_names,
        score_min=1,
        score_max=10,
    )

    report_metrics = build_report_metrics(
        aggregated,
        RagasEvaluation(),
        metric_names=metric_names,
        include_ragas=False,
        use_normalized=False,
    )

    assert report_metrics["intent_fit"] == 9
    assert compute_primary_score(
        aggregated,
        metric_names=metric_names,
        use_normalized=False,
    ) == pytest.approx(9)

"""Versioned judge rubric profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROMPTS_DIR = Path(__file__).parent / "prompts"

LEGACY_JUDGE_METRIC_NAMES: tuple[str, ...] = (
    "answer_accuracy",
    "grounded_response",
    "safety_conservativeness",
    "handoff_judgment",
    "user_guidance_quality",
)

COMPARATIVE_JUDGE_METRIC_NAMES: tuple[str, ...] = (
    "intent_fit",
    "accuracy",
    "groundedness",
    "safety_conservatism",
    "handoff_appropriateness",
    "guidance_quality",
)


@dataclass(frozen=True)
class JudgeRubric:
    """Configuration for one judge prompt and scoring shape."""

    name: str
    prompt_path: Path
    metric_names: tuple[str, ...]
    score_min: int
    score_max: int
    include_ragas: bool
    report_normalized_scores: bool
    score_scale_label: str
    primary_score_source: str
    review_risk_metric_names: tuple[str, ...]


JUDGE_RUBRICS: dict[str, JudgeRubric] = {
    "legacy_5": JudgeRubric(
        name="legacy_5",
        prompt_path=PROMPTS_DIR / "judge_v2.md",
        metric_names=LEGACY_JUDGE_METRIC_NAMES,
        score_min=1,
        score_max=5,
        include_ragas=True,
        report_normalized_scores=True,
        score_scale_label="1_to_5",
        primary_score_source="llm_as_a_judge_5_metrics",
        review_risk_metric_names=(
            "grounded_response",
            "safety_conservativeness",
            "handoff_judgment",
        ),
    ),
    "comparative_10": JudgeRubric(
        name="comparative_10",
        prompt_path=PROMPTS_DIR / "judge_v4_comparative.md",
        metric_names=COMPARATIVE_JUDGE_METRIC_NAMES,
        score_min=1,
        score_max=10,
        include_ragas=False,
        report_normalized_scores=False,
        score_scale_label="1_to_10",
        primary_score_source="llm_as_a_judge_6_comparative_metrics",
        review_risk_metric_names=(
            "groundedness",
            "safety_conservatism",
            "handoff_appropriateness",
        ),
    ),
}


def get_judge_rubric(name: str | None) -> JudgeRubric:
    """Returns a rubric profile by name."""
    rubric_name = name or "legacy_5"
    try:
        return JUDGE_RUBRICS[rubric_name]
    except KeyError as exc:
        valid = ", ".join(sorted(JUDGE_RUBRICS))
        raise ValueError(f"Unknown judge rubric: {rubric_name}. Valid values: {valid}") from exc

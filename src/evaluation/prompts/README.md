# Evaluation Prompts

This directory stores versioned prompt templates used by the evaluation module.

## Files

- `judge_v1.md`: Korean financial 상담 LLM-as-a-Judge prompt for the 5 primary judge-scored benchmark items.
- `judge_v2.md`: v1 criteria plus explicit candidate workflow system/user prompt separation, route awareness, and judge bias guardrails.
- `judge_v3_light.md`: compact judge prompt for fast workflow-time checks. Kept for latency experiments.
- `judge_v4_comparative.md`: operational prompt for model comparison. It uses a 1-10 score scale, no RAGAS, model-tendency-oriented scoring, hard flags, and the current workflow JSON payload shape.

## Scoring

- `legacy_5` uses a 1-5 raw score and the v1/v2/v3 metric shape.
- `comparative_10` uses a 1-10 raw score and the v4 metric shape.
- RAGAS items are legacy-only diagnostics and are not used in `comparative_10`.
- Judge raw scores are normalized with `(score - score_min) / (score_max - score_min)`.
- `legacy_5` report metrics are stored as normalized 0.0-1.0 values.
- `comparative_10` report metrics and primary score are stored as raw 1-10 values.

## Operating Recommendation

Use `comparative_10` for current AICC model selection.
It stores the six-dimensional judge vector directly:

- `intent_fit`
- `accuracy`
- `groundedness`
- `safety_conservatism`
- `handoff_appropriateness`
- `guidance_quality`

This covers the old RAGAS support roles through domain-aware judge instructions while avoiding unstable interpretation in no-context and handoff scenarios.

# Evaluation Prompts

This directory stores versioned prompt templates used by the evaluation module.

## Files

- `judge_v1.md`: Korean financial 상담 LLM-as-a-Judge prompt for the 7 judge-scored benchmark items.

## Scoring

- LLM-as-a-Judge items use a 1-5 raw score.
- RAGAS items use their native 0.0-1.0 score.
- Judge raw scores are normalized with `(score - 1) / 4`.

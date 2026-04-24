"""Evaluation interface skeleton for response quality scoring."""

from common.schemas import EvalInput, EvalResult


class EvaluationService:
    """Defines evaluation methods for RAGAS and judge scoring."""

    async def evaluate_ragas(self, payload: EvalInput) -> EvalResult:
        """Evaluates response quality with RAGAS metrics."""
        raise NotImplementedError

    async def evaluate_judge(self, payload: EvalInput) -> EvalResult:
        """Evaluates response quality with LLM-as-a-Judge."""
        raise NotImplementedError


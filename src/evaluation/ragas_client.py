"""RAGAS diagnostic scoring client."""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from src.common.schemas import LLMResponse
from src.evaluation.env import load_evaluation_env
from src.evaluation.schemas import EvaluationScenario, RagasEvaluation


class DatasetFactory(Protocol):
    """Minimal protocol for Hugging Face Dataset factory."""

    @classmethod
    def from_dict(cls, data: dict[str, list[Any]]) -> Any:
        """Builds a dataset from column lists."""


@dataclass(frozen=True)
class RagasDependencies:
    """Runtime RAGAS dependencies imported lazily."""

    evaluate_fn: Callable[..., Any]
    dataset_factory: DatasetFactory
    metrics: list[Any]
    evaluate_kwargs: dict[str, Any] = field(default_factory=dict)


def build_ragas_row(
    scenario: EvaluationScenario, answer: LLMResponse
) -> dict[str, Any]:
    """Builds one RAGAS single-turn row from an evaluation record."""
    row: dict[str, Any] = {
        "user_input": scenario.user_query,
        "response": answer.text,
        "retrieved_contexts": [scenario.retrieved_context]
        if scenario.retrieved_context
        else [],
    }
    if scenario.reference_answer:
        row["reference"] = scenario.reference_answer
    return row


class RagasClient:
    """Runs RAGAS faithfulness and answer relevancy metrics."""

    def __init__(
        self,
        evaluate_fn: Callable[..., Any] | None = None,
        dataset_factory: DatasetFactory | None = None,
        metrics: list[Any] | None = None,
        dependency_loader: Callable[[], RagasDependencies] | None = None,
    ) -> None:
        self.evaluate_fn = evaluate_fn
        self.dataset_factory = dataset_factory
        self.metrics = metrics
        self.dependency_loader = dependency_loader or load_ragas_dependencies

    async def evaluate(
        self, scenario: EvaluationScenario, answer: LLMResponse
    ) -> RagasEvaluation:
        """Scores one answer with RAGAS, returning not_configured if unavailable."""
        try:
            deps = self._dependencies()
        except ImportError as exc:
            return RagasEvaluation(
                faithfulness=None,
                answer_relevancy=None,
                details={"status": "not_configured", "reason": str(exc)},
            )

        row = build_ragas_row(scenario, answer)
        dataset = deps.dataset_factory.from_dict(
            {column: [value] for column, value in row.items()}
        )
        has_context = bool(scenario.retrieved_context.strip())
        metrics = _metrics_for_context(deps.metrics, has_context)
        result = await asyncio.to_thread(
            deps.evaluate_fn,
            dataset=dataset,
            metrics=metrics,
            raise_exceptions=False,
            show_progress=False,
            **deps.evaluate_kwargs,
        )
        details = {"status": "ok"}
        if not has_context:
            details["faithfulness_status"] = "not_applicable_no_context"
        return RagasEvaluation(
            faithfulness=None
            if not has_context
            else _optional_float(_metric_value(result, "faithfulness")),
            answer_relevancy=_optional_float(_metric_value(result, "answer_relevancy")),
            details=details,
        )

    def _dependencies(self) -> RagasDependencies:
        if self.evaluate_fn and self.dataset_factory and self.metrics is not None:
            return RagasDependencies(
                evaluate_fn=self.evaluate_fn,
                dataset_factory=self.dataset_factory,
                metrics=self.metrics,
            )
        return self.dependency_loader()


def load_ragas_dependencies() -> RagasDependencies:
    """Imports RAGAS dependencies lazily so tests can run without provider calls."""
    configure_ragas_openai_env()
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    try:
        from ragas.metrics.collections import Faithfulness, ResponseRelevancy
    except ImportError:  # pragma: no cover - older ragas fallback
        from ragas.metrics import Faithfulness, ResponseRelevancy

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=_first_env(
                (
                    "RAGAS_OPENAI_MODEL",
                    "LLM_GPT_MODEL",
                )
            )
            or "gpt-4o-mini",
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=_first_env(("RAGAS_OPENAI_EMBEDDING_MODEL",))
            or "text-embedding-3-small"
        )
    )
    return RagasDependencies(
        evaluate_fn=evaluate,
        dataset_factory=Dataset,
        metrics=[
            Faithfulness(llm=llm),
            ResponseRelevancy(llm=llm, embeddings=embeddings),
        ],
    )


def configure_ragas_openai_env() -> None:
    """Fills OpenAI env names expected by RAGAS from project-specific aliases."""
    load_evaluation_env()
    if not os.getenv("OPENAI_API_KEY"):
        api_key = _first_env(
            ("JUDGE_OPENAI_API_KEY", "LLM_GPT_API_KEY", "OPENAI_API_KEY")
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    if not os.getenv("OPENAI_BASE_URL"):
        base_url = _first_env(
            ("JUDGE_OPENAI_BASE_URL", "LLM_GPT_BASE_URL", "OPENAI_BASE_URL")
        )
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url


def _first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _metric_value(result: Any, key: str) -> Any:
    if isinstance(result, dict):
        return result.get(key)
    try:
        return result[key]
    except (KeyError, TypeError):
        pass
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        if key in frame:
            return frame[key].mean()
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        scores = [
            score
            for item in value
            if (score := _optional_float(item)) is not None
        ]
        if not scores:
            return None
        return sum(scores) / len(scores)
    score = float(value)
    if math.isnan(score):
        return None
    return score


def _metrics_for_context(metrics: list[Any], has_context: bool) -> list[Any]:
    if has_context:
        return metrics
    return [metric for metric in metrics if not _is_faithfulness_metric(metric)]


def _is_faithfulness_metric(metric: Any) -> bool:
    name = metric if isinstance(metric, str) else getattr(metric, "name", "")
    return "faithfulness" in str(name)

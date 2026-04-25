import pytest

from src.common.schemas import LLMResponse
from src.evaluation.ragas_client import (
    RagasClient,
    build_ragas_row,
    configure_ragas_openai_env,
)
from src.evaluation.schemas import EvaluationScenario, RagasEvaluation


class FakeDataset:
    def __init__(self, data: dict) -> None:
        self.data = data


class FakeDatasetFactory:
    calls: list[dict] = []

    @classmethod
    def from_dict(cls, data: dict) -> FakeDataset:
        cls.calls.append(data)
        return FakeDataset(data)


def test_build_ragas_row_maps_scenario_and_answer() -> None:
    row = build_ragas_row(
        EvaluationScenario(
            scenario_id="s1",
            user_query="고정금리와 변동금리 차이는?",
            intent="설명형",
            retrieved_context="고정금리는 약정 기간 동안 동일합니다.",
            reference_answer="고정금리와 변동금리의 차이를 설명합니다.",
        ),
        LLMResponse(
            session_id="s1",
            provider="gpt",
            text="고정금리는 동일하고 변동금리는 바뀝니다.",
            latency_ms=1,
        ),
    )

    assert row == {
        "user_input": "고정금리와 변동금리 차이는?",
        "response": "고정금리는 동일하고 변동금리는 바뀝니다.",
        "retrieved_contexts": ["고정금리는 약정 기간 동안 동일합니다."],
        "reference": "고정금리와 변동금리의 차이를 설명합니다.",
    }


def test_configure_ragas_openai_env_uses_existing_gpt_key_alias(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_GPT_API_KEY", "gpt-key")
    monkeypatch.setenv("LLM_GPT_BASE_URL", "https://example.test/v1")

    configure_ragas_openai_env()

    assert __import__("os").environ["OPENAI_API_KEY"] == "gpt-key"
    assert __import__("os").environ["OPENAI_BASE_URL"] == "https://example.test/v1"


@pytest.mark.asyncio
async def test_ragas_client_returns_metric_scores_from_injected_evaluator() -> None:
    def fake_evaluate(**kwargs):
        dataset = kwargs["dataset"]
        assert dataset.data["user_input"] == ["질문"]
        assert dataset.data["response"] == ["답변"]
        assert dataset.data["retrieved_contexts"] == [["문서"]]
        assert kwargs["metrics"] == ["faithfulness_metric", "answer_relevancy_metric"]
        assert kwargs["raise_exceptions"] is False
        assert kwargs["show_progress"] is False
        return {"faithfulness": 0.75, "answer_relevancy": 0.5}

    client = RagasClient(
        evaluate_fn=fake_evaluate,
        dataset_factory=FakeDatasetFactory,
        metrics=["faithfulness_metric", "answer_relevancy_metric"],
    )

    result = await client.evaluate(
        EvaluationScenario(
            scenario_id="s1",
            user_query="질문",
            intent="설명형",
            retrieved_context="문서",
        ),
        LLMResponse(session_id="s1", provider="gpt", text="답변", latency_ms=1),
    )

    assert result == RagasEvaluation(
        faithfulness=0.75,
        answer_relevancy=0.5,
        details={"status": "ok"},
    )


@pytest.mark.asyncio
async def test_ragas_client_records_not_configured_when_dependency_missing() -> None:
    def missing_import():
        raise ImportError("missing ragas")

    client = RagasClient(dependency_loader=missing_import)

    result = await client.evaluate(
        EvaluationScenario(
            scenario_id="s1",
            user_query="질문",
            intent="설명형",
            retrieved_context="문서",
        ),
        LLMResponse(session_id="s1", provider="gpt", text="답변", latency_ms=1),
    )

    assert result.faithfulness is None
    assert result.answer_relevancy is None
    assert result.details["status"] == "not_configured"

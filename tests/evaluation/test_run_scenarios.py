import asyncio
import json

from src.evaluation.run_scenarios import run_scenarios_one_by_one
from src.evaluation.schemas import CandidateModel, EvaluationRunResult, EvaluationScenario, JudgeModel


class RecordingRunner:
    def __init__(self) -> None:
        self.scenario_ids: list[str] = []

    async def run(
        self,
        scenarios: list[EvaluationScenario],
        candidate_models: list[CandidateModel],
        judge_models: list[JudgeModel],
        repeat_count: int = 1,
    ) -> EvaluationRunResult:
        self.scenario_ids.append(scenarios[0].scenario_id)
        return EvaluationRunResult(records=[])


def test_run_scenarios_one_by_one_saves_each_scenario_and_index(tmp_path) -> None:
    asyncio.run(_run_scenarios_one_by_one_assertions(tmp_path))


async def _run_scenarios_one_by_one_assertions(tmp_path) -> None:
    scenarios = [
        EvaluationScenario(
            scenario_id="scenario/one",
            user_query="질문 1",
            intent="설명형",
            retrieved_context="문서 1",
        ),
        EvaluationScenario(
            scenario_id="scenario two",
            user_query="질문 2",
            intent="절차형",
            retrieved_context="문서 2",
        ),
    ]
    runner = RecordingRunner()

    summaries = await run_scenarios_one_by_one(
        scenarios=scenarios,
        candidate_models=[CandidateModel(provider="gpt", model_id="gpt-4o")],
        judge_models=[JudgeModel(provider="openai", model_id="gpt-5.5")],
        runner=runner,
        output_dir=tmp_path,
        repeat_count=1,
    )

    assert runner.scenario_ids == ["scenario/one", "scenario two"]
    assert [summary["scenario_id"] for summary in summaries] == ["scenario/one", "scenario two"]
    assert (tmp_path / "scenario_one.json").exists()
    assert (tmp_path / "scenario_two.json").exists()

    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index["scenario_count"] == 2
    assert [item["output_path"] for item in index["scenarios"]] == [
        str(tmp_path / "scenario_one.json"),
        str(tmp_path / "scenario_two.json"),
    ]

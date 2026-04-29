import argparse
import asyncio
import json

from src.evaluation.run_scenarios import build_parser, run_scenarios_one_by_one
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


class ConcurrentRecordingRunner(RecordingRunner):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0

    async def run(
        self,
        scenarios: list[EvaluationScenario],
        candidate_models: list[CandidateModel],
        judge_models: list[JudgeModel],
        repeat_count: int = 1,
    ) -> EvaluationRunResult:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return await super().run(scenarios, candidate_models, judge_models, repeat_count)


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
    timer_values = iter([10.0, 12.5, 20.0, 21.0])

    summaries = await run_scenarios_one_by_one(
        scenarios=scenarios,
        candidate_models=[CandidateModel(provider="gpt", model_id="gpt-4o")],
        judge_models=[JudgeModel(provider="openai", model_id="gpt-5.5")],
        runner=runner,
        output_dir=tmp_path,
        repeat_count=1,
        timer=lambda: next(timer_values),
    )

    assert runner.scenario_ids == ["scenario/one", "scenario two"]
    assert [summary["scenario_id"] for summary in summaries] == ["scenario/one", "scenario two"]
    assert [summary["duration_ms"] for summary in summaries] == [2500, 1000]
    assert (tmp_path / "scenario_one.json").exists()
    assert (tmp_path / "scenario_two.json").exists()

    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index["scenario_count"] == 2
    assert [item["output_path"] for item in index["scenarios"]] == [
        str(tmp_path / "scenario_one.json"),
        str(tmp_path / "scenario_two.json"),
    ]
    assert [item["duration_ms"] for item in index["scenarios"]] == [2500, 1000]


def test_run_scenarios_one_by_one_keeps_scenarios_sequential(tmp_path) -> None:
    asyncio.run(_run_scenarios_one_by_one_sequential_assertions(tmp_path))


async def _run_scenarios_one_by_one_sequential_assertions(tmp_path) -> None:
    scenarios = [
        EvaluationScenario(
            scenario_id="scenario-one",
            user_query="질문 1",
            intent="설명형",
            retrieved_context="문서 1",
        ),
        EvaluationScenario(
            scenario_id="scenario-two",
            user_query="질문 2",
            intent="절차형",
            retrieved_context="문서 2",
        ),
    ]
    runner = ConcurrentRecordingRunner()

    await run_scenarios_one_by_one(
        scenarios=scenarios,
        candidate_models=[CandidateModel(provider="gpt", model_id="gpt-4o")],
        judge_models=[JudgeModel(provider="openai", model_id="gpt-5.5")],
        runner=runner,
        output_dir=tmp_path,
        repeat_count=1,
    )

    assert runner.max_active == 1


def test_run_scenarios_parser_supports_comparative_10_rubric() -> None:
    args = build_parser().parse_args(["--judge-rubric", "comparative_10"])

    assert isinstance(args, argparse.Namespace)
    assert args.judge_rubric == "comparative_10"
    assert args.disable_ragas is False

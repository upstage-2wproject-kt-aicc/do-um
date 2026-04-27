"""CLI for running a scenario file one scenario at a time."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from src.evaluation.clients import CompositeCandidateClient, CompositeJudgeClient
from src.evaluation.env import load_evaluation_env
from src.evaluation.run_evaluation import (
    build_ragas_client,
    default_candidate_models_from_env,
    default_judge_models_from_env,
    load_scenarios_by_extension,
    parse_candidate_model_specs,
    parse_judge_model_specs,
)
from src.evaluation.runner import EvaluationRunner, save_run_result
from src.evaluation.schemas import CandidateModel, EvaluationScenario, JudgeModel


async def run_scenarios_one_by_one(
    *,
    scenarios: list[EvaluationScenario],
    candidate_models: list[CandidateModel],
    judge_models: list[JudgeModel],
    runner: EvaluationRunner,
    output_dir: str | Path,
    repeat_count: int = 1,
) -> list[dict[str, Any]]:
    """Runs each scenario independently and writes one result JSON per scenario."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for scenario in scenarios:
        result = await runner.run(
            scenarios=[scenario],
            candidate_models=candidate_models,
            judge_models=judge_models,
            repeat_count=repeat_count,
        )
        scenario_output_path = output_path / f"{_safe_filename(scenario.scenario_id)}.json"
        save_run_result(result, scenario_output_path)
        summaries.append(
            {
                "scenario_id": scenario.scenario_id,
                "output_path": str(scenario_output_path),
                "record_count": len(result.records),
            }
        )

    _save_index(output_path / "index.json", summaries)
    return summaries


async def run_from_args(args: argparse.Namespace) -> None:
    """Runs one-by-one scenario evaluation from parsed CLI args."""
    load_evaluation_env()
    scenarios = load_scenarios_by_extension(args.scenarios)
    if args.scenario_id:
        wanted = set(args.scenario_id)
        scenarios = [scenario for scenario in scenarios if scenario.scenario_id in wanted]
    if not scenarios:
        raise ValueError("실행할 시나리오가 없습니다.")

    candidate_models = (
        parse_candidate_model_specs(args.candidate)
        if args.candidate
        else default_candidate_models_from_env()
    )
    judge_models = (
        parse_judge_model_specs(args.judge)
        if args.judge
        else default_judge_models_from_env()
    )
    if not candidate_models:
        raise ValueError("평가할 candidate 모델이 없습니다. --candidate 또는 LLM_*_MODEL을 설정하세요.")
    if not judge_models:
        raise ValueError("평가할 judge 모델이 없습니다. --judge 또는 JUDGE_*_MODEL을 설정하세요.")

    runner = EvaluationRunner(
        candidate_client=CompositeCandidateClient(),
        judge_client=CompositeJudgeClient(),
        ragas_client=build_ragas_client(args.disable_ragas),
    )
    summaries = await run_scenarios_one_by_one(
        scenarios=scenarios,
        candidate_models=candidate_models,
        judge_models=judge_models,
        runner=runner,
        output_dir=args.output_dir,
        repeat_count=args.repeat_count,
    )
    print(f"saved {len(summaries)} scenario result files to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Builds the one-by-one scenario evaluation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run AICC evaluation from a CSV/TSV scenario file one scenario at a time."
    )
    parser.add_argument(
        "--scenarios",
        default="evaluation/scenarios/rag_faq_v1.tsv",
        help="CSV/TSV scenario file path.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_runs/scenarios",
        help="Directory where per-scenario JSON results are written.",
    )
    parser.add_argument(
        "--scenario-id",
        action="append",
        default=[],
        help="Run only this scenario_id. Repeatable.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate model spec, e.g. solar:solar-pro3. Repeatable.",
    )
    parser.add_argument(
        "--judge",
        action="append",
        default=[],
        help="Judge model spec, e.g. openai:gpt-5.5. Repeatable.",
    )
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument(
        "--disable-ragas",
        action="store_true",
        help="Skip RAGAS metrics and record them as not_configured.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    asyncio.run(run_from_args(build_parser().parse_args()))


def _safe_filename(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", value.strip())
    normalized = normalized.strip("._")
    return normalized or "scenario"


def _save_index(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(
            {
                "scenario_count": len(summaries),
                "scenarios": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

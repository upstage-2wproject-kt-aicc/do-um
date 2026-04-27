"""CLI for running fixed-scenario model evaluation."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from src.evaluation.clients import (
    CompositeCandidateClient,
    CompositeJudgeClient,
    NullRagasClient,
)
from src.evaluation.env import load_evaluation_env
from src.evaluation.loader import load_scenarios_csv, load_scenarios_tsv
from src.evaluation.ragas_client import RagasClient
from src.evaluation.runner import EvaluationRunner, save_run_result
from src.evaluation.schemas import CandidateModel, EvaluationScenario, JudgeModel


EnvNames = str | tuple[str, ...]


_CANDIDATE_MODEL_ENVS: tuple[tuple[str, EnvNames], ...] = (
    ("solar", "LLM_SOLAR_MODEL"),
    ("gpt", "LLM_GPT_MODEL"),
    ("claude-sonnet", "LLM_CLAUDE_SONNET_MODEL"),
    ("google", "LLM_GOOGLE_VERTEX_MODEL"),
)

_JUDGE_MODEL_ENVS: tuple[tuple[str, EnvNames], ...] = (
    ("openai", ("JUDGE_OPENAI_MODEL", "EVALUATION_PROVIDER_OPENAI_MODEL", "EVALUATION_PROVIDER_GPT_MODEL")),
    ("anthropic", ("JUDGE_ANTHROPIC_MODEL", "EVALUATION_PROVIDER_CLAUDE_MODEL")),
    ("google", "JUDGE_GOOGLE_VERTEX_MODEL"),
)


def load_scenarios_by_extension(path: str | Path) -> list[EvaluationScenario]:
    """Loads scenarios from CSV or TSV based on file extension."""
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return load_scenarios_csv(input_path)
    if suffix in {".tsv", ".tab"}:
        return load_scenarios_tsv(input_path)
    raise ValueError("시나리오 파일은 csv 또는 tsv 형식이어야 합니다.")


def parse_candidate_model_specs(specs: list[str]) -> list[CandidateModel]:
    """Parses provider:model_id candidate specs."""
    return [
        CandidateModel(provider=provider, model_id=model_id)
        for provider, model_id in _parse_model_specs(specs)
    ]


def parse_judge_model_specs(specs: list[str]) -> list[JudgeModel]:
    """Parses provider:model_id judge specs."""
    return [
        JudgeModel(provider=provider, model_id=model_id)
        for provider, model_id in _parse_model_specs(specs)
    ]


def default_candidate_models_from_env() -> list[CandidateModel]:
    """Builds candidate model list from configured env vars."""
    return [
        CandidateModel(provider=provider, model_id=model_id)
        for provider, model_id in _models_from_env(_CANDIDATE_MODEL_ENVS)
    ]


def default_judge_models_from_env() -> list[JudgeModel]:
    """Builds judge model list from configured env vars."""
    return [
        JudgeModel(provider=provider, model_id=model_id)
        for provider, model_id in _models_from_env(_JUDGE_MODEL_ENVS)
    ]


async def run_from_args(args: argparse.Namespace) -> None:
    """Runs evaluation from parsed CLI args."""
    load_evaluation_env()
    scenarios = load_scenarios_by_extension(args.scenarios)
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
    result = await runner.run(
        scenarios=scenarios,
        candidate_models=candidate_models,
        judge_models=judge_models,
        repeat_count=args.repeat_count,
    )
    save_run_result(result, args.output)
    print(f"saved {len(result.records)} records to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    """Builds the evaluation CLI parser."""
    parser = argparse.ArgumentParser(description="Run fixed-scenario AICC model evaluation.")
    parser.add_argument("--scenarios", required=True, help="CSV/TSV scenario file path.")
    parser.add_argument(
        "--output",
        default="evaluation_runs/latest.json",
        help="Output JSON path.",
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
        help="Judge model spec, e.g. openai:gpt-4o. Repeatable.",
    )
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument(
        "--disable-ragas",
        action="store_true",
        help="Skip RAGAS metrics and record them as not_configured.",
    )
    return parser


def build_ragas_client(disable_ragas: bool):
    """Builds the configured RAGAS client for CLI runs."""
    if disable_ragas:
        return NullRagasClient()
    return RagasClient()


def main() -> None:
    """CLI entry point."""
    asyncio.run(run_from_args(build_parser().parse_args()))


def _parse_model_specs(specs: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for spec in specs:
        provider, separator, model_id = spec.partition(":")
        if not separator or not provider.strip() or not model_id.strip():
            raise ValueError("모델 스펙은 provider:model_id 형식이어야 합니다.")
        parsed.append((provider.strip(), model_id.strip()))
    return parsed


def _models_from_env(env_specs: tuple[tuple[str, EnvNames], ...]) -> list[tuple[str, str]]:
    models: list[tuple[str, str]] = []
    for provider, env_names in env_specs:
        model_id = _first_env(env_names)
        if model_id:
            models.append((provider, model_id))
    return models


def _first_env(names: EnvNames) -> str:
    env_names = (names,) if isinstance(names, str) else names
    for env_name in env_names:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""


if __name__ == "__main__":
    main()

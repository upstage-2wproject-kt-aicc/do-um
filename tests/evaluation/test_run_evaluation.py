from pathlib import Path

import pytest

from src.evaluation.run_evaluation import (
    build_ragas_client,
    default_candidate_models_from_env,
    default_judge_models_from_env,
    load_scenarios_by_extension,
    parse_candidate_model_specs,
    parse_judge_model_specs,
)


def test_parse_model_specs_use_provider_and_model_id() -> None:
    candidates = parse_candidate_model_specs(["solar:solar-pro", "gpt:gpt-4o"])
    judges = parse_judge_model_specs(["openai:gpt-judge", "google:gemini-judge"])

    assert [(model.provider, model.model_id) for model in candidates] == [
        ("solar", "solar-pro"),
        ("gpt", "gpt-4o"),
    ]
    assert [(model.provider, model.model_id) for model in judges] == [
        ("openai", "gpt-judge"),
        ("google", "gemini-judge"),
    ]


def test_parse_model_specs_reject_invalid_format() -> None:
    with pytest.raises(ValueError, match="provider:model_id"):
        parse_candidate_model_specs(["solar"])


def test_default_models_from_env_skip_missing(monkeypatch) -> None:
    monkeypatch.setenv("LLM_SOLAR_MODEL", "solar-pro")
    monkeypatch.delenv("LLM_GPT_MODEL", raising=False)
    monkeypatch.delenv("LLM_CLAUDE_SONNET_MODEL", raising=False)
    monkeypatch.delenv("LLM_GOOGLE_MODEL", raising=False)
    monkeypatch.delenv("LLM_GEMINI_MODEL", raising=False)
    monkeypatch.setenv("JUDGE_OPENAI_MODEL", "gpt-judge")
    monkeypatch.delenv("JUDGE_ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_GOOGLE_MODEL", raising=False)
    monkeypatch.delenv("EVALUATION_PROVIDER_CLAUDE_MODEL", raising=False)
    monkeypatch.delenv("EVALUATION_PROVIDER_GEMINI_MODEL", raising=False)

    assert [(model.provider, model.model_id) for model in default_candidate_models_from_env()] == [
        ("solar", "solar-pro")
    ]
    assert [(model.provider, model.model_id) for model in default_judge_models_from_env()] == [
        ("openai", "gpt-judge")
    ]


def test_default_models_from_env_use_aliases(monkeypatch) -> None:
    for name in [
        "LLM_SOLAR_MODEL",
        "LLM_GPT_MODEL",
        "LLM_CLAUDE_SONNET_MODEL",
        "LLM_GOOGLE_MODEL",
        "JUDGE_OPENAI_MODEL",
        "JUDGE_ANTHROPIC_MODEL",
        "JUDGE_GOOGLE_MODEL",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_GEMINI_MODEL", "gemini-candidate")
    monkeypatch.setenv("EVALUATION_PROVIDER_GPT_MODEL", "gpt-judge")
    monkeypatch.setenv("EVALUATION_PROVIDER_CLAUDE_MODEL", "claude-judge")
    monkeypatch.setenv("EVALUATION_PROVIDER_GEMINI_MODEL", "gemini-judge")

    assert [(model.provider, model.model_id) for model in default_candidate_models_from_env()] == [
        ("google", "gemini-candidate")
    ]
    assert [(model.provider, model.model_id) for model in default_judge_models_from_env()] == [
        ("openai", "gpt-judge"),
        ("anthropic", "claude-judge"),
        ("google", "gemini-judge"),
    ]


def test_load_scenarios_by_extension_supports_csv_and_tsv(tmp_path: Path) -> None:
    header = "scenario_id,question,intent,retrieved_context\n"
    row = "s1,질문,설명형,문서\n"
    csv_path = tmp_path / "scenarios.csv"
    csv_path.write_text(header + row, encoding="utf-8")

    scenarios = load_scenarios_by_extension(csv_path)

    assert len(scenarios) == 1
    assert scenarios[0].scenario_id == "s1"


def test_load_scenarios_by_extension_rejects_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "scenarios.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="csv 또는 tsv"):
        load_scenarios_by_extension(path)


def test_build_ragas_client_can_disable_ragas() -> None:
    assert build_ragas_client(disable_ragas=True).__class__.__name__ == "NullRagasClient"
    assert build_ragas_client(disable_ragas=False).__class__.__name__ == "RagasClient"

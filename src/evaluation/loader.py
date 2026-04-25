"""Scenario file loaders for evaluation runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.evaluation.schemas import EvaluationScenario


def load_scenarios_tsv(path: str | Path) -> list[EvaluationScenario]:
    """Loads workflow-ready evaluation scenarios from a TSV file."""
    return _load_scenarios(path, delimiter="\t")


def load_scenarios_csv(path: str | Path) -> list[EvaluationScenario]:
    """Loads workflow-ready evaluation scenarios from a CSV file."""
    return _load_scenarios(path, delimiter=",")


def _load_scenarios(path: str | Path, delimiter: str) -> list[EvaluationScenario]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        return [_row_to_scenario(row) for row in reader]


def _row_to_scenario(row: dict[str, str | None]) -> EvaluationScenario:
    metadata = _parse_metadata(row.get("context_metadata") or "")
    keywords = _split_keywords(row.get("keywords") or "")
    if keywords:
        metadata["keywords"] = keywords

    return EvaluationScenario(
        scenario_id=_required(row, "scenario_id"),
        user_query=_required(row, "question"),
        intent=_required(row, "intent"),
        domain=row.get("domain") or "",
        subdomain=row.get("subdomain") or "",
        retrieved_context=_required(row, "retrieved_context"),
        reference_answer=row.get("reference_answer") or "",
        metadata=metadata,
    )


def _required(row: dict[str, str | None], key: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required scenario column: {key}")
    return value


def _parse_metadata(raw: str) -> dict[str, Any]:
    stripped = raw.strip()
    if not stripped:
        return {}
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("context_metadata must be a JSON object.")
    return data


def _split_keywords(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]

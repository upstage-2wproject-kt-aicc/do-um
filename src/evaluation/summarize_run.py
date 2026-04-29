"""Builds a human-readable Markdown report from evaluation JSON outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from src.evaluation.pricing import estimate_token_cost_usd, normalize_token_usage


def summarize_run(run_dir: str | Path, output: str | Path | None = None) -> Path:
    """Writes a Markdown summary for one scenario-run directory."""
    run_path = Path(run_dir)
    index_path = run_path / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"index.json not found: {index_path}")

    index = _load_json(index_path)
    scenario_items = index.get("scenarios", [])
    if not isinstance(scenario_items, list):
        raise ValueError("index.json scenarios must be a list.")

    lines: list[str] = [
        "# Evaluation Summary",
        "",
        f"- run_dir: `{run_path}`",
        f"- scenario_count: {index.get('scenario_count', len(scenario_items))}",
        "",
    ]

    loaded_scenarios: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in scenario_items:
        if not isinstance(item, dict):
            continue
        result_path = Path(str(item.get("output_path", "")))
        if not result_path.is_absolute():
            result_path = Path.cwd() / result_path
        data = _load_json(result_path)
        loaded_scenarios.append((item, data))

    lines.extend(_overall_model_stats(loaded_scenarios))

    for item, data in loaded_scenarios:
        lines.extend(_scenario_section(item, data))

    output_path = Path(output) if output else run_path / "summary.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Builds the summary CLI parser."""
    parser = argparse.ArgumentParser(
        description="Create a Markdown summary from evaluation run JSON files."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Evaluation run directory containing index.json.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output Markdown path. Defaults to <run-dir>/summary.md.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    path = summarize_run(args.run_dir, args.output)
    print(f"saved summary to {path}")


def _overall_model_stats(
    loaded_scenarios: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[str]:
    records_by_model: dict[str, list[dict[str, Any]]] = {}
    for _, data in loaded_scenarios:
        records = data.get("records", [])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            model = str(record.get("candidate_model", ""))
            if model:
                records_by_model.setdefault(model, []).append(record)
    if not records_by_model:
        return []

    lines = [
        "## Model Cost / Latency Overview",
        "",
        "| Model | Avg Primary | Avg Total ms | Primary Std | Avg Cand In | Avg Cand Out | Avg Cand Cost | Cand / 1k | Avg Judge Cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in sorted(records_by_model):
        lines.append(_overall_model_row(model, records_by_model[model]))
    lines.extend(
        [
            "",
            "> Cost is a token-price estimate from the checked-in pricing snapshot. Existing runs without judge token usage show judge cost as N/A.",
            "",
        ]
    )
    return lines


def _overall_model_row(model: str, records: list[dict[str, Any]]) -> str:
    primary_scores = [
        float(record["primary_score"])
        for record in records
        if isinstance(record.get("primary_score"), (int, float))
    ]
    total_ms_values = _total_ms_values(records)
    candidate_usages = [
        normalize_token_usage(_dict_or_empty(record.get("token_usage")))
        for record in records
    ]
    candidate_costs = [
        cost.usd
        for record in records
        if (
            cost := estimate_token_cost_usd(
                model,
                _dict_or_empty(record.get("token_usage")),
            )
        )
        is not None
    ]
    judge_costs = [
        float(judge_cost["usd"])
        for record in records
        if (judge_cost := _total_judge_cost(record))["known"]
    ]
    return (
        f"| {model} "
        f"| {_fmt_score(_mean(primary_scores))} "
        f"| {_fmt_int(round(_mean(total_ms_values)))} "
        f"| {_fmt_score(_std(primary_scores))} "
        f"| {_fmt_int(round(_mean([usage.input_tokens for usage in candidate_usages])))} "
        f"| {_fmt_int(round(_mean([usage.output_tokens for usage in candidate_usages])))} "
        f"| {_fmt_usd(_mean(candidate_costs)) if candidate_costs else 'N/A'} "
        f"| {_fmt_usd4(_mean(candidate_costs) * 1000) if candidate_costs else 'N/A'} "
        f"| {_fmt_usd(_mean(judge_costs)) if judge_costs else 'N/A'} |"
    )


def _scenario_section(index_item: dict[str, Any], data: dict[str, Any]) -> list[str]:
    records = data.get("records", [])
    if not isinstance(records, list):
        records = []
    first = records[0] if records else {}
    prompt = first.get("llm_request", {}).get("prompt", "") if isinstance(first, dict) else ""
    question = _extract_block(prompt, "USER_QUERY")
    context = _extract_block(prompt, "INTERNAL_CONTEXT")
    metadata = _extract_block(prompt, "ROUTING_METADATA")

    lines = [
        f"## {index_item.get('scenario_id', 'unknown_scenario')}",
        "",
        f"- duration_ms: {index_item.get('duration_ms', 0)}",
        f"- record_count: {index_item.get('record_count', len(records))}",
        f"- question: {question or 'N/A'}",
        "",
        "<details>",
        "<summary>Context / Metadata</summary>",
        "",
        "```text",
        f"{context or 'N/A'}",
        "",
        "[ROUTING_METADATA]",
        f"{metadata or 'N/A'}",
        "```",
        "",
        "</details>",
        "",
        "| Model | Primary | Intent | Accuracy | Grounded | Safety | Handoff | Guidance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for record in records:
        if isinstance(record, dict):
            lines.append(_score_row(record))

    lines.extend(
        [
            "",
            "| Model | Primary | Cand In | Cand Out | Cand Cost | Cand / 1k | Judge Out | Judge Cost |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in records:
        if isinstance(record, dict):
            lines.append(_cost_row(record))

    lines.append("")
    for record in records:
        if isinstance(record, dict):
            lines.extend(_model_detail(record))
    return lines


def _score_row(record: dict[str, Any]) -> str:
    metrics = record.get("report_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    return (
        f"| {record.get('candidate_model', '')} "
        f"| {_fmt_score(record.get('primary_score'))} "
        f"| {_fmt_score(metrics.get('intent_fit'))} "
        f"| {_fmt_score(metrics.get('accuracy'))} "
        f"| {_fmt_score(metrics.get('groundedness'))} "
        f"| {_fmt_score(metrics.get('safety_conservatism'))} "
        f"| {_fmt_score(metrics.get('handoff_appropriateness'))} "
        f"| {_fmt_score(metrics.get('guidance_quality'))} |"
    )


def _model_detail(record: dict[str, Any]) -> list[str]:
    model = str(record.get("candidate_model", "unknown_model"))
    answer = str(record.get("answer_text", "")).strip() or "N/A"
    timing = record.get("timing_ms", {})
    candidate_cost = estimate_token_cost_usd(model, _dict_or_empty(record.get("token_usage")))
    lines = [
        f"### {model}",
        "",
        f"- primary_score: {_fmt_score(record.get('primary_score'))}",
        f"- timing_ms: `{json.dumps(timing, ensure_ascii=False)}`",
        f"- token_usage: `{json.dumps(_dict_or_empty(record.get('token_usage')), ensure_ascii=False)}`",
        f"- estimated_candidate_cost_usd: {_fmt_usd(candidate_cost.usd) if candidate_cost else 'N/A'}",
        "",
        "**Candidate Answer**",
        "",
        "> " + answer.replace("\n", "\n> "),
        "",
        "**Judge Summary**",
        "",
    ]
    for evaluation in record.get("judge_evaluations", []):
        if not isinstance(evaluation, dict):
            continue
        summary = evaluation.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        lines.append(
            f"- `{evaluation.get('judge_model', '')}`: "
            f"{summary.get('overall_profile', '')} "
            f"(strongest: {summary.get('strongest_dimension', '')}, "
            f"weakest: {summary.get('weakest_dimension', '')})"
        )
    lines.append("")
    lines.extend(_judge_metric_details(record))
    lines.append("")
    return lines


def _cost_row(record: dict[str, Any]) -> str:
    model = str(record.get("candidate_model", ""))
    candidate_usage = _dict_or_empty(record.get("token_usage"))
    usage = normalize_token_usage(candidate_usage)
    candidate_cost = estimate_token_cost_usd(model, candidate_usage)
    judge_cost = _total_judge_cost(record)
    return (
        f"| {model} "
        f"| {_fmt_score(record.get('primary_score'))} "
        f"| {_fmt_int(usage.input_tokens)} "
        f"| {_fmt_int(usage.output_tokens)} "
        f"| {_fmt_usd(candidate_cost.usd) if candidate_cost else 'N/A'} "
        f"| {_fmt_usd4(candidate_cost.usd * 1000) if candidate_cost else 'N/A'} "
        f"| {_fmt_int(judge_cost['output_tokens'])} "
        f"| {_fmt_usd(judge_cost['usd']) if judge_cost['known'] else 'N/A'} |"
    )


def _total_judge_cost(record: dict[str, Any]) -> dict[str, float | int | bool]:
    total_usd = 0.0
    output_tokens = 0
    known = False
    for evaluation in record.get("judge_evaluations", []):
        if not isinstance(evaluation, dict):
            continue
        model = str(evaluation.get("judge_model", ""))
        token_usage = _dict_or_empty(evaluation.get("token_usage"))
        usage = normalize_token_usage(token_usage)
        output_tokens += usage.output_tokens
        if usage.total_tokens == 0:
            continue
        cost = estimate_token_cost_usd(model, token_usage)
        if cost is not None:
            known = True
            total_usd += cost.usd
    return {"usd": total_usd, "output_tokens": output_tokens, "known": known}


def _judge_metric_details(record: dict[str, Any]) -> list[str]:
    lines = [
        "<details>",
        "<summary>Judge Metric Details</summary>",
        "",
        "| Judge | Intent | Accuracy | Grounded | Safety | Handoff | Guidance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for evaluation in record.get("judge_evaluations", []):
        if not isinstance(evaluation, dict):
            continue
        metrics = evaluation.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        lines.append(
            f"| {evaluation.get('judge_model', '')} "
            f"| {_metric_score(metrics, 'intent_fit')} "
            f"| {_metric_score(metrics, 'accuracy')} "
            f"| {_metric_score(metrics, 'groundedness')} "
            f"| {_metric_score(metrics, 'safety_conservatism')} "
            f"| {_metric_score(metrics, 'handoff_appropriateness')} "
            f"| {_metric_score(metrics, 'guidance_quality')} |"
        )
    lines.extend(["", "</details>"])
    return lines


def _metric_score(metrics: dict[str, Any], name: str) -> str:
    metric = metrics.get(name, {})
    if not isinstance(metric, dict):
        return "-"
    return _fmt_score(metric.get("score"))


def _extract_block(text: str, name: str) -> str:
    marker = f"[{name}]"
    if marker not in text:
        return ""
    start = text.index(marker) + len(marker)
    rest = text[start:].lstrip("\n")
    next_marker = rest.find("\n[")
    if next_marker == -1:
        return rest.strip()
    return rest[:next_marker].strip()


def _fmt_score(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return "-"


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt_usd(value: float) -> str:
    return f"${value:.6f}"


def _fmt_usd4(value: float) -> str:
    return f"${value:.4f}"


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _mean(values: list[float | int]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _total_ms_values(records: list[dict[str, Any]]) -> list[int]:
    values: list[int] = []
    for record in records:
        timing = record.get("timing_ms", {})
        if isinstance(timing, dict) and isinstance(timing.get("total"), (int, float)):
            values.append(int(timing["total"]))
        elif isinstance(record.get("latency_ms"), (int, float)):
            values.append(int(record["latency_ms"]))
    return values


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return data


if __name__ == "__main__":
    main()

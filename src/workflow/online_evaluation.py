"""Online model comparison for counselor-facing workflow telemetry."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.common.schemas import (
    LLMBatchResponse,
    LLMRequest,
    LLMResponse,
    NLUEvidence,
    WorkflowOutput,
    WorkflowRoutingInput,
)
from src.evaluation.clients import CompositeCandidateClient, CompositeJudgeClient
from src.evaluation.run_evaluation import (
    default_candidate_models_from_env,
    default_judge_models_from_env,
)
from src.evaluation.rubrics import get_judge_rubric
from src.evaluation.schemas import (
    CandidateModel,
    EvaluationScenario,
    JudgeEvaluation,
    JudgeModel,
)
from src.evaluation.scoring import aggregate_judge_evaluations, compute_primary_score
from src.workflow.context_builder import ContextBuilder
from src.workflow.formatter import format_workflow_output
from src.workflow.graph import _select_route_with_reason
from src.workflow.prompt import build_system_prompt


class CandidateClient(Protocol):
    async def generate(self, model: CandidateModel, request: LLMRequest) -> LLMResponse:
        """Generates one model candidate."""


class JudgeClient(Protocol):
    async def evaluate(
        self,
        judge: JudgeModel,
        scenario: EvaluationScenario,
        answer: LLMResponse,
    ) -> JudgeEvaluation:
        """Evaluates one generated candidate answer."""


class OnlineEvaluationStore(Protocol):
    def save(self, session_id: str, payload: dict[str, Any]) -> str:
        """Persists the online evaluation payload and returns a detail reference."""


class FileOnlineEvaluationStore:
    """Stores online counselor-panel payloads as local JSON files."""

    def __init__(self, root: str | Path = "evaluation_runs/online_sessions") -> None:
        self.root = Path(root)

    def save(self, session_id: str, payload: dict[str, Any]) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{_safe_filename(session_id)}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)


@dataclass(frozen=True)
class OnlineWorkflowRun:
    """Immediate workflow result plus a background evaluation task."""

    workflow_output: WorkflowOutput
    evaluation_task: asyncio.Task[dict[str, Any]]


class OnlineEvaluationService:
    """Runs Solar fast path and shadow model evaluation for counselor panels."""

    def __init__(
        self,
        *,
        candidate_client: CandidateClient | None = None,
        judge_client: JudgeClient | None = None,
        store: OnlineEvaluationStore | None = None,
        candidate_models: list[CandidateModel] | None = None,
        judge_models: list[JudgeModel] | None = None,
        customer_provider: str = "solar",
    ) -> None:
        rubric = get_judge_rubric("comparative_10")
        self.candidate_client = candidate_client or CompositeCandidateClient()
        self.judge_client = judge_client or CompositeJudgeClient(
            prompt_path=rubric.prompt_path,
            metric_names=rubric.metric_names,
            score_min=rubric.score_min,
            score_max=rubric.score_max,
            score_scale_label=rubric.score_scale_label,
            primary_score_source=rubric.primary_score_source,
        )
        self.store = store or FileOnlineEvaluationStore()
        self.candidate_models = candidate_models or default_candidate_models_from_env()
        self.judge_models = judge_models or default_judge_models_from_env()
        self.customer_provider = customer_provider
        self.metric_names = rubric.metric_names
        self.score_min = rubric.score_min
        self.score_max = rubric.score_max
        self.report_normalized_scores = rubric.report_normalized_scores

    async def start(self, payload: WorkflowRoutingInput) -> OnlineWorkflowRun:
        """Starts all candidate calls and returns as soon as the customer model finishes."""
        if not self.candidate_models:
            raise ValueError("At least one online candidate model is required.")
        request, evidence = build_workflow_request(payload)
        tasks = {
            model.provider: asyncio.create_task(
                self._generate_candidate(model=model, request=request)
            )
            for model in self.candidate_models
        }
        customer_provider = (
            self.customer_provider
            if self.customer_provider in tasks
            else self.candidate_models[0].provider
        )
        customer_response = await tasks[customer_provider]
        workflow_output = self._build_workflow_output(
            payload=payload,
            evidence=evidence,
            response=customer_response,
        )
        evaluation_task = asyncio.create_task(
            self._finish_evaluation(
                payload=payload,
                evidence=evidence,
                tasks=tasks,
                customer_provider=customer_provider,
            )
        )
        return OnlineWorkflowRun(
            workflow_output=workflow_output,
            evaluation_task=evaluation_task,
        )

    async def _generate_candidate(
        self,
        *,
        model: CandidateModel,
        request: LLMRequest,
    ) -> LLMResponse:
        try:
            return await self.candidate_client.generate(model, request)
        except Exception as exc:
            return LLMResponse(
                session_id=request.session_id,
                provider=model.provider,
                text="",
                latency_ms=0,
                ttft_ms=0,
                finish_reason=None,
                error=exc.__class__.__name__,
                token_usage={},
            )

    def _build_workflow_output(
        self,
        *,
        payload: WorkflowRoutingInput,
        evidence: NLUEvidence,
        response: LLMResponse,
    ) -> WorkflowOutput:
        output = format_workflow_output(
            LLMBatchResponse(session_id=payload.session_id, responses=[response])
        )
        reference_links = _reference_links(payload, output.reference_links)
        return output.model_copy(
            update={
                "nlu_evidence": evidence,
                "reference_links": reference_links,
            }
        )

    async def _finish_evaluation(
        self,
        *,
        payload: WorkflowRoutingInput,
        evidence: NLUEvidence,
        tasks: dict[str, asyncio.Task[LLMResponse]],
        customer_provider: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        responses = [await task for task in tasks.values()]
        scenario = _scenario_from_payload(payload)
        panels = await asyncio.gather(
            *[
                self._build_model_panel(
                    scenario=scenario,
                    response=response,
                    is_customer_answer=response.provider == customer_provider,
                )
                for response in responses
            ]
        )
        payload_json: dict[str, Any] = {
            "type": "evaluation_result",
            "session_id": payload.session_id,
            "route": evidence.selected_route.value,
            "route_reason": evidence.route_reason,
            "customer_provider": customer_provider,
            "model_panels": list(panels),
            "timing_ms": {"evaluation_total": int((time.perf_counter() - started) * 1000)},
        }
        detail_ref = self.store.save(payload.session_id, payload_json)
        for panel in payload_json["model_panels"]:
            panel["details_ref"] = detail_ref
        self.store.save(payload.session_id, payload_json)
        return payload_json

    async def _build_model_panel(
        self,
        *,
        scenario: EvaluationScenario,
        response: LLMResponse,
        is_customer_answer: bool,
    ) -> dict[str, Any]:
        base = {
            "provider": response.provider,
            "is_customer_answer": is_customer_answer,
            "answer": response.text,
            "latency_ms": response.latency_ms,
            "token_usage": response.token_usage,
            "evaluation_status": "completed" if not response.error and response.text else "failed",
            "error": response.error,
        }
        if response.error or not response.text:
            return {
                **base,
                "score": None,
                "quality_badge": "확인 필요",
                "summary": "모델 응답 생성에 실패했습니다.",
                "flags": ["candidate_generation_failed"],
                "metrics": {},
            }

        judge_evaluations = await asyncio.gather(
            *[
                self.judge_client.evaluate(judge, scenario, response)
                for judge in self.judge_models
            ]
        )
        aggregated = aggregate_judge_evaluations(
            list(judge_evaluations),
            metric_names=self.metric_names,
            score_min=self.score_min,
            score_max=self.score_max,
        )
        score = compute_primary_score(
            aggregated,
            metric_names=self.metric_names,
            use_normalized=self.report_normalized_scores,
        )
        flags = _merge_flags(judge_evaluations)
        return {
            **base,
            "score": round(score, 2),
            "quality_badge": _quality_badge(score),
            "summary": _panel_summary(judge_evaluations),
            "flags": flags,
            "metrics": {
                name: {
                    "score": metric.raw_median,
                    "disagreement": metric.disagreement,
                }
                for name, metric in aggregated.metrics.items()
            },
            "judge_models": [evaluation.judge_model for evaluation in judge_evaluations],
        }


def build_workflow_request(
    payload: WorkflowRoutingInput,
) -> tuple[LLMRequest, NLUEvidence]:
    """Builds the real workflow prompt and NLU evidence once for online fan-out."""
    route, route_reason = _select_route_with_reason(payload)
    request = ContextBuilder().build_request(
        payload=payload,
        route=route,
        system_prompt=build_system_prompt(route),
    )
    evidence = NLUEvidence(
        intent=payload.routing_info.intent,
        domain=payload.routing_info.domain,
        subdomain=payload.routing_info.subdomain,
        router_confidence=payload.routing_info.router_confidence,
        selected_route=route,
        route_reason=route_reason,
        metadata=payload.routing_info.metadata,
    )
    return request, evidence


def build_workflow_result_event(
    *,
    session_id: str,
    transcript_text: str,
    nlu_payload: dict[str, Any],
    workflow_output: WorkflowOutput | None,
    action: Any,
    evaluation_status: str,
) -> dict[str, Any]:
    """Builds the first websocket event while keeping the existing response shape."""
    return {
        "type": "workflow_result",
        "session_id": session_id,
        "transcript": transcript_text,
        "is_final": True,
        "nlu_analysis": nlu_payload,
        "workflow": workflow_output.model_dump(mode="json") if workflow_output else None,
        "action": action,
        "evaluation_status": evaluation_status,
    }


def _scenario_from_payload(payload: WorkflowRoutingInput) -> EvaluationScenario:
    return EvaluationScenario(
        scenario_id=payload.session_id,
        user_query=payload.original_query,
        intent=payload.routing_info.intent,
        domain=payload.routing_info.domain,
        subdomain=payload.routing_info.subdomain,
        router_confidence=payload.routing_info.router_confidence,
        retrieved_context="\n".join(item.content for item in payload.internal_context),
        reference_answer="",
        policy_rules=[rule.model_dump(mode="json") for rule in payload.policy_rules],
        metadata=payload.routing_info.metadata,
    )


def _reference_links(payload: WorkflowRoutingInput, links: list[str]) -> list[str]:
    source_url = str(payload.routing_info.metadata.get("source_url", "")).strip()
    output = [*links]
    if source_url and source_url not in output:
        output.append(source_url)
    return output


def _merge_flags(evaluations: list[JudgeEvaluation]) -> list[str]:
    flags: list[str] = []
    for evaluation in evaluations:
        for key, value in evaluation.flags.items():
            if value and key not in flags:
                flags.append(key)
    return flags


def _panel_summary(evaluations: list[JudgeEvaluation]) -> str:
    for evaluation in evaluations:
        profile = str(evaluation.summary.get("overall_profile", "")).strip()
        if profile:
            return profile
    return "평가가 완료되었습니다."


def _quality_badge(score: float) -> str:
    if score >= 8.0:
        return "좋음"
    if score >= 6.0:
        return "주의"
    return "확인 필요"


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe or "session"


__all__ = [
    "FileOnlineEvaluationStore",
    "OnlineEvaluationService",
    "OnlineWorkflowRun",
    "build_workflow_request",
    "build_workflow_result_event",
]

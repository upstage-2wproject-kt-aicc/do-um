"""LangGraph skeleton for four-way workflow routing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

try:
    from langgraph.graph import END, StateGraph
except ModuleNotFoundError:  # pragma: no cover - runtime dependency guard
    END = "__end__"
    StateGraph = object  # type: ignore[assignment]

from src.common.schemas import (
    IntentResult,
    NLUEvidence,
    RouteType,
    Transcript,
    WorkflowRoutingInput,
    WorkflowRoutingResult,
)
from src.common.schemas import WorkflowOutput
from src.workflow.context_builder import ContextBuilder
from src.workflow.formatter import format_workflow_output
from src.workflow.multi_llm import MultiLLMService
from src.workflow.prompt import build_system_prompt


SECURITY_KEYWORDS: tuple[str, ...] = (
    "보안",
    "분실",
    "도난",
    "피싱",
    "해킹",
    "사기",
    "인증",
    "비밀번호",
    "잠김",
    "명의도용",
)
LLM_SERVICE = MultiLLMService()


def _is_high_risk(value: Any) -> bool:
    """Normalizes risk_level metadata to a high-risk boolean."""
    normalized = str(value).strip().lower()
    return normalized in {"high", "높음", "상", "critical"}


def _is_handoff_required(value: Any) -> bool:
    """Normalizes handoff_required metadata to a required boolean."""
    normalized = str(value).strip().lower()
    return normalized in {"y", "yes", "true", "1", "required"}


async def faq_node(state: WorkflowRoutingInput) -> WorkflowRoutingInput:
    """Handles FAQ route state transitions."""
    return state


async def handoff_node(state: WorkflowRoutingInput) -> WorkflowRoutingInput:
    """Handles handoff route state transitions."""
    return state


async def procedure_node(state: WorkflowRoutingInput) -> WorkflowRoutingInput:
    """Handles procedure route state transitions."""
    return state


async def security_node(state: WorkflowRoutingInput) -> WorkflowRoutingInput:
    """Handles security route state transitions."""
    return state


def route_selector(state: WorkflowRoutingInput) -> RouteType:
    """Selects one of the four workflow routes."""
    route, _ = _select_route_with_reason(state)
    return route


def _select_route_with_reason(state: WorkflowRoutingInput) -> tuple[RouteType, str]:
    """Selects route and keeps human-readable decision reason."""
    text_space = " ".join(
        [state.original_query, state.routing_info.subdomain, state.routing_info.domain]
    ).lower()
    if any(keyword in text_space for keyword in SECURITY_KEYWORDS):
        return RouteType.SECURITY, "security_keyword_match"

    metadata = state.routing_info.metadata
    if _is_high_risk(metadata.get("risk_level")):
        return RouteType.HANDOFF, "metadata_risk_level_high"
    if _is_handoff_required(metadata.get("handoff_required")):
        return RouteType.HANDOFF, "metadata_handoff_required"

    intent = state.routing_info.intent.strip()
    if intent == "절차형":
        return RouteType.PROCEDURE, "intent_procedure"
    return RouteType.FAQ, "default_faq"


def build_workflow_graph() -> StateGraph:
    """Builds the LangGraph state graph skeleton."""
    if StateGraph is object:
        raise RuntimeError("langgraph is required to build workflow graph.")
    graph = StateGraph(WorkflowRoutingInput)
    graph.add_node("router", faq_node)
    graph.add_node("faq", faq_node)
    graph.add_node("handoff", handoff_node)
    graph.add_node("procedure", procedure_node)
    graph.add_node("security", security_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _route_key,
        {
            RouteType.FAQ.value: "faq",
            RouteType.HANDOFF.value: "handoff",
            RouteType.PROCEDURE.value: "procedure",
            RouteType.SECURITY.value: "security",
        },
    )
    graph.add_edge("faq", END)
    graph.add_edge("handoff", END)
    graph.add_edge("procedure", END)
    graph.add_edge("security", END)
    return graph


def compile_workflow_graph() -> object:
    """Compiles the LangGraph workflow graph object."""
    return build_workflow_graph().compile()


def parse_workflow_inputs(payload: dict | list[dict]) -> list[WorkflowRoutingInput]:
    """Parses JSON payload into validated routing input objects."""
    if isinstance(payload, dict):
        return [WorkflowRoutingInput.model_validate(payload)]
    if isinstance(payload, list):
        return [WorkflowRoutingInput.model_validate(item) for item in payload]
    raise TypeError("Payload must be dict or list[dict].")


def load_workflow_inputs_from_json(path: str | Path) -> list[WorkflowRoutingInput]:
    """Loads and validates workflow routing inputs from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_workflow_inputs(data)


def parse_nlu_outputs(payload: dict | list[dict]) -> list[WorkflowRoutingInput]:
    """Parses NLU output payload into validated workflow routing input objects."""
    if isinstance(payload, dict):
        return [workflow_input_from_nlu_dict(payload)]
    if isinstance(payload, list):
        return [workflow_input_from_nlu_dict(item) for item in payload]
    raise TypeError("Payload must be dict or list[dict].")


def load_nlu_outputs_from_json(path: str | Path) -> list[WorkflowRoutingInput]:
    """Loads NLU output JSON and converts it to workflow routing inputs."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_nlu_outputs(data)


def route_from_inputs(inputs: list[WorkflowRoutingInput]) -> list[WorkflowRoutingResult]:
    """Computes route results from validated workflow inputs."""
    return [
        WorkflowRoutingResult(
            session_id=item.session_id,
            selected_route=route_selector(item),
            original_query=item.original_query,
        )
        for item in inputs
    ]


def load_and_route_json(path: str | Path) -> list[WorkflowRoutingResult]:
    """Loads JSON workflow inputs and returns route-only results."""
    return route_from_inputs(load_workflow_inputs_from_json(path))


async def execute_workflow_item(payload: WorkflowRoutingInput) -> WorkflowOutput:
    """Executes one workflow item from routing to normalized output."""
    route, route_reason = _select_route_with_reason(payload)
    builder = ContextBuilder()
    request = builder.build_request(
        payload=payload,
        route=route,
        system_prompt=build_system_prompt(route),
    )
    batch = await LLM_SERVICE.invoke_all(request)
    output = format_workflow_output(batch)
    source_url = str(payload.routing_info.metadata.get("source_url", "")).strip()
    reference_links = output.reference_links
    if source_url and not reference_links:
        reference_links = [source_url]
    evidence = NLUEvidence(
        intent=payload.routing_info.intent,
        domain=payload.routing_info.domain,
        subdomain=payload.routing_info.subdomain,
        router_confidence=payload.routing_info.router_confidence,
        selected_route=route,
        route_reason=route_reason,
        metadata=payload.routing_info.metadata,
    )
    return output.model_copy(
        update={
            "nlu_evidence": evidence,
            "reference_links": reference_links,
        }
    )


async def execute_workflow_json(path: str | Path) -> list[WorkflowOutput]:
    """Executes workflow for all JSON inputs and returns formatted outputs."""
    inputs = load_workflow_inputs_from_json(path)
    outputs = [execute_workflow_item(item) for item in inputs]
    return await asyncio.gather(*outputs)


async def execute_workflow_nlu_json(path: str | Path) -> list[WorkflowOutput]:
    """Executes workflow from NLU output JSON payload."""
    inputs = load_nlu_outputs_from_json(path)
    outputs = [execute_workflow_item(item) for item in inputs]
    return await asyncio.gather(*outputs)


def workflow_input_from_nlu_result(
    transcript: Transcript,
    intent_result: IntentResult,
    chat_history: list[dict[str, Any]] | None = None,
    internal_context: list[dict[str, Any]] | None = None,
    policy_rules: list[dict[str, Any]] | None = None,
) -> WorkflowRoutingInput:
    """Builds workflow input from typed NLU outputs."""
    payload: dict[str, Any] = {
        "session_id": transcript.session_id,
        "user_query": transcript.text,
        "intent_result": intent_result.model_dump(mode="json"),
        "chat_history": chat_history or [],
        "internal_context": internal_context or [],
        "policy_rules": policy_rules or [],
    }
    return workflow_input_from_nlu_dict(payload)


def workflow_input_from_nlu_dict(payload: dict[str, Any]) -> WorkflowRoutingInput:
    """Converts one raw NLU output dict into WorkflowRoutingInput."""
    session_id = str(
        payload.get("session_id")
        or payload.get("transcript", {}).get("session_id", "")
        or payload.get("intent_result", {}).get("session_id", "")
    )
    user_query = str(
        payload.get("user_query")
        or payload.get("original_query")
        or payload.get("transcript", {}).get("text", "")
    )

    routing_info = payload.get("routing_info")
    if isinstance(routing_info, dict):
        normalized_routing = {
            "intent": routing_info.get("intent", ""),
            "subdomain": routing_info.get("subdomain", ""),
            "router_confidence": routing_info.get("router_confidence", 0.0),
            "domain": routing_info.get("domain", ""),
            "metadata": routing_info.get("metadata", payload.get("metadata", {})),
        }
    else:
        intent_result = payload.get("intent_result", {})
        rag_meta = (
            payload.get("metadata")
            or intent_result.get("rag_context", {}).get("metadata", {})
            or {}
        )
        normalized_routing = {
            "intent": payload.get("intent") or intent_result.get("intent", ""),
            "subdomain": rag_meta.get("subdomain", ""),
            "router_confidence": payload.get("router_confidence")
            or intent_result.get("score", 0.0),
            "domain": rag_meta.get("domain", ""),
            "metadata": rag_meta,
        }

    internal_context = payload.get("internal_context", [])
    retrieved_context = payload.get("retrieved_context")
    if retrieved_context and not internal_context:
        internal_context = [
            {
                "source": "nlu_rag",
                "content": retrieved_context,
                "metadata": payload.get("metadata", {}),
            }
        ]

    return WorkflowRoutingInput.model_validate(
        {
            "session_id": session_id,
            "original_query": user_query,
            "routing_info": normalized_routing,
            "chat_history": payload.get("chat_history", []),
            "internal_context": internal_context,
            "policy_rules": payload.get("policy_rules", []),
        }
    )


def _route_key(state: WorkflowRoutingInput) -> str:
    """Converts route enum into LangGraph conditional edge key."""
    return route_selector(state).value


__all__ = [
    "END",
    "faq_node",
    "handoff_node",
    "procedure_node",
    "security_node",
    "route_selector",
    "build_workflow_graph",
    "compile_workflow_graph",
    "parse_workflow_inputs",
    "parse_nlu_outputs",
    "load_workflow_inputs_from_json",
    "load_nlu_outputs_from_json",
    "route_from_inputs",
    "load_and_route_json",
    "execute_workflow_item",
    "execute_workflow_json",
    "execute_workflow_nlu_json",
    "workflow_input_from_nlu_result",
    "workflow_input_from_nlu_dict",
]

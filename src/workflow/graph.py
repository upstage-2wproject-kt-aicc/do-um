"""LangGraph skeleton for four-way workflow routing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

try:
    from langgraph.graph import END, StateGraph
except ModuleNotFoundError:  # pragma: no cover - runtime dependency guard
    END = "__end__"
    StateGraph = object  # type: ignore[assignment]

from src.common.schemas import RouteType, WorkflowRoutingInput, WorkflowRoutingResult
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
    text_space = " ".join(
        [state.original_query, state.routing_info.subdomain, state.routing_info.domain]
    ).lower()
    if any(keyword in text_space for keyword in SECURITY_KEYWORDS):
        return RouteType.SECURITY
    intent = state.routing_info.intent.strip()
    if intent == "민원형":
        return RouteType.HANDOFF
    if intent == "절차형":
        return RouteType.PROCEDURE
    return RouteType.FAQ


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
    route = route_selector(payload)
    builder = ContextBuilder()
    request = builder.build_request(
        payload=payload,
        route=route,
        system_prompt=build_system_prompt(route),
    )
    batch = await MultiLLMService().invoke_all(request)
    return format_workflow_output(batch)


async def execute_workflow_json(path: str | Path) -> list[WorkflowOutput]:
    """Executes workflow for all JSON inputs and returns formatted outputs."""
    inputs = load_workflow_inputs_from_json(path)
    outputs = [execute_workflow_item(item) for item in inputs]
    return await asyncio.gather(*outputs)


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
    "load_workflow_inputs_from_json",
    "route_from_inputs",
    "load_and_route_json",
    "execute_workflow_item",
    "execute_workflow_json",
]

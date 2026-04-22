"""LangGraph skeleton for four-way workflow routing."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.common.schemas import RouteType, WorkflowState


async def faq_node(state: WorkflowState) -> WorkflowState:
    """Handles FAQ route state transitions."""
    raise NotImplementedError


async def handoff_node(state: WorkflowState) -> WorkflowState:
    """Handles handoff route state transitions."""
    raise NotImplementedError


async def procedure_node(state: WorkflowState) -> WorkflowState:
    """Handles procedure route state transitions."""
    raise NotImplementedError


async def security_node(state: WorkflowState) -> WorkflowState:
    """Handles security route state transitions."""
    raise NotImplementedError


def route_selector(state: WorkflowState) -> RouteType:
    """Selects one of the four workflow routes."""
    raise NotImplementedError


def build_workflow_graph() -> StateGraph:
    """Builds the LangGraph state graph skeleton."""
    raise NotImplementedError


def compile_workflow_graph() -> object:
    """Compiles the LangGraph workflow graph object."""
    raise NotImplementedError


__all__ = [
    "END",
    "faq_node",
    "handoff_node",
    "procedure_node",
    "security_node",
    "route_selector",
    "build_workflow_graph",
    "compile_workflow_graph",
]


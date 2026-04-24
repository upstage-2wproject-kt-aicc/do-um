"""Prompt templates for route-aware multi-LLM execution."""

from __future__ import annotations

from src.common.schemas import RouteType

BASE_SYSTEM_PROMPT = (
    "You are a financial customer-assistant agent. "
    "Follow policy rules strictly. "
    "Do not fabricate facts. "
    "If evidence is insufficient, state limitation and request handoff."
)

ROUTE_PROMPT_MAP: dict[RouteType, str] = {
    RouteType.FAQ: (
        "Route=FAQ. Explain clearly with concise structure. "
        "Cite evidence when available."
    ),
    RouteType.HANDOFF: (
        "Route=HANDOFF. Prioritize safe escalation language. "
        "Provide minimal guidance and handoff rationale."
    ),
    RouteType.PROCEDURE: (
        "Route=PROCEDURE. Return ordered step-by-step instructions. "
        "Avoid speculative steps."
    ),
    RouteType.SECURITY: (
        "Route=SECURITY. Prioritize fraud/risk prevention. "
        "Provide immediate protective actions and escalation path."
    ),
}


def build_system_prompt(route: RouteType) -> str:
    """Builds the final system prompt from base and route templates."""
    return f"{BASE_SYSTEM_PROMPT}\n{ROUTE_PROMPT_MAP[route]}"


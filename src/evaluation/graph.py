"""Evaluation graph helpers that reuse service workflow prompt rules."""

from __future__ import annotations

from src.common.schemas import WorkflowRoutingInput
from src.evaluation.schemas import EvaluationScenario
from src.workflow.context_builder import ContextBuilder
from src.workflow.graph import route_selector
from src.workflow.prompt import build_system_prompt


def build_workflow_input(scenario: EvaluationScenario) -> WorkflowRoutingInput:
    """Converts a fixed evaluation scenario into service workflow input."""
    internal_context = []
    if scenario.retrieved_context:
        internal_context.append(
            {
                "source": "evaluation_context",
                "content": scenario.retrieved_context,
                "metadata": scenario.metadata,
            }
        )

    return WorkflowRoutingInput.model_validate(
        {
            "session_id": scenario.scenario_id,
            "original_query": scenario.user_query,
            "routing_info": {
                "intent": scenario.intent,
                "domain": scenario.domain,
                "subdomain": scenario.subdomain,
                "router_confidence": scenario.router_confidence,
            },
            "internal_context": internal_context,
            "policy_rules": scenario.policy_rules,
        }
    )


def build_llm_request(scenario: EvaluationScenario):
    """Builds the same LLM request shape used by the service workflow."""
    payload = build_workflow_input(scenario)
    route = route_selector(payload)
    return ContextBuilder().build_request(
        payload=payload,
        route=route,
        system_prompt=build_system_prompt(route),
    )

"""Context synthesis utilities for workflow prompt construction."""

from __future__ import annotations

import os

from src.common.schemas import LLMRequest, RouteType, WorkflowRoutingInput

ROUTE_MAX_TOKENS: dict[RouteType, int] = {
    RouteType.FAQ: 512,
    RouteType.PROCEDURE: 448,
    RouteType.SECURITY: 320,
    RouteType.HANDOFF: 160,
}


class ContextBuilder:
    """Builds one consolidated prompt from multi-source workflow context."""

    def build_prompt(self, payload: WorkflowRoutingInput) -> str:
        """Merges query, chat history, internal data, and policy rules into one prompt."""
        history_block = self._format_history(payload)
        internal_block = self._format_internal(payload)
        metadata_block = self._format_routing_metadata(payload)
        policy_block = self._format_policy(payload)
        return (
            f"[USER_QUERY]\n{payload.original_query}\n\n"
            f"[CHAT_HISTORY]\n{history_block}\n\n"
            f"[INTERNAL_CONTEXT]\n{internal_block}\n\n"
            f"[ROUTING_METADATA]\n{metadata_block}\n\n"
            f"[POLICY_RULES]\n{policy_block}"
        )

    def build_request(
        self, payload: WorkflowRoutingInput, route: RouteType, system_prompt: str
    ) -> LLMRequest:
        """Builds an LLM request model from workflow routing input."""
        max_tokens = int(os.getenv("WORKFLOW_MAX_TOKENS", str(ROUTE_MAX_TOKENS[route])))
        return LLMRequest(
            session_id=payload.session_id,
            prompt=self.build_prompt(payload),
            system_prompt=system_prompt,
            route=route,
            max_tokens=max_tokens,
        )

    def _format_history(self, payload: WorkflowRoutingInput) -> str:
        """Formats chat history as line-based conversation context."""
        if not payload.chat_history:
            return "N/A"
        return "\n".join(
            f"{turn.role}: {turn.text} ({turn.timestamp})" for turn in payload.chat_history
        )

    def _format_internal(self, payload: WorkflowRoutingInput) -> str:
        """Formats internal API/DB context entries."""
        if not payload.internal_context:
            return "N/A"
        lines = []
        for item in payload.internal_context:
            line = f"- [{item.source}] {item.content}"
            source_url = str(item.metadata.get("source_url", "")).strip()
            if source_url:
                line = f"{line}\n  source_url: {source_url}"
            lines.append(line)
        return "\n".join(lines)

    def _format_routing_metadata(self, payload: WorkflowRoutingInput) -> str:
        """Formats routing metadata that can affect safety, handoff, and citations."""
        metadata = payload.routing_info.metadata
        if not metadata:
            return "N/A"
        allowed_keys = (
            "risk_level",
            "handoff_required",
            "source_url",
            "retrieval_status",
            "keywords",
        )
        lines = []
        for key in allowed_keys:
            value = metadata.get(key)
            if value not in (None, "", []):
                lines.append(f"- {key}: {value}")
        return "\n".join(lines) if lines else "N/A"

    def _format_policy(self, payload: WorkflowRoutingInput) -> str:
        """Formats policy rule entries for guardrail conditioning."""
        if not payload.policy_rules:
            return "N/A"
        return "\n".join(
            f"- ({rule.rule_id}) {rule.title}: {rule.description}"
            for rule in payload.policy_rules
        )

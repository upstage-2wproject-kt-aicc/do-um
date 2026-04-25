"""Context synthesis utilities for workflow prompt construction."""

from __future__ import annotations

import os

from src.common.schemas import LLMRequest, RouteType, WorkflowRoutingInput


class ContextBuilder:
    """Builds one consolidated prompt from multi-source workflow context."""

    def build_prompt(self, payload: WorkflowRoutingInput) -> str:
        """Merges query, chat history, internal data, and policy rules into one prompt."""
        history_block = self._format_history(payload)
        internal_block = self._format_internal(payload)
        policy_block = self._format_policy(payload)
        return (
            f"[USER_QUERY]\n{payload.original_query}\n\n"
            f"[CHAT_HISTORY]\n{history_block}\n\n"
            f"[INTERNAL_CONTEXT]\n{internal_block}\n\n"
            f"[POLICY_RULES]\n{policy_block}"
        )

    def build_request(
        self, payload: WorkflowRoutingInput, route: RouteType, system_prompt: str
    ) -> LLMRequest:
        """Builds an LLM request model from workflow routing input."""
        max_tokens = int(os.getenv("WORKFLOW_MAX_TOKENS", "220"))
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
        return "\n".join(
            f"- [{item.source}] {item.content}" for item in payload.internal_context
        )

    def _format_policy(self, payload: WorkflowRoutingInput) -> str:
        """Formats policy rule entries for guardrail conditioning."""
        if not payload.policy_rules:
            return "N/A"
        return "\n".join(
            f"- ({rule.rule_id}) {rule.title}: {rule.description}"
            for rule in payload.policy_rules
        )

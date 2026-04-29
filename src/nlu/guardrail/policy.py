"""Guardrail decision -> policy_rules 변환."""

from __future__ import annotations


def build_policy_rules(decision: str, limit_message: str) -> list[dict[str, str]]:
    if decision != "LIMIT":
        return []
    return [
        {
            "rule_id": "guardrail_limit",
            "title": "안전 축소 답변",
            "description": limit_message,
        }
    ]


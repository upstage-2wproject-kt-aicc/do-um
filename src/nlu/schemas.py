"""NLU 단계 간 데이터 계약(스키마) 정의."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NLUCoreResult:
    intent: str
    subdomain_pred: str | None
    query_vector: Any
    timings: dict[str, float | None] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    retrieved_context: str
    metadata: dict[str, Any]
    retrieved_faq_ids: list[str]
    routing_signals: dict[str, Any]


@dataclass
class GuardrailResult:
    decision: str
    score: float
    reasons: list[str]
    components: dict[str, int] = field(default_factory=dict)


@dataclass
class NLUFinalResult:
    payload: dict[str, Any]


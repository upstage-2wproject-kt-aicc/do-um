"""Semantic-like cache layer for repeated user questions."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from src.common.schemas import WorkflowOutput


@dataclass
class CacheEntry:
    """Represents one cached workflow output entry."""

    key: str
    output: WorkflowOutput
    created_at: float


class SemanticCache:
    """Stores and retrieves near-duplicate query answers."""

    def __init__(self, ttl_s: float = 600.0) -> None:
        """Initializes cache with entry TTL in seconds."""
        self.ttl_s = ttl_s
        self._entries: dict[str, CacheEntry] = {}

    def get(self, query: str) -> WorkflowOutput | None:
        """Returns cached output for a semantically similar query."""
        self._evict_expired()
        key = self._normalize(query)
        if key in self._entries:
            return self._entries[key].output
        for entry in self._entries.values():
            if self._similarity(key, entry.key) >= 0.9:
                return entry.output
        return None

    def set(self, query: str, output: WorkflowOutput) -> None:
        """Caches workflow output for later query reuse."""
        key = self._normalize(query)
        self._entries[key] = CacheEntry(key=key, output=output, created_at=time.time())

    def _normalize(self, text: str) -> str:
        """Normalizes text into cache key representation."""
        lowered = text.strip().lower()
        return re.sub(r"\s+", " ", lowered)

    def _similarity(self, left: str, right: str) -> float:
        """Computes token-overlap similarity score."""
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        if not left_tokens or not right_tokens:
            return 0.0
        common = len(left_tokens & right_tokens)
        total = len(left_tokens | right_tokens)
        return common / total

    def _evict_expired(self) -> None:
        """Deletes expired cache entries by TTL."""
        now = time.time()
        expired = [
            key
            for key, entry in self._entries.items()
            if now - entry.created_at > self.ttl_s
        ]
        for key in expired:
            self._entries.pop(key, None)


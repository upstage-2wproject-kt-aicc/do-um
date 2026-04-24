"""NLU classifier interfaces for dual-track intent modeling."""

from __future__ import annotations

from abc import ABC, abstractmethod

from common.schemas import IntentResult, Transcript


class BaseClassifier(ABC):
    """Defines the async interface for intent classification."""

    @abstractmethod
    async def classify(self, transcript: Transcript) -> IntentResult:
        """Classifies intent from transcript input."""
        raise NotImplementedError


class sLLMClassifier(BaseClassifier):
    """Implements classifier contract using a small LLM approach."""

    async def classify(self, transcript: Transcript) -> IntentResult:
        """Classifies intent using sLLM-based inference."""
        raise NotImplementedError


class DLClassifier(BaseClassifier):
    """Implements classifier contract using deep learning models."""

    async def classify(self, transcript: Transcript) -> IntentResult:
        """Classifies intent using DL-based inference."""
        raise NotImplementedError


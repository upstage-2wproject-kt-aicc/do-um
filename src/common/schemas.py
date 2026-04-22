"""Shared Pydantic schemas for all module boundaries."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RouteType(str, Enum):
    """Defines the workflow route categories."""

    FAQ = "faq"
    HANDOFF = "handoff"
    PROCEDURE = "procedure"
    SECURITY = "security"


class AudioChunk(BaseModel):
    """Represents a chunk of inbound audio bytes."""

    session_id: str = Field(..., description="Unique conversation/session ID.")
    chunk_id: int = Field(..., ge=0, description="Monotonic audio chunk index.")
    sample_rate_hz: int = Field(..., gt=0, description="Audio sample rate.")
    channels: int = Field(..., gt=0, description="Number of audio channels.")
    pcm_bytes: bytes = Field(..., description="Raw PCM bytes.")
    timestamp_ms: int = Field(..., ge=0, description="Chunk timestamp in ms.")


class Transcript(BaseModel):
    """Represents STT output text and metadata."""

    session_id: str = Field(..., description="Unique conversation/session ID.")
    text: str = Field(..., description="Recognized text.")
    language: str = Field(..., description="Language code.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="ASR confidence.")
    is_final: bool = Field(..., description="Whether transcript is finalized.")


class RagContext(BaseModel):
    """Carries retrieved context for downstream reasoning."""

    session_id: str = Field(..., description="Unique conversation/session ID.")
    query: str = Field(..., description="Normalized retrieval query.")
    documents: list[str] = Field(default_factory=list, description="Retrieved docs.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="RAG metadata.")


class IntentResult(BaseModel):
    """Carries NLU intent classification output."""

    session_id: str = Field(..., description="Unique conversation/session ID.")
    intent: str = Field(..., description="Predicted intent label.")
    route: RouteType = Field(..., description="Mapped workflow route.")
    score: float = Field(..., ge=0.0, le=1.0, description="Intent confidence.")
    rag_context: RagContext | None = Field(None, description="Optional RAG context.")


class WorkflowState(BaseModel):
    """Captures state exchanged inside workflow orchestration."""

    session_id: str = Field(..., description="Unique conversation/session ID.")
    transcript: Transcript = Field(..., description="Latest transcript payload.")
    intent_result: IntentResult = Field(..., description="Intent and route result.")
    selected_route: RouteType | None = Field(None, description="Resolved route.")
    llm_request: "LLMRequest | None" = Field(None, description="Prepared LLM input.")
    llm_batch_response: "LLMBatchResponse | None" = Field(
        None, description="Collected multi-LLM output."
    )
    tts_stream_ready: bool = Field(False, description="Whether TTS stream is ready.")


class LLMRequest(BaseModel):
    """Defines the request contract passed to LLM providers."""

    session_id: str = Field(..., description="Unique conversation/session ID.")
    prompt: str = Field(..., description="Prompt for generation.")
    system_prompt: str | None = Field(None, description="Optional system prompt.")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature.")
    max_tokens: int = Field(1024, gt=0, description="Token cap.")


class LLMResponse(BaseModel):
    """Defines a normalized response from one LLM provider."""

    session_id: str = Field(..., description="Unique conversation/session ID.")
    provider: str = Field(..., description="Provider identifier.")
    text: str = Field(..., description="Generated text.")
    latency_ms: int = Field(..., ge=0, description="Provider latency in ms.")
    finish_reason: str | None = Field(None, description="Model finish reason.")


class LLMBatchResponse(BaseModel):
    """Aggregates responses from multiple LLM providers."""

    session_id: str = Field(..., description="Unique conversation/session ID.")
    responses: list[LLMResponse] = Field(
        default_factory=list, description="Provider responses."
    )


class TTSChunk(BaseModel):
    """Represents a streamed TTS audio chunk."""

    session_id: str = Field(..., description="Unique conversation/session ID.")
    chunk_id: int = Field(..., ge=0, description="Monotonic TTS chunk index.")
    audio_bytes: bytes = Field(..., description="Encoded audio bytes.")
    is_last: bool = Field(..., description="Whether this is the final chunk.")


class EvalInput(BaseModel):
    """Defines evaluation input for quality assessment modules."""

    session_id: str = Field(..., description="Unique conversation/session ID.")
    user_query: str = Field(..., description="Original user query.")
    response_text: str = Field(..., description="System response text.")
    rag_context: RagContext | None = Field(None, description="Optional RAG context.")
    references: list[str] = Field(
        default_factory=list, description="Reference answers/docs."
    )


class EvalResult(BaseModel):
    """Defines evaluation output for one assessment run."""

    session_id: str = Field(..., description="Unique conversation/session ID.")
    ragas_score: float | None = Field(None, ge=0.0, le=1.0, description="RAGAS score.")
    judge_score: float | None = Field(
        None, ge=0.0, le=1.0, description="LLM-as-a-Judge score."
    )
    verdict: str = Field(..., description="Human-readable verdict label.")
    details: dict[str, Any] = Field(default_factory=dict, description="Extra metadata.")


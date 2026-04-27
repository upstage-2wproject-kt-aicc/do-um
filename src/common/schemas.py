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


class ClientInfo(BaseModel):
    """Carries caller identity and device metadata from an inbound call."""

    caller_number: str = Field(..., description="Caller's phone number.")
    device_type: str | None = Field(None, description="Device type (mobile, landline, etc.).")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional caller/device info.")


class ProviderErrorCode(str, Enum):
    """Defines standardized provider failure codes."""

    MISSING_API_KEY = "MISSING_API_KEY"
    MODEL_NOT_CONFIGURED = "MODEL_NOT_CONFIGURED"
    RESP_DELAY = "RESP_DELAY"
    PROVIDER_EXCEPTION = "PROVIDER_EXCEPTION"
    UNGROUNDED_RESPONSE = "UNGROUNDED_RESPONSE"
    PROVIDER_DISABLED = "PROVIDER_DISABLED"
    NETWORK_ERROR = "NETWORK_ERROR"
    AUTH_FAILED = "AUTH_FAILED"
    RATE_LIMIT = "RATE_LIMIT"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"


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
    text: str = Field(..., description="Recognized text (user_query for downstream modules).")
    language: str = Field(..., description="Language code.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="ASR confidence.")
    is_final: bool = Field(..., description="Whether transcript is finalized.")
    timestamp_ms: int = Field(..., ge=0, description="Transcription completion timestamp in ms.")


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


class RoutingInfo(BaseModel):
    """Defines routing metadata from the NLU stage."""

    intent: str = Field(..., description="Intent label from NLU.")
    subdomain: str = Field(..., description="Subdomain label from NLU.")
    router_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Router confidence from NLU."
    )
    domain: str = Field(..., description="Top-level domain label.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Raw NLU metadata used for routing decisions."
    )


class ChatTurn(BaseModel):
    """Represents one conversation turn for context."""

    role: str = Field(..., description="Speaker role (user/assistant/system).")
    text: str = Field(..., description="Turn text.")
    timestamp: str = Field(..., description="ISO timestamp string.")


class InternalContextItem(BaseModel):
    """Represents one internal API or DB context item."""

    source: str = Field(..., description="Context source name.")
    content: str = Field(..., description="Normalized context content.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra context info.")


class PolicyRule(BaseModel):
    """Represents one policy rule item for prompt constraints."""

    rule_id: str = Field(..., description="Policy rule identifier.")
    title: str = Field(..., description="Policy title.")
    description: str = Field(..., description="Policy description text.")


class WorkflowRoutingInput(BaseModel):
    """Defines JSON input contract for workflow routing entry."""

    session_id: str = Field(..., description="Unique conversation/session ID.")
    original_query: str = Field(..., description="Original user query from NLU output.")
    routing_info: RoutingInfo = Field(..., description="Routing metadata from NLU.")
    chat_history: list[ChatTurn] = Field(
        default_factory=list, description="Conversation history for context."
    )
    internal_context: list[InternalContextItem] = Field(
        default_factory=list, description="Internal API/DB query results."
    )
    policy_rules: list[PolicyRule] = Field(
        default_factory=list, description="Policy rules to enforce in generation."
    )


class WorkflowRoutingResult(BaseModel):
    """Defines routing result before LLM fan-out begins."""

    session_id: str = Field(..., description="Unique conversation/session ID.")
    selected_route: RouteType = Field(..., description="Resolved workflow route.")
    original_query: str = Field(..., description="Original user query.")


class ProviderResult(BaseModel):
    """Defines one normalized provider result row."""

    provider: str = Field(..., description="Provider name.")
    model: str = Field(..., description="Model name.")
    answer: str = Field(..., description="Generated answer text.")
    ttft_ms: int = Field(0, ge=0, description="Time to first token in ms.")
    latency_ms: int = Field(0, ge=0, description="Total response latency in ms.")
    grounded: bool = Field(False, description="Whether answer is evidence-grounded.")
    citations: list[str] = Field(default_factory=list, description="Evidence link list.")
    error: ProviderErrorCode | None = Field(None, description="Standard error code.")
    token_usage: dict[str, int] = Field(
        default_factory=dict, description="Token usage by type."
    )


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
    route: RouteType | None = Field(None, description="Route selected by workflow.")


class LLMResponse(BaseModel):
    """Defines a normalized response from one LLM provider."""

    session_id: str = Field(..., description="Unique conversation/session ID.")
    provider: str = Field(..., description="Provider identifier.")
    text: str = Field(..., description="Generated text.")
    ttft_ms: int = Field(0, ge=0, description="Time to first token in ms.")
    latency_ms: int = Field(..., ge=0, description="Provider latency in ms.")
    finish_reason: str | None = Field(None, description="Model finish reason.")
    grounded: bool = Field(False, description="Whether answer is grounded by evidence.")
    citations: list[str] = Field(default_factory=list, description="Evidence citations.")
    error: str | None = Field(None, description="Provider error code or message.")
    token_usage: dict[str, int] = Field(
        default_factory=dict, description="Token usage map."
    )


class LLMBatchResponse(BaseModel):
    """Aggregates responses from multiple LLM providers."""

    session_id: str = Field(..., description="Unique conversation/session ID.")
    responses: list[LLMResponse] = Field(
        default_factory=list, description="Provider responses."
    )


class NLUEvidence(BaseModel):
    """Captures NLU evidence that influenced workflow answer generation."""

    intent: str = Field(..., description="NLU predicted intent label.")
    domain: str = Field(..., description="NLU predicted domain label.")
    subdomain: str = Field(..., description="NLU predicted subdomain label.")
    router_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="NLU router confidence score."
    )
    selected_route: RouteType = Field(..., description="Final workflow route.")
    route_reason: str = Field(..., description="Human-readable route decision reason.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Raw NLU metadata passed from upstream."
    )


class WorkflowOutput(BaseModel):
    """Defines workflow output contract passed to the next stage."""

    session_id: str = Field(..., description="Unique conversation/session ID.")
    results: list[ProviderResult] = Field(
        default_factory=list, description="Provider result rows."
    )
    final_answer_text: str = Field("", description="Selected final answer text.")
    pre_tts_text: str = Field("", description="Exact text that should be used before TTS.")
    is_handoff_decided: bool = Field(False, description="Whether handoff is decided.")
    reference_links: list[str] = Field(
        default_factory=list, description="Reference links for final answer."
    )
    llm_token_usage: dict[str, int] = Field(
        default_factory=dict, description="Aggregated token usage."
    )
    nlu_evidence: NLUEvidence | None = Field(
        None, description="NLU evidence used for routing and answer generation."
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

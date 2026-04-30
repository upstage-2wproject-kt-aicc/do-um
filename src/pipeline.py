"""전체 모듈을 관통하는 End-to-end 비동기 파이프라인 규약 모듈입니다."""

import os
import re
from typing import AsyncIterator
from common.logger import get_logger
from common.exceptions import TTSException
from common.schemas import (
    AudioChunk,
    EvalResult,
    LLMResponse,
    TTSChunk,
    WorkflowOutput,
    WorkflowRoutingInput,
)
from tts.factory import TTSFactory
from workflow.graph import execute_workflow_item

logger = get_logger()
KOREAN_CHAR_PATTERN = re.compile(r"[가-힣]")

class VoiceAIPipeline:
    """최상위 비동기 파이프라인 인터페이스를 정의합니다."""

    def __init__(self):
        self.tts_providers = self._build_tts_provider_chain()

    def _build_tts_provider_chain(self) -> list[str]:
        """Builds ordered TTS provider chain from environment settings."""
        primary = os.getenv("TTS_PROVIDER", "openai").strip().lower()
        fallback = os.getenv("TTS_FALLBACK_PROVIDERS", "google,azure").strip().lower()
        ordered = [primary] + [item.strip() for item in fallback.split(",") if item.strip()]
        unique: list[str] = []
        for provider in ordered:
            if provider not in unique:
                unique.append(provider)
        return unique

    async def run_workflow_to_tts(self, payload: WorkflowRoutingInput) -> AsyncIterator[TTSChunk]:
        """
        워크플로우를 실행하고 그 결과를 TTS로 합성하여 스트리밍합니다.
        """
        logger.info("Workflow 실행 시작: Session ID = {}", payload.session_id)
        
        # 1. Workflow 실행
        workflow_output = await execute_workflow_item(payload)
        
        final_text = workflow_output.final_answer_text
        if not final_text or workflow_output.is_handoff_decided:
            logger.info("워크플로우 결과가 없거나 상담원 연결이 결정되었습니다. (TTS 생략)")
            return
        if not KOREAN_CHAR_PATTERN.search(final_text):
            logger.warning("워크플로우 응답이 비한국어로 감지되어 한국어 대체 문구로 변환합니다.")
            final_text = "죄송합니다. 현재 상담사 연결을 도와드리겠습니다."
            
        logger.info("TTS 음성 합성 시작. 텍스트 길이: {}", len(final_text))
            
        # 2. Workflow 출력을 TTS 입력(LLMResponse)으로 매핑 (Adapter)
        tts_input = LLMResponse(
            session_id=workflow_output.session_id,
            provider="workflow",
            text=final_text,
            latency_ms=0,
            ttft_ms=0,
            finish_reason="stop",
            grounded=False,
            error=None
        )
        
        # 3. TTS 스트리밍 반환 (provider fallback)
        last_error: Exception | None = None
        for provider in self.tts_providers:
            service = TTSFactory.get_service(provider)
            try:
                async for chunk in service.stream(tts_input):
                    yield chunk
                logger.info("TTS 완료. provider={}", provider)
                return
            except TTSException as exc:
                last_error = exc
                logger.warning("TTS 실패. provider={} error={}", provider, str(exc))
        if last_error is not None:
            raise last_error

    async def run(self, audio_chunk: AudioChunk) -> EvalResult:
        """STT, NLU, 워크플로우, 다중 LLM, TTS 및 평가 단계를 차례로 실행합니다."""
        raise NotImplementedError

    async def stream_tts_for_workflow_output(
        self, workflow_output: WorkflowOutput
    ) -> AsyncIterator[TTSChunk]:
        """Runs only TTS stage for an already computed workflow output."""
        final_text = workflow_output.pre_tts_text or workflow_output.final_answer_text
        if not final_text:
            return
        if not KOREAN_CHAR_PATTERN.search(final_text):
            final_text = "죄송합니다. 현재 상담사 연결을 도와드리겠습니다."

        tts_input = LLMResponse(
            session_id=workflow_output.session_id,
            provider="workflow",
            text=final_text,
            latency_ms=0,
            ttft_ms=0,
            finish_reason="stop",
            grounded=False,
            error=None,
        )

        last_error: Exception | None = None
        for provider in self.tts_providers:
            service = TTSFactory.get_service(provider)
            try:
                async for chunk in service.stream(tts_input):
                    yield chunk
                logger.info("TTS 완료. provider={}", provider)
                return
            except TTSException as exc:
                last_error = exc
                logger.warning("TTS 실패. provider={} error={}", provider, str(exc))
        if last_error is not None:
            raise last_error

    async def stream_tts_for_text(
        self,
        *,
        session_id: str,
        text: str,
    ) -> AsyncIterator[TTSChunk]:
        final_text = (text or "").strip()
        if not final_text:
            return

        tts_input = LLMResponse(
            session_id=session_id,
            provider="workflow",
            text=final_text,
            latency_ms=0,
            ttft_ms=0,
            finish_reason="stop",
            grounded=False,
            error=None,
        )

        last_error: Exception | None = None
        for provider in self.tts_providers:
            service = TTSFactory.get_service(provider)
            try:
                async for chunk in service.stream(tts_input):
                    yield chunk
                logger.info("TTS 완료. provider={}", provider)
                return
            except TTSException as exc:
                last_error = exc
                logger.warning("TTS 실패. provider={} error={}", provider, str(exc))
        if last_error is not None:
            raise last_error

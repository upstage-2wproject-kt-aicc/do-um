"""전체 모듈을 관통하는 End-to-end 비동기 파이프라인 규약 모듈입니다."""

from typing import AsyncIterator
from common.logger import get_logger
from common.schemas import AudioChunk, EvalResult, WorkflowRoutingInput, LLMResponse, TTSChunk
from tts.factory import TTSFactory
from workflow.graph import execute_workflow_item

logger = get_logger(__name__)

class VoiceAIPipeline:
    """최상위 비동기 파이프라인 인터페이스를 정의합니다."""

    def __init__(self):
        # 팩토리를 통해 TTS 서비스를 동적으로 주입합니다.
        self.tts_service = TTSFactory.get_service("azure")

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
            
        logger.info("TTS 음성 합성 시작. 텍스트 길이: {}", len(final_text))
            
        # 2. Workflow 출력을 TTS 입력(LLMResponse)으로 매핑 (Adapter)
        tts_input = LLMResponse(
            session_id=workflow_output.session_id,
            provider="workflow",
            text=final_text,
            latency_ms=0
        )
        
        # 3. TTS 스트리밍 반환
        async for chunk in self.tts_service.stream(tts_input):
            yield chunk

    async def run(self, audio_chunk: AudioChunk) -> EvalResult:
        """STT, NLU, 워크플로우, 다중 LLM, TTS 및 평가 단계를 차례로 실행합니다."""
        raise NotImplementedError


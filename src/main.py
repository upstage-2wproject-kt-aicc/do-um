from fastapi import FastAPI
from common.error_handlers import register_error_handlers
from middleware.logging import LoggingMiddleware
from tts.factory import TTSFactory
from pipeline import VoiceAIPipeline
from common.schemas import LLMResponse, WorkflowRoutingInput
from common.exceptions import ValidationException
from fastapi.responses import StreamingResponse
from llm.mock import mock_llm_stream, sentence_chunker

app = FastAPI(title="금융 챗봇 TTS API")

# 1. 미들웨어 등록 (요청 로깅 및 지연 시간 측정)
app.add_middleware(LoggingMiddleware)

# 2. 전역 에러 처리기 등록 (일관된 에러 응답 보장)
register_error_handlers(app)

@app.get("/")
async def root():
    """서버 상태 확인용 엔드포인트"""
    return {"message": "금융 챗봇 TTS API가 정상 동작 중입니다."}

@app.get("/tts/test")
async def tts_test(text: str = None):
    """
    에러 핸들링 테스트를 위한 엔드포인트입니다.
    텍스트가 없으면 ValidationException을, TTS 호출 실패 시 TTSException을 발생시킵니다.
    """
    # 1. 유효성 검사 에러 테스트
    if not text:
        raise ValidationException(
            message="합성할 텍스트가 누락되었습니다.",
            detail="쿼리 파라미터 'text'를 입력해주세요."
        )
    
    # 2. TTS 서비스 호출 (에러 발생 시 전역 처리기가 자동으로 낚아챔)
    service = TTSFactory.get_service()
    resp = LLMResponse(session_id="api_test", provider=service.__class__.__name__, text=text, latency_ms=0)
    
    audio_chunks = []
    async for chunk in service.stream(resp):
        audio_chunks.append(chunk)
        
    return {
        "success": True, 
        "audio_size": sum(len(c.audio_bytes) for c in audio_chunks)
    }

@app.get("/tts/stream")
async def tts_stream(text: str = None):
    """
    LLM의 가상 스트림을 문장 단위로 버퍼링하고, 
    각 문장이 완성될 때마다 TTS로 합성하여 오디오 바이트를 스트리밍합니다.
    """
    async def audio_generator():
        # 1. LLM 토큰 스트림 생성
        llm_stream = mock_llm_stream(text)
        
        # 2. 문장 단위 Chunker 연결
        sentences = sentence_chunker(llm_stream)
        
        # 3. TTS 서비스 초기화 (설정된 기본값 사용)
        tts_service = TTSFactory.get_service()
        
        chunk_index = 0
        async for sentence in sentences:
            # 임시 LLMResponse 객체 생성
            resp = LLMResponse(
                session_id=f"stream_test_{chunk_index}", 
                provider=tts_service.__class__.__name__, 
                text=sentence, 
                latency_ms=0
            )
            
            # 4. 문장에 대해 음성 합성 실행 및 오디오 바이트 스트리밍
            async for audio_chunk in tts_service.stream(resp):
                # 클라이언트에 오디오 바이트 청크를 실시간으로 전송
                yield audio_chunk.audio_bytes
                
            chunk_index += 1

    # Chunked Transfer Encoding으로 클라이언트에 전송
    return StreamingResponse(audio_generator(), media_type="audio/mpeg")

@app.post("/pipeline/stream")
async def pipeline_stream(payload: WorkflowRoutingInput, provider: str = None):
    """
    NLU의 라우팅 정보(JSON)를 받아 워크플로우를 실행하고 그 결과를 오디오 스트림으로 반환합니다.
    - provider: TTS 공급자 ('openai', 'azure', 'naver', 'google'). 생략 시 환경 변수의 기본값 사용.
    """
    pipeline = VoiceAIPipeline(tts_provider=provider)
    
    async def audio_generator():
        async for chunk in pipeline.run_workflow_to_tts(payload):
            yield chunk.audio_bytes
            
    return StreamingResponse(audio_generator(), media_type="audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    # uvicorn src.main:app --reload 명령어로 실행 가능합니다.
    uvicorn.run(app, host="0.0.0.0", port=8000)

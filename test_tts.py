import asyncio
from src.tts.service import TTSService, AzureTTSService
from src.common.schemas import LLMResponse
from src.common.logger import get_logger

logger = get_logger()

class ChatbotManager:
    """
    DI(의존성 주입)가 적용된 예시 클래스입니다.
    TTSService 인터페이스에만 의존하며, 구체적인 구현체(Azure, Google 등)를 외부에서 주입받습니다.
    """
    def __init__(self, tts_service: TTSService):
        self.tts_service = tts_service

    async def speak(self, text: str):
        logger.info("ChatbotManager: 텍스트 음성 변환 요청 시작")
        resp = LLMResponse(session_id="test_session", provider="test", text=text, latency_ms=0)
        
        chunks = []
        async for chunk in self.tts_service.stream(resp):
            chunks.append(chunk)
            logger.debug("청크 수신: {} bytes, is_last={}", len(chunk.audio_bytes), chunk.is_last)

        if len(chunks) > 0 and len(chunks[0].audio_bytes) > 0:
            logger.success("오디오 스트림 생성 성공! 총 {}개의 청크를 받았습니다.", len(chunks))
            with open("test_audio.mp3", "wb") as f:
                f.write(chunks[0].audio_bytes)
            logger.info("파일 저장 완료: test_audio.mp3")
        else:
            logger.warning("Fallback 오디오(또는 빈 오디오)가 반환되었습니다.")

async def main():
    logger.info("=== TTS 파이프라인 리팩토링 검증 ===")
    
    # 1. 의존성 객체 생성 (실무에서는 DI Container나 팩토리 사용)
    azure_tts = AzureTTSService()
    
    # 2. 의존성 주입 (Dependency Injection)
    # ChatbotManager는 주입된 객체가 Azure인지 Google인지 모른 채 인터페이스만 보고 동작합니다.
    manager = ChatbotManager(tts_service=azure_tts)
    
    # 3. 서비스 실행
    text = "테스트입니다. 결제 금액은 ₩12,500 이며, 결제일은 2026-04-23 입니다."
    await manager.speak(text)
    
    logger.info("=== 검증 종료 ===")

if __name__ == "__main__":
    asyncio.run(main())

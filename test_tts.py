import asyncio
import time
from src.tts.service import TTSService, AzureTTSService, NaverTTSService, GoogleTTSService
from src.common.schemas import LLMResponse
from src.common.logger import get_logger

logger = get_logger()

async def compare_tts_models(text: str):
    logger.info("=== TTS 모델 성능 비교 시작 ===")
    
    # 1. 테스트할 모델 리스트 준비 (DI 패턴 활용)
    services = {
        "Azure": AzureTTSService(),
        "Naver": NaverTTSService(),
        "Google": GoogleTTSService()
    }
    
    results = []
    
    for name, service in services.items():
        logger.info(f"[{name}] 테스트 시작...")
        resp = LLMResponse(session_id="compare_session", provider="test", text=text, latency_ms=0)
        
        start_time = time.perf_counter()
        chunks = []
        try:
            async for chunk in service.stream(resp):
                chunks.append(chunk)
            
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000 # ms
            
            audio_size = sum(len(c.audio_bytes) for c in chunks)
            
            results.append({
                "model": name,
                "latency_ms": latency,
                "audio_size": audio_size,
                "success": audio_size > 0
            })
            
            if audio_size > 0:
                logger.success(f"[{name}] 완료 - Latency: {latency:.2f}ms, Size: {audio_size} bytes")
                # 파일 저장 (확인용)
                with open(f"test_audio_{name.lower()}.mp3", "wb") as f:
                    f.write(chunks[0].audio_bytes)
            else:
                logger.warning(f"[{name}] 응답이 비어있습니다. (자격 증명 확인 필요)")
                
        except Exception as e:
            logger.error(f"[{name}] 오류 발생: {e}")
            results.append({"model": name, "success": False, "error": str(e)})

    # 2. 결과 요약 출력
    logger.info("=== 비교 결과 요약 ===")
    for res in results:
        status = "✅ 성공" if res.get("success") else "❌ 실패"
        l_str = f"{res['latency_ms']:.2f}ms" if "latency_ms" in res else "N/A"
        logger.info(f"{res['model']}: {status} | Latency: {l_str}")

async def main():
    test_text = "안녕하세요. 금융 챗봇 TTS 성능 비교 테스트를 위한 문장입니다."
    await compare_tts_models(test_text)

if __name__ == "__main__":
    asyncio.run(main())

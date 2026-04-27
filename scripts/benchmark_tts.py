import os
import sys
import asyncio
import time
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# src 경로 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from tts.factory import TTSFactory
from common.schemas import LLMResponse
from common.logger import get_logger

logger = get_logger()

async def benchmark_service(provider: str, text: str, iterations: int = 1):
    logger.info(f"--- Benchmark: {provider} (Iterations: {iterations}) ---")
    
    try:
        service = TTSFactory.get_service(provider)
    except Exception as e:
        logger.error(f"Failed to get service for {provider}: {e}")
        return

    total_ttfa = 0
    total_ttl = 0
    success_count = 0

    for i in range(iterations):
        resp = LLMResponse(
            session_id=f"bench_{provider}_{i}", 
            provider="test", 
            text=text, 
            latency_ms=0
        )
        
        start_time = time.perf_counter()
        first_chunk_time = None
        chunk_count = 0
        
        try:
            async for chunk in service.stream(resp):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                chunk_count += 1
            
            end_time = time.perf_counter()
            
            if first_chunk_time:
                ttfa = (first_chunk_time - start_time) * 1000
                ttl = (end_time - start_time) * 1000
                total_ttfa += ttfa
                total_ttl += ttl
                success_count += 1
                logger.info(f"Iteration {i+1}: TTFA={ttfa:.2f}ms, TTL={ttl:.2f}ms, Chunks={chunk_count}")
            else:
                logger.warning(f"Iteration {i+1}: No chunks received.")
                
        except Exception as e:
            logger.error(f"Iteration {i+1} failed: {e}")

    if success_count > 0:
        avg_ttfa = total_ttfa / success_count
        avg_ttl = total_ttl / success_count
        logger.success(f"[{provider}] Average - TTFA: {avg_ttfa:.2f}ms, TTL: {avg_ttl:.2f}ms")
    else:
        logger.error(f"[{provider}] No successful iterations.")

async def main():
    test_text = "금융 인공지능 비서입니다. 무엇을 도와드릴까요? 실시간 스트리밍 테스트를 진행 중입니다."
    
    # 설정에 등록된 공급자 테스트
    providers = ["openai", "google"] 
    
    for p in providers:
        await benchmark_service(p, test_text, iterations=3)

if __name__ == "__main__":
    asyncio.run(main())

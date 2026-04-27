from abc import ABC, abstractmethod
from typing import AsyncIterator
import time

from common.schemas import LLMResponse, TTSChunk
from common.logger import get_logger, TTAMetrics

logger = get_logger()


class BaseTTSService(ABC):
    """TTS 서비스의 규격(Contract)을 정의하는 추상 기본 클래스입니다."""

    async def stream(self, response: LLMResponse) -> AsyncIterator[TTSChunk]:
        """
        성능 측정(TTFA/TTL)이 내장된 공통 스트리밍 진입점입니다.
        실제 구현은 _stream_impl에서 이루어집니다.
        """
        metrics = TTAMetrics(response.session_id)
        metrics.start()
        
        is_first_chunk = True
        provider_name = self.__class__.__name__

        try:
            async for chunk in self._stream_impl(response):
                if is_first_chunk:
                    metrics.record_first_chunk()
                    is_first_chunk = False
                
                yield chunk
            
            metrics.record_last_chunk()
            metrics.log_results(provider_name)
            
        except Exception as e:
            logger.error(f"[{provider_name}] Streaming failed: {e}")
            raise

    @abstractmethod
    async def _stream_impl(self, response: LLMResponse) -> AsyncIterator[TTSChunk]:
        """자식 클래스에서 구현해야 할 실제 스트리밍 로직입니다."""
        raise NotImplementedError

    def _normalize_financial_text(self, text: str) -> str:
        """TTS 발음을 개선하기 위해 금융 텍스트를 정규화합니다."""
        if not text:
            return ""
        normalized = text.replace("₩", "원").replace("$", "달러")
        # 추가적인 정규화 로직 (예: 숫자 읽기 방식 등)을 여기에 확장합니다.
        return normalized


class GenericHTTPTTSService(BaseTTSService):
    """
    HTTP REST API를 사용하는 TTS 엔진들을 위한 제네릭 기반 클래스입니다.
    """
    def __init__(self, api_url: str, headers: dict):
        self.api_url = api_url
        self.headers = headers

    @abstractmethod
    def _build_payload(self, text: str) -> dict:
        """API 요청을 위한 페이로드를 생성합니다. 자식 클래스에서 구현합니다."""
        raise NotImplementedError

    async def _stream_impl(self, response: LLMResponse) -> AsyncIterator[TTSChunk]:
        import aiohttp
        
        normalized_text = self._normalize_financial_text(response.text)
        payload = self._build_payload(normalized_text)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=self.headers, json=payload) as api_res:
                if api_res.status != 200:
                    error_text = await api_res.text()
                    raise Exception(f"HTTP Error {api_res.status}: {error_text}")
                
                chunk_index = 0
                # 8KB 단위로 스트리밍 읽기 (필요시 조정 가능)
                async for chunk_bytes in api_res.content.iter_chunked(8192):
                    if chunk_bytes:
                        yield TTSChunk(
                            session_id=response.session_id,
                            chunk_id=chunk_index,
                            audio_bytes=chunk_bytes,
                            is_last=False
                        )
                        chunk_index += 1
                
                # 마지막 청크 표시
                yield TTSChunk(
                    session_id=response.session_id,
                    chunk_id=chunk_index,
                    audio_bytes=b"",
                    is_last=True
                )

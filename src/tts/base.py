"""TTS 서비스를 위한 기본 인터페이스 모듈입니다."""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from common.schemas import LLMResponse, TTSChunk


class BaseTTSService(ABC):
    """TTS 서비스의 규격(Contract)을 정의하는 추상 기본 클래스입니다."""

    @abstractmethod
    async def stream(self, response: LLMResponse) -> AsyncIterator[TTSChunk]:
        """
        LLMResponse의 텍스트를 오디오 청크의 비동기 스트림으로 변환합니다.
        
        Args:
            response (LLMResponse): LLM에서 생성된 텍스트가 포함된 응답 객체.
            
        Yields:
            TTSChunk: 합성된 오디오 청크 데이터.
        """
        raise NotImplementedError

    def _normalize_financial_text(self, text: str) -> str:
        """
        TTS 발음을 개선하기 위해 금융 텍스트를 정규화하는 공용 유틸리티입니다.
        필요에 따라 각 벤더별 클래스에서 오버라이드할 수 있습니다.
        """
        normalized = text.replace("₩", "원").replace("$", "달러")
        # 여기에 추가적인 정규화 로직을 구현합니다 (예: 숫자에 대한 정규식 처리)
        return normalized

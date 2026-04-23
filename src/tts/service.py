"""TTS 스트리밍 출력을 위한 인터페이스 및 구현체 모듈입니다."""

import os
import re
import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator

import azure.cognitiveservices.speech as speechsdk
from cachetools import LRUCache

from src.common.schemas import LLMResponse, TTSChunk
from src.common.logger import get_logger

logger = get_logger()

class TTSService(ABC):
    """
    TTS 합성을 위한 추상 기본 클래스 (인터페이스)입니다.
    외부 호출자는 이 인터페이스만 의존하게 되어 특정 TTS 벤더(Azure 등)에 종속되지 않습니다.
    """

    @abstractmethod
    async def stream(self, response: LLMResponse) -> AsyncIterator[TTSChunk]:
        """
        LLM의 텍스트 응답을 TTS 청크(Chunk) 스트림으로 변환합니다.
        각 구현체(Azure, Google 등)는 이 메서드를 반드시 재정의해야 합니다.
        """
        pass


class AzureTTSService(TTSService):
    """
    Azure TTS를 활용한 음성 합성 서비스 구현체입니다.
    내부적으로 캐싱(Caching), 정규화(Normalization), 예외 처리(Fallback)를 지원합니다.
    """

    def __init__(self, cache_size: int = 100):
        # Azure SDK 설정 초기화
        self.speech_key = os.environ.get("AZURE_SPEECH_KEY", "")
        self.speech_region = os.environ.get("AZURE_SPEECH_REGION", "")
        
        if self.speech_key and self.speech_region:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key, 
                region=self.speech_region
            )
            # 챗봇 스트리밍에 적합한 빠르고 가벼운 모노 포맷 지정
            self.speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
            )
            logger.info("Azure TTS 설정이 성공적으로 로드되었습니다. [Region: {}]", self.speech_region)
        else:
            self.speech_config = None
            logger.warning("Azure TTS 환경 변수(KEY 또는 REGION)가 누락되었습니다. Fallback 모드로 동작합니다.")
            
        # 생성된 오디오 청크를 저장하기 위한 인메모리 LRU 캐시
        self.audio_cache = LRUCache(maxsize=cache_size)
        self.voice_name = "ko-KR-SunHiNeural"
        
        # 장애 시 반환할 예비 오디오 바이트 (실제 상용 환경에서는 안내 mp3 파일을 로드하여 사용)
        self.fallback_audio_bytes = b"" 

    def _normalize_financial_text(self, text: str) -> str:
        """
        한국어 TTS 발음을 개선하기 위해 금액과 날짜를 한글 텍스트로 정규화합니다.
        예: ₩12,500 -> 12500원
        """
        if not text:
            return ""
        # 통화 기호 변환
        text = re.sub(r'₩\s*([0-9,]+)', lambda m: m.group(1).replace(',', '') + '원', text)
        # 날짜 포맷 변환 (YYYY-MM-DD -> YYYY년 MM월 DD일)
        text = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\1년 \2월 \3일', text)
        return text

    def _build_ssml(self, text: str) -> str:
        """Azure TTS API에 전달할 SSML(음성 마크업) 문자열을 생성합니다."""
        return f"""
        <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='ko-KR'>
            <voice name='{self.voice_name}'>
                {text}
            </voice>
        </speak>
        """

    async def stream(self, response: LLMResponse) -> AsyncIterator[TTSChunk]:
        """LLM 텍스트를 기반으로 TTS 오디오 스트림을 생성 및 반환합니다."""
        raw_text = response.text
        if not raw_text or not raw_text.strip():
            logger.debug("TTS 입력 텍스트가 비어있어 스트림을 종료합니다. [SessionID: {}]", response.session_id)
            return
            
        # 1. 텍스트 정규화 (발음 교정)
        normalized_text = self._normalize_financial_text(raw_text)
        
        # 2. 캐시 조회 (중복 합성 방지)
        cache_key = f"{self.voice_name}:{normalized_text}"
        if cache_key in self.audio_cache:
            logger.info("캐시된 오디오를 반환합니다. [Key: {}]", cache_key)
            yield TTSChunk(
                session_id=response.session_id,
                chunk_id=0,
                audio_bytes=self.audio_cache[cache_key],
                is_last=True
            )
            return

        # 3. 환경 변수 누락에 따른 Fallback 처리
        if not self.speech_config:
            logger.error("Azure TTS 자격 증명이 없습니다. Fallback 오디오를 반환합니다. [SessionID: {}]", response.session_id)
            yield TTSChunk(
                session_id=response.session_id,
                chunk_id=0,
                audio_bytes=self.fallback_audio_bytes,
                is_last=True
            )
            return

        # 4. SSML 생성 및 API 호출
        ssml = self._build_ssml(normalized_text)
        
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config, 
            audio_config=None  # 스피커 출력이 아닌 메모리로 결과를 받음
        )
        
        def _synthesize():
            return synthesizer.speak_ssml(ssml)

        # 메인 이벤트 루프 차단 방지를 위해 별도 스레드에서 동기 API 호출
        logger.debug("Azure TTS 합성을 시작합니다. [Text: {}]", normalized_text[:20] + "...")
        result = await asyncio.to_thread(_synthesize)

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_data = result.audio_data
            logger.info("TTS 합성이 완료되었습니다. [Audio Size: {} bytes, SessionID: {}]", len(audio_data), response.session_id)
            
            # 생성된 결과 캐시 저장
            self.audio_cache[cache_key] = audio_data
            
            yield TTSChunk(
                session_id=response.session_id,
                chunk_id=0,
                audio_bytes=audio_data,
                is_last=True
            )
        else:
            # 5. API 오류 발생 시 Fallback 처리
            if result.cancellation_details:
                logger.error(
                    "Azure TTS 합성 실패 [Reason: {}, Details: {}]", 
                    result.reason, 
                    result.cancellation_details.error_details
                )
            else:
                logger.error("Azure TTS 합성 실패 [Reason: {}]", result.reason)
            
            yield TTSChunk(
                session_id=response.session_id,
                chunk_id=0,
                audio_bytes=self.fallback_audio_bytes,
                is_last=True
            )


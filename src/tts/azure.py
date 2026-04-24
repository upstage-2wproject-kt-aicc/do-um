"""Azure TTS 서비스 구현체 모듈입니다."""

import asyncio
from typing import AsyncIterator
from cachetools import LRUCache

from common.logger import get_logger
from common.exceptions import TTSException
from common.schemas import LLMResponse, TTSChunk
from common.config import AzureTTSConfig
from tts.base import BaseTTSService

logger = get_logger()


class AzureTTSService(BaseTTSService):
    """
    캐싱(Caching) 및 텍스트 정규화 기능을 지원하는 Azure TTS 구현체입니다.
    """

    def __init__(self, config: AzureTTSConfig, cache_size: int = 100):
        self.config = config
        self.speech_config = None
        
        if self.config.speech_key and self.config.service_region:
            try:
                import azure.cognitiveservices.speech as speechsdk
                self.speech_config = speechsdk.SpeechConfig(
                    subscription=self.config.speech_key, 
                    region=self.config.service_region
                )
                self.speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
                )
                self.speechsdk = speechsdk
                logger.info("Azure TTS 설정이 성공적으로 로드되었습니다. [Region: {}]", self.config.service_region)
            except ImportError:
                logger.warning("Azure Speech SDK가 설치되지 않았습니다.")
        else:
            logger.warning("Azure TTS 환경 변수가 누락되었습니다.")
            
        self.audio_cache = LRUCache(maxsize=cache_size)

    def _build_ssml(self, text: str) -> str:
        return f"""
        <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='ko-KR'>
            <voice name='{self.config.voice}'>
                {text}
            </voice>
        </speak>
        """

    async def stream(self, response: LLMResponse) -> AsyncIterator[TTSChunk]:
        raw_text = response.text
        if not raw_text or not raw_text.strip():
            return
            
        normalized_text = self._normalize_financial_text(raw_text)
        
        # 캐시 확인
        cache_key = f"{self.config.voice}:{normalized_text}"
        if cache_key in self.audio_cache:
            logger.info("캐시된 Azure 오디오를 반환합니다. [Key: {}]", cache_key)
            yield TTSChunk(
                session_id=response.session_id,
                chunk_id=0,
                audio_bytes=self.audio_cache[cache_key],
                is_last=True
            )
            return

        if not self.speech_config:
            raise TTSException(
                message="Azure TTS 자격 증명이 설정되지 않았습니다.",
                error_code="AZURE_CONFIG_ERROR"
            )

        ssml = self._build_ssml(normalized_text)
        synthesizer = self.speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
        
        def _synthesize():
            return synthesizer.speak_ssml(ssml)

        logger.debug("Azure TTS 합성을 시작합니다. [Text: {}]", normalized_text[:20] + "...")
        result = await asyncio.to_thread(_synthesize)

        if result.reason == self.speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_data = result.audio_data
            self.audio_cache[cache_key] = audio_data
            
            # 현재는 전체 오디오를 한 번에 반환하지만,
            # 향후 Azure의 스트림 콜백 이벤트를 받아 처리하도록 확장할 수 있습니다.
            yield TTSChunk(
                session_id=response.session_id,
                chunk_id=0,
                audio_bytes=audio_data,
                is_last=True
            )
        else:
            error_detail = "Unknown error"
            if result.cancellation_details:
                error_detail = result.cancellation_details.error_details
            
            raise TTSException(
                message="Azure 음성 합성에 실패했습니다.",
                error_code="AZURE_TTS_FAILURE",
                detail=error_detail
            )

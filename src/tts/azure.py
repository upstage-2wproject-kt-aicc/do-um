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

    async def _stream_impl(self, response: LLMResponse) -> AsyncIterator[TTSChunk]:
        raw_text = response.text
        if not raw_text or not raw_text.strip():
            return
            
        normalized_text = self._normalize_financial_text(raw_text)
        
        # 1. 캐시 확인
        cache_key = f"{self.config.voice}:{normalized_text}"
        if cache_key in self.audio_cache:
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

        # 2. Azure 스트리밍 설정
        pull_stream = self.speechsdk.audio.PullAudioOutputStream()
        audio_config = self.speechsdk.audio.AudioOutputConfig(stream=pull_stream)
        synthesizer = self.speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config, 
            audio_config=audio_config
        )
        
        ssml = self._build_ssml(normalized_text)
        
        # 합성을 비동기로 시작 (스레드 풀 활용)
        result_future = synthesizer.start_speaking_ssml_async(ssml)
        
        # 3. 데이터 읽기 및 yield 루프
        chunk_index = 0
        all_audio = bytearray()
        buffer = bytearray(8192) # 8KB 가변 버퍼 (SDK write용)
        
        try:
            while True:
                # read()는 Blocking이므로 별도 스레드에서 실행
                filled_size = await asyncio.to_thread(pull_stream.read, buffer)
                if filled_size == 0:
                    break
                
                chunk_bytes = buffer[:filled_size]
                all_audio.extend(chunk_bytes)
                
                yield TTSChunk(
                    session_id=response.session_id,
                    chunk_id=chunk_index,
                    audio_bytes=chunk_bytes,
                    is_last=False
                )
                chunk_index += 1
            
            # 스트림 종료 알림
            yield TTSChunk(
                session_id=response.session_id,
                chunk_id=chunk_index,
                audio_bytes=b"",
                is_last=True
            )
            
            # 4. 전체 데이터 캐시 저장
            self.audio_cache[cache_key] = bytes(all_audio)
            
        except Exception as e:
            raise TTSException(
                message="Azure 음성 합성 스트리밍 중 오류 발생",
                error_code="AZURE_STREAM_FAILURE",
                detail=str(e)
            )
        finally:
            # 리소스 정리
            synthesizer = None

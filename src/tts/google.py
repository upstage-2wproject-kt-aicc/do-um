"""Google TTS 서비스 구현체 모듈입니다."""

import asyncio
from typing import AsyncIterator

from common.logger import get_logger
from common.exceptions import TTSException
from common.schemas import LLMResponse, TTSChunk
from common.config import GoogleTTSConfig
from tts.base import BaseTTSService

logger = get_logger()


class GoogleTTSService(BaseTTSService):
    """
    Google Cloud TTS 구현체입니다.
    """

    def __init__(self, config: GoogleTTSConfig):
        self.config = config
        self.client = None
        self.voice = None
        self.audio_config = None
        
        try:
            from google.cloud import texttospeech
            from google.api_core.client_options import ClientOptions
            
            self.texttospeech = texttospeech
            
            # 명시된 project_id가 있으면 ClientOptions에 설정하여 주입
            client_options = None
            if self.config.project_id:
                client_options = ClientOptions(quota_project_id=self.config.project_id)
                logger.info("Google TTS 클라이언트에 Project ID({})가 명시적으로 설정되었습니다.", self.config.project_id)
                
            self.client = texttospeech.TextToSpeechClient(client_options=client_options)
            
            self.voice = texttospeech.VoiceSelectionParams(
                language_code=self.config.language_code, name=self.config.voice
            )
            self.audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            logger.info("Google TTS 클라이언트가 초기화되었습니다.")
        except ImportError:
            logger.warning("Google Cloud SDK가 설치되지 않았습니다.")
        except Exception as e:
            logger.warning("Google TTS 초기화 실패 (인증 문제일 수 있음): {}", str(e))

    async def stream(self, response: LLMResponse) -> AsyncIterator[TTSChunk]:
        if not self.client:
            raise TTSException(
                message="Google TTS 클라이언트가 활성화되지 않았습니다.",
                error_code="GOOGLE_CONFIG_ERROR"
            )

        synthesis_input = self.texttospeech.SynthesisInput(text=response.text)
        
        def _call_api():
            return self.client.synthesize_speech(
                input=synthesis_input, voice=self.voice, audio_config=self.audio_config
            )

        logger.debug("Google TTS 합성을 시작합니다. [Text: {}]", response.text[:20] + "...")
        try:
            res = await asyncio.to_thread(_call_api)
            yield TTSChunk(
                session_id=response.session_id, 
                chunk_id=0, 
                audio_bytes=res.audio_content, 
                is_last=True
            )
        except Exception as e:
            raise TTSException(
                message="Google 음성 합성에 실패했습니다.",
                error_code="GOOGLE_TTS_FAILURE",
                detail=str(e)
            )

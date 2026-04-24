"""실시간 스트리밍을 지원하는 OpenAI TTS 서비스 구현체 모듈입니다."""

import asyncio
from typing import AsyncIterator

from common.logger import get_logger
from common.exceptions import TTSException
from common.schemas import LLMResponse, TTSChunk
from common.config import OpenAITTSConfig
from tts.base import BaseTTSService

logger = get_logger()


class OpenAITTSService(BaseTTSService):
    """
    오디오 청크를 실시간으로 스트리밍하는 OpenAI TTS 구현체입니다.
    """

    def __init__(self, config: OpenAITTSConfig):
        self.config = config
        self.client = None
        
        if self.config.api_key:
            try:
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI(api_key=self.config.api_key)
                logger.info("OpenAI Async TTS 클라이언트가 초기화되었습니다.")
            except ImportError:
                logger.warning("OpenAI 패키지가 설치되지 않았습니다. (uv pip install openai)")
            except Exception as e:
                logger.warning("OpenAI TTS 초기화 실패: {}", str(e))
        else:
            logger.warning("OpenAI API 키가 설정되지 않았습니다.")

    async def stream(self, response: LLMResponse) -> AsyncIterator[TTSChunk]:
        """
        OpenAI TTS API로부터 오디오를 청크 단위로 직접 스트리밍합니다.
        """
        if not self.client:
            raise TTSException(
                message="OpenAI API 클라이언트가 활성화되지 않았습니다.",
                error_code="OPENAI_CONFIG_ERROR"
            )

        raw_text = response.text
        if not raw_text or not raw_text.strip():
            return

        normalized_text = self._normalize_financial_text(raw_text)
        logger.debug("OpenAI TTS 스트리밍을 시작합니다. [Text: {}]", normalized_text[:20] + "...")

        try:
            # OpenAI로부터 직접 비동기 스트리밍 수신
            async with self.client.audio.speech.with_streaming_response.create(
                model=self.config.model,
                voice=self.config.voice,
                input=normalized_text,
                response_format="mp3"
            ) as api_response:
                chunk_index = 0
                
                async for chunk_bytes in api_response.iter_bytes(chunk_size=4096):
                    if chunk_bytes:
                        yield TTSChunk(
                            session_id=response.session_id,
                            chunk_id=chunk_index,
                            audio_bytes=chunk_bytes,
                            is_last=False
                        )
                        chunk_index += 1
                
                # 스트림 종료를 알리기 위해 마지막 빈 청크 전송
                yield TTSChunk(
                    session_id=response.session_id,
                    chunk_id=chunk_index,
                    audio_bytes=b"",
                    is_last=True
                )
                logger.info("OpenAI TTS 스트리밍이 완료되었습니다.")
                
        except Exception as e:
            logger.exception("OpenAI 스트리밍 중 오류 발생")
            raise TTSException(
                message="OpenAI 음성 합성에 실패했습니다.",
                error_code="OPENAI_TTS_FAILURE",
                detail=str(e)
            )

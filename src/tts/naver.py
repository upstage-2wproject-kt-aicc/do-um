"""Naver TTS 서비스 구현체 모듈입니다."""

import asyncio
import requests
from typing import AsyncIterator

from common.logger import get_logger
from common.exceptions import TTSException
from common.schemas import LLMResponse, TTSChunk
from common.config import NaverTTSConfig
from tts.base import BaseTTSService

logger = get_logger()


class NaverTTSService(BaseTTSService):
    """
    Naver Clova TTS 구현체입니다.
    """

    def __init__(self, config: NaverTTSConfig):
        self.config = config

    async def stream(self, response: LLMResponse) -> AsyncIterator[TTSChunk]:
        if not self.config.client_id or not self.config.client_secret:
            raise TTSException(
                message="Naver TTS 자격 증명이 설정되지 않았습니다.",
                error_code="NAVER_CONFIG_ERROR"
            )

        headers = {
            "X-NCP-APIGW-API-KEY-ID": self.config.client_id,
            "X-NCP-APIGW-API-KEY": self.config.client_secret,
        }
        data = {
            "speaker": self.config.voice,
            "speed": "0",
            "text": response.text,
            "format": "mp3"
        }

        def _call_api():
            return requests.post(self.config.url, headers=headers, data=data)

        logger.debug("Naver TTS 합성을 시작합니다. [Text: {}]", response.text[:20] + "...")
        res = await asyncio.to_thread(_call_api)

        if res.status_code == 200:
            yield TTSChunk(
                session_id=response.session_id, 
                chunk_id=0, 
                audio_bytes=res.content, 
                is_last=True
            )
        else:
            raise TTSException(
                message="Naver 음성 합성에 실패했습니다.",
                error_code="NAVER_TTS_FAILURE",
                detail=f"Status: {res.status_code}, Response: {res.text}"
            )

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


from tts.base import GenericHTTPTTSService

class NaverTTSService(GenericHTTPTTSService):
    """
    Naver Clova TTS 구현체 (GenericHTTPTTSService 상속)
    """
    def __init__(self, config: NaverTTSConfig):
        headers = {
            "X-NCP-APIGW-API-KEY-ID": config.client_id,
            "X-NCP-APIGW-API-KEY": config.client_secret,
        }
        super().__init__(api_url=config.url, headers=headers)
        self.config = config

    def _build_payload(self, text: str) -> dict:
        return {
            "speaker": self.config.voice,
            "speed": "0",
            "text": text,
            "format": "mp3"
        }

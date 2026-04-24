"""서비스 인스턴스 생성을 위한 TTS 팩토리 모듈입니다."""

from common.config import config
from common.exceptions import TTSException
from tts.base import BaseTTSService
from tts.azure import AzureTTSService
from tts.openai import OpenAITTSService


class TTSFactory:
    """설정에 기반하여 적절한 TTS 서비스를 인스턴스화하는 팩토리 클래스입니다."""

    @staticmethod
    def get_service(provider: str | None = None) -> BaseTTSService:
        """
        요청된 공급자(Provider)에 맞는 TTS 서비스 인스턴스를 반환합니다.
        
        Args:
            provider (str | None): 공급자 이름 ('azure', 'openai', 'naver', 'google'). 지정하지 않으면 설정의 기본값을 사용합니다.
            
        Returns:
            BaseTTSService: 초기화된 TTS 서비스 객체.
        """
        provider = (provider or config.default_tts_provider).lower().strip()
        
        if provider == "azure":
            return AzureTTSService(config.azure_tts)
        elif provider == "openai":
            return OpenAITTSService(config.openai_tts)
        # 참고: Naver와 Google 서비스도 생성되었으므로 이곳에서 임포트하여 처리합니다.
        elif provider == "naver":
            from tts.naver import NaverTTSService
            return NaverTTSService(config.naver_tts)
        elif provider == "google":
            from tts.google import GoogleTTSService
            return GoogleTTSService(config.google_tts)
        else:
            raise TTSException(
                message=f"지원하지 않는 TTS 제공자입니다: {provider}",
                error_code="UNSUPPORTED_PROVIDER"
            )

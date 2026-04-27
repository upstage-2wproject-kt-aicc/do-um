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
        """
        provider = (provider or config.default_tts_provider).lower().strip()
        
        # 공급자별 생성 매핑 (필요 시 지연 임포트 활용)
        provider_map = {
            "azure": lambda: AzureTTSService(config.azure_tts),
            "openai": lambda: OpenAITTSService(config.openai_tts),
            "naver": lambda: TTSFactory._get_naver_service(),
            "google": lambda: TTSFactory._get_google_service(),
        }
        
        if provider not in provider_map:
            raise TTSException(
                message=f"지원하지 않는 TTS 제공자입니다: {provider}",
                error_code="UNSUPPORTED_PROVIDER"
            )
            
        return provider_map[provider]()

    @staticmethod
    def _get_naver_service():
        from tts.naver import NaverTTSService
        return NaverTTSService(config.naver_tts)

    @staticmethod
    def _get_google_service():
        from tts.google import GoogleTTSService
        return GoogleTTSService(config.google_tts)

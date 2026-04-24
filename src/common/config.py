"""음성 AI 애플리케이션을 위한 중앙 설정 모듈입니다."""

import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# .env 환경변수 로드
load_dotenv()


class OpenAITTSConfig(BaseModel):
    """OpenAI TTS 설정"""
    api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    voice: str = Field("alloy", description="OpenAI 음성 모델")
    model: str = Field("tts-1", description="OpenAI TTS 모델 (tts-1, tts-1-hd)")


class AzureTTSConfig(BaseModel):
    """Azure TTS 설정"""
    speech_key: str = Field(default_factory=lambda: os.environ.get("AZURE_SPEECH_KEY", ""))
    service_region: str = Field(default_factory=lambda: os.environ.get("AZURE_SERVICE_REGION", ""))
    voice: str = Field("ko-KR-SunHiNeural", description="Azure 음성 모델")


class NaverTTSConfig(BaseModel):
    """Naver Clova TTS 설정"""
    client_id: str = Field(default_factory=lambda: os.environ.get("NAVER_CLIENT_ID", ""))
    client_secret: str = Field(default_factory=lambda: os.environ.get("NAVER_CLIENT_SECRET", ""))
    voice: str = Field("nara", description="Naver 음성 모델")
    url: str = Field("https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts")


class GoogleTTSConfig(BaseModel):
    """Google Cloud TTS 설정"""
    project_id: str = Field(default_factory=lambda: os.environ.get("GOOGLE_PROJECT_ID", ""), description="Google Cloud 프로젝트 ID")
    voice: str = Field("ko-KR-Wavenet-A", description="Google 음성 모델")
    language_code: str = Field("ko-KR")


class AppConfig(BaseModel):
    """메인 애플리케이션 통합 설정"""
    default_tts_provider: str = Field(default_factory=lambda: os.environ.get("DEFAULT_TTS_PROVIDER", "openai").lower())
    openai_tts: OpenAITTSConfig = Field(default_factory=OpenAITTSConfig)
    azure_tts: AzureTTSConfig = Field(default_factory=AzureTTSConfig)
    naver_tts: NaverTTSConfig = Field(default_factory=NaverTTSConfig)
    google_tts: GoogleTTSConfig = Field(default_factory=GoogleTTSConfig)


# 전역 설정 인스턴스
config = AppConfig()

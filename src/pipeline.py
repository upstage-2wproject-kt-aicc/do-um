"""전체 모듈을 관통하는 End-to-end 비동기 파이프라인 규약 모듈입니다."""

from common.schemas import AudioChunk, EvalResult
from tts.factory import TTSFactory


class VoiceAIPipeline:
    """최상위 비동기 파이프라인 인터페이스를 정의합니다."""

    def __init__(self):
        # 팩토리를 통해 TTS 서비스를 동적으로 주입하는 예시입니다.
        self.tts_service = TTSFactory.get_service("azure")

    async def run(self, audio_chunk: AudioChunk) -> EvalResult:
        """STT, NLU, 워크플로우, 다중 LLM, TTS 및 평가 단계를 차례로 실행합니다."""
        raise NotImplementedError



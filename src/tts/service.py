"""TTS interface skeleton for streaming audio output."""

from src.common.schemas import LLMResponse, TTSChunk


class TTSService:
    """Defines TTS boundary methods for stream synthesis."""

    async def stream(self, response: LLMResponse) -> TTSChunk:
        """Converts LLM text response into a TTS chunk stream."""
        raise NotImplementedError


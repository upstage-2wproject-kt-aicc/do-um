"""End-to-end async pipeline contract across all modules."""

from src.common.schemas import AudioChunk, EvalResult


class VoiceAIPipeline:
    """Defines the top-level async pipeline interface."""

    async def run(self, audio_chunk: AudioChunk) -> EvalResult:
        """Executes STT, NLU, workflow, LLM, TTS, and evaluation stages."""
        raise NotImplementedError


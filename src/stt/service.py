"""STT interface skeleton for VAD and transcript generation."""

from common.schemas import AudioChunk, Transcript


class STTService:
    """Defines STT boundary methods for audio processing."""

    async def apply_vad(self, audio_chunk: AudioChunk) -> AudioChunk:
        """Applies voice activity detection to an audio chunk."""
        raise NotImplementedError

    async def refine_audio(self, audio_chunk: AudioChunk) -> AudioChunk:
        """Applies pre-STT audio refinement to an audio chunk."""
        raise NotImplementedError

    async def transcribe(self, audio_chunk: AudioChunk) -> Transcript:
        """Converts an audio chunk into transcript text."""
        raise NotImplementedError


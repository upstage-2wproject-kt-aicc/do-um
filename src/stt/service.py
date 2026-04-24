"""STT interface skeleton for VAD and transcript generation."""

from src.common.schemas import AudioChunk, ClientInfo, Transcript


class STTService:
    """Defines STT boundary methods for audio processing."""

    async def start_session(self, client_info: ClientInfo) -> str:
        """Creates a new session from an inbound call.

        Generates a session_id (uuid4), persists client_info in the session
        store keyed by session_id, and returns the session_id.
        Downstream modules retrieve caller context via session_id as needed.
        """
        raise NotImplementedError

    async def apply_vad(self, audio_chunk: AudioChunk) -> AudioChunk:
        """Applies voice activity detection to an audio chunk."""
        raise NotImplementedError

    async def refine_audio(self, audio_chunk: AudioChunk) -> AudioChunk:
        """Applies pre-STT audio refinement to an audio chunk."""
        raise NotImplementedError

    async def transcribe(self, audio_chunk: AudioChunk) -> Transcript:
        """Converts an audio chunk into transcript text."""
        raise NotImplementedError


"""TTS interface for streaming audio output."""

import os
import re
import asyncio
from typing import AsyncIterator

import azure.cognitiveservices.speech as speechsdk
from cachetools import LRUCache

from src.common.schemas import LLMResponse, TTSChunk


class TTSService:
    """Defines TTS boundary methods for stream synthesis."""

    async def stream(self, response: LLMResponse) -> AsyncIterator[TTSChunk]:
        """Converts LLM text response into a TTS chunk stream."""
        raise NotImplementedError
        # Use yield for typing purposes in skeleton
        yield TTSChunk(session_id="", chunk_id=0, audio_bytes=b"", is_last=True)


class AzureTTSService(TTSService):
    """Azure TTS implementation with caching, normalization, and fallback."""

    def __init__(self, cache_size: int = 100):
        # Initialize Azure SDK config
        self.speech_key = os.environ.get("AZURE_SPEECH_KEY", "")
        self.speech_region = os.environ.get("AZURE_SPEECH_REGION", "")
        
        if self.speech_key and self.speech_region:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key, 
                region=self.speech_region
            )
            # Use a fast, mono format suitable for chatbot streaming
            self.speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
            )
        else:
            self.speech_config = None
            
        # In-memory LRU cache for generated audio chunks
        self.audio_cache = LRUCache(maxsize=cache_size)
        self.voice_name = "ko-KR-SunHiNeural"
        
        # Pre-recorded fallback chunk (empty or default message)
        # In a real system, this would be a loaded mp3 byte array
        self.fallback_audio_bytes = b"" 

    def _normalize_financial_text(self, text: str) -> str:
        """Normalizes numbers and symbols for Korean TTS."""
        if not text:
            return ""
        # Currency symbol to text (e.g., ₩12,500 -> 12500원)
        text = re.sub(r'₩\s*([0-9,]+)', lambda m: m.group(1).replace(',', '') + '원', text)
        # Dates (YYYY-MM-DD -> YYYY년 MM월 DD일)
        text = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\1년 \2월 \3일', text)
        return text

    def _build_ssml(self, text: str) -> str:
        """Wraps text in SSML for Azure TTS."""
        return f"""
        <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='ko-KR'>
            <voice name='{self.voice_name}'>
                {text}
            </voice>
        </speak>
        """

    async def stream(self, response: LLMResponse) -> AsyncIterator[TTSChunk]:
        """Generates TTS audio stream from LLM text."""
        raw_text = response.text
        if not raw_text or not raw_text.strip():
            return
            
        # 1. Normalize
        normalized_text = self._normalize_financial_text(raw_text)
        
        # 2. Check Cache
        cache_key = f"{self.voice_name}:{normalized_text}"
        if cache_key in self.audio_cache:
            yield TTSChunk(
                session_id=response.session_id,
                chunk_id=0,
                audio_bytes=self.audio_cache[cache_key],
                is_last=True
            )
            return

        # 3. Fallback if config is missing
        if not self.speech_config:
            print("Warning: Azure TTS credentials not found. Returning fallback.")
            yield TTSChunk(
                session_id=response.session_id,
                chunk_id=0,
                audio_bytes=self.fallback_audio_bytes,
                is_last=True
            )
            return

        # 4. Generate SSML
        ssml = self._build_ssml(normalized_text)
        
        # 5. Call Azure API (Async wrapper around sync SDK)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config, 
            audio_config=None  # Receive audio in memory instead of playing it
        )
        
        def _synthesize():
            return synthesizer.speak_ssml(ssml)

        # Run synthesis in a separate thread to avoid blocking the event loop
        result = await asyncio.to_thread(_synthesize)

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_data = result.audio_data
            # Save to cache
            self.audio_cache[cache_key] = audio_data
            
            yield TTSChunk(
                session_id=response.session_id,
                chunk_id=0,
                audio_bytes=audio_data,
                is_last=True
            )
        else:
            # Fallback on failure
            print(f"TTS synthesis failed: {result.reason}")
            if result.cancellation_details:
                print(f"Error details: {result.cancellation_details.error_details}")
            
            yield TTSChunk(
                session_id=response.session_id,
                chunk_id=0,
                audio_bytes=self.fallback_audio_bytes,
                is_last=True
            )


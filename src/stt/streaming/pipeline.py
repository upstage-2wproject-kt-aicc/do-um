"""실시간 스트리밍 STT 파이프라인: VAD → STT(Streaming).

STT 백엔드 선택:
    stt_provider="google"  → Google Cloud STT (Streaming API)
    stt_provider="openai"  → OpenAI Whisper API (One-shot)
"""

import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Generator

import webrtcvad
from dotenv import load_dotenv

from common.schemas import Transcript
from stt.streaming.vocabulary import get_financial_vocabulary

load_dotenv()

_FINANCE_VOCAB = get_financial_vocabulary()
_WHISPER_PROMPT = ", ".join(_FINANCE_VOCAB)

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples
FRAME_BYTES = FRAME_SAMPLES * 2                      # 16-bit PCM = 2 bytes/sample
SILENCE_LIMIT_FRAMES = 12                            # 12 * 30ms = 360ms 침묵 → 발화 종료 (레이턴시 최적화)


@dataclass
class StreamingPipeline:
    """webrtcvad로 발화 구간 감지 → Google Streaming STT or OpenAI."""

    google_project_id: str = ""
    stt_provider: Literal["google", "openai"] = "google"
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vad_aggressiveness: int = 2
    silence_limit: int = SILENCE_LIMIT_FRAMES

    _speech_buffer: list[bytes] = field(default_factory=list, init=False, repr=False)
    _pre_roll_buffer: deque = field(default_factory=lambda: deque(maxlen=10), init=False, repr=False)
    _silent_frames: int = field(default=0, init=False, repr=False)
    _is_speaking: bool = field(default=False, init=False, repr=False)
    _vad: webrtcvad.Vad = field(init=False, repr=False)

    def __post_init__(self):
        self._vad = webrtcvad.Vad(self.vad_aggressiveness)
        
        if self.stt_provider == "google":
            from google.cloud import speech
            self._google_client = speech.SpeechClient()
            self._google_config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE,
                language_code="ko-KR",
                model="telephony",
                use_enhanced=True,
                speech_contexts=[
                    speech.SpeechContext(phrases=_FINANCE_VOCAB, boost=5.0)
                ],
            )
            self._streaming_config = speech.StreamingRecognitionConfig(
                config=self._google_config,
                interim_results=False
            )
            
        if self.stt_provider == "openai":
            from openai import OpenAI
            if not os.getenv("LLM_GPT_API_KEY"):
                raise ValueError("OpenAI STT 사용 시 .env에 LLM_GPT_API_KEY가 필요합니다.")
            self._openai_client = OpenAI(api_key=os.getenv("LLM_GPT_API_KEY"))

    def feed(self, frame: bytes) -> Transcript | None:
        """30ms PCM 프레임 입력 → 발화 완료 시 Transcript 반환."""
        if len(frame) != FRAME_BYTES:
            return None

        is_speech = self._vad.is_speech(frame, SAMPLE_RATE)

        if is_speech:
            if not self._is_speaking:
                # 무음 -> 유음 전환 시: 링 버퍼에 모아둔 앞부분(Pre-roll) 오디오를 먼저 버퍼에 넣음
                self._speech_buffer.extend(self._pre_roll_buffer)
                self._pre_roll_buffer.clear()
                
            self._speech_buffer.append(frame)
            self._silent_frames = 0
            self._is_speaking = True
        elif self._is_speaking:
            self._speech_buffer.append(frame)
            self._silent_frames += 1
            if self._silent_frames >= self.silence_limit:
                result = self._flush()
                self._reset()
                return result
        else:
            # 말을 하고 있지 않을 때는 항상 최근 N개의 프레임을 링 버퍼에 보관
            self._pre_roll_buffer.append(frame)

        return None

    def _reset(self):
        self._speech_buffer = []
        self._pre_roll_buffer.clear()
        self._silent_frames = 0
        self._is_speaking = False

    def _flush(self) -> Transcript | None:
        if not self._speech_buffer:
            return None
        
        raw_pcm = b"".join(self._speech_buffer)
        
        if self.stt_provider == "google":
            return self._transcribe_google_stream(self._speech_buffer)
        else:
            # NR 없이 바로 전달 (레이턴시 최적화)
            import wave
            import io
            with io.BytesIO() as wav_io:
                with wave.open(wav_io, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(raw_pcm)
                wav_bytes = wav_io.getvalue()
            return self._transcribe_openai(wav_bytes)

    def _transcribe_google_stream(self, audio_chunks: list[bytes]) -> Transcript | None:
        """Google StreamingRecognize API를 사용하여 인식 (발화 단위 스트리밍)."""
        from google.cloud import speech

        def request_generator() -> Generator:
            for chunk in audio_chunks:
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        try:
            responses = self._google_client.streaming_recognize(
                config=self._streaming_config,
                requests=request_generator(),
            )

            full_text = ""
            confidence = 0.0
            
            for response in responses:
                for result in response.results:
                    if result.is_final:
                        alternative = result.alternatives[0]
                        full_text += alternative.transcript
                        confidence = max(confidence, alternative.confidence)

            if not full_text:
                return None

            return Transcript(
                session_id=self.session_id,
                text=full_text.strip(),
                language="ko-KR",
                confidence=confidence,
                is_final=True,
                timestamp_ms=int(time.time() * 1000),
            )
        except Exception as e:
            print(f"  ⚠️ Google Streaming STT 오류: {e}")
            return None

    def _transcribe_openai(self, wav_bytes: bytes) -> Transcript | None:
        import io
        try:
            response = self._openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
                language="ko",
                prompt=_WHISPER_PROMPT,
            )
            text = response.text
            if not text:
                return None
            return Transcript(
                session_id=self.session_id,
                text=text,
                language="ko-KR",
                confidence=1.0,
                is_final=True,
                timestamp_ms=int(time.time() * 1000),
            )
        except Exception as e:
            print(f"  ⚠️ OpenAI STT 오류: {e}")
            return None

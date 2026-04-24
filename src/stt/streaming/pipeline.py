"""실시간 스트리밍 STT 파이프라인: VAD → NR → STT.

STT 백엔드 선택:
    stt_provider="google"  → Google Cloud STT (기본값, 현재 사용 가능)
    stt_provider="openai"  → OpenAI Whisper API (.env의 OPENAI_API_KEY 필요)
"""

import os
import subprocess
import tempfile
import time
import uuid
import wave
from dataclasses import dataclass, field
from typing import Literal

import webrtcvad
from dotenv import load_dotenv

from common.schemas import Transcript

load_dotenv()

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples
FRAME_BYTES = FRAME_SAMPLES * 2                      # 16-bit PCM = 2 bytes/sample
SILENCE_LIMIT_FRAMES = 15                            # 15 * 30ms = 450ms 침묵 → 발화 종료


@dataclass
class StreamingPipeline:
    """webrtcvad로 발화 구간 감지 → FFmpeg NR → STT (Google or OpenAI)."""

    google_project_id: str = ""
    stt_provider: Literal["google", "openai"] = "google"
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vad_aggressiveness: int = 3
    silence_limit: int = SILENCE_LIMIT_FRAMES

    _speech_buffer: list[bytes] = field(default_factory=list, init=False, repr=False)
    _silent_frames: int = field(default=0, init=False, repr=False)
    _is_speaking: bool = field(default=False, init=False, repr=False)
    _vad: webrtcvad.Vad = field(init=False, repr=False)

    def __post_init__(self):
        self._vad = webrtcvad.Vad(self.vad_aggressiveness)
        if self.stt_provider == "google" and not self.google_project_id:
            raise ValueError("Google STT 사용 시 google_project_id가 필요합니다.")
        if self.stt_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI STT 사용 시 .env에 OPENAI_API_KEY가 필요합니다.")
        if self.stt_provider == "openai":
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def feed(self, frame: bytes) -> Transcript | None:
        """30ms PCM 프레임(16kHz·16-bit·mono) 입력 → 발화 완료 시 Transcript 반환."""
        if len(frame) != FRAME_BYTES:
            return None

        is_speech = self._vad.is_speech(frame, SAMPLE_RATE)

        if is_speech:
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

        return None

    def _reset(self):
        self._speech_buffer = []
        self._silent_frames = 0
        self._is_speaking = False

    def _flush(self) -> Transcript | None:
        if not self._speech_buffer:
            return None
        raw_pcm = b"".join(self._speech_buffer)
        nr_wav = self._apply_nr(raw_pcm)
        return self._transcribe(nr_wav)

    def _apply_nr(self, raw_pcm: bytes) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            in_path = tmp.name
        _write_wav(in_path, raw_pcm)
        out_path = in_path.replace(".wav", "_nr.wav")

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", in_path, "-af", "afftdn=nf=-25",
                 "-loglevel", "error", out_path],
                check=True,
            )
            with open(out_path, "rb") as f:
                return f.read()
        except Exception as e:
            print(f"  ⚠️ NR 실패, 원본 사용: {e}")
            with open(in_path, "rb") as f:
                return f.read()
        finally:
            for p in [in_path, out_path]:
                if os.path.exists(p):
                    os.remove(p)

    def _transcribe(self, wav_bytes: bytes) -> Transcript | None:
        if self.stt_provider == "google":
            return self._transcribe_google(wav_bytes)
        return self._transcribe_openai(wav_bytes)

    def _transcribe_google(self, wav_bytes: bytes) -> Transcript | None:
        from google.cloud import speech_v2

        client = speech_v2.SpeechClient()
        config = speech_v2.RecognitionConfig(
            auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
            language_codes=["ko-KR"],
            model="telephony",
        )
        request = speech_v2.RecognizeRequest(
            recognizer=f"projects/{self.google_project_id}/locations/global/recognizers/_",
            config=config,
            content=wav_bytes,
        )
        try:
            response = client.recognize(request=request)
            if not response.results:
                return None
            text = " ".join(r.alternatives[0].transcript for r in response.results)
            confidence = response.results[0].alternatives[0].confidence if response.results else 0.0
            return Transcript(
                session_id=self.session_id,
                text=text,
                language="ko-KR",
                confidence=confidence,
                is_final=True,
                timestamp_ms=int(time.time() * 1000),
            )
        except Exception as e:
            print(f"  ⚠️ Google STT 오류: {e}")
            return None

    def _transcribe_openai(self, wav_bytes: bytes) -> Transcript | None:
        import io
        try:
            response = self._openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
                language="ko",
            )
            text = response.text
            if not text:
                return None
            return Transcript(
                session_id=self.session_id,
                text=text,
                language="ko-KR",
                confidence=1.0,  # Whisper는 confidence를 제공하지 않음
                is_final=True,
                timestamp_ms=int(time.time() * 1000),
            )
        except Exception as e:
            print(f"  ⚠️ OpenAI STT 오류: {e}")
            return None


def _write_wav(path: str, pcm_bytes: bytes):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)

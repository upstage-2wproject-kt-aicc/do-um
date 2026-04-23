import asyncio
from src.tts.service import AzureTTSService
from src.common.schemas import LLMResponse

async def main():
    print("Testing Azure TTS Service...")
    service = AzureTTSService()
    
    # Check normalization
    text = "테스트입니다. 결제 금액은 ₩12,500 이며, 결제일은 2026-04-23 입니다."
    norm_text = service._normalize_financial_text(text)
    print(f"Normalized Text: {norm_text}")
    assert "12500원" in norm_text
    assert "2026년 04월 23일" in norm_text
    print("Normalization test passed!")

    # Test stream (will use fallback if no key)
    print("Testing stream generator...")
    resp = LLMResponse(session_id="test_session", provider="test", text=text, latency_ms=0)
    
    chunks = []
    async for chunk in service.stream(resp):
        chunks.append(chunk)
        print(f"Received chunk: {len(chunk.audio_bytes)} bytes, is_last={chunk.is_last}")

    if len(chunks) > 0 and len(chunks[0].audio_bytes) > 0:
        print("Audio generated successfully!")
        with open("test_audio.mp3", "wb") as f:
            f.write(chunks[0].audio_bytes)
        print("Saved to test_audio.mp3")
    else:
        print("Using fallback audio (no credentials or failed)")

if __name__ == "__main__":
    asyncio.run(main())

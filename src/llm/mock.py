import asyncio
from collections.abc import AsyncIterator


async def mock_llm_stream(text: str) -> AsyncIterator[str]:
    content = text or "테스트 응답입니다."
    for token in content.split():
        yield token + " "
        await asyncio.sleep(0.01)


async def sentence_chunker(token_stream: AsyncIterator[str]) -> AsyncIterator[str]:
    buffer = ""
    delimiters = (".", "?", "!", "。", "？", "！")
    async for token in token_stream:
        buffer += token
        if any(d in token for d in delimiters):
            sentence = buffer.strip()
            if sentence:
                yield sentence
            buffer = ""
    tail = buffer.strip()
    if tail:
        yield tail

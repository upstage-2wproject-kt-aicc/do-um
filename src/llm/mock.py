import asyncio
import re
from typing import AsyncGenerator

async def mock_llm_stream(text: str = None) -> AsyncGenerator[str, None]:
    """
    가상의 LLM 응답을 토큰 단위로 반환하는 비동기 제너레이터입니다.
    기본 텍스트가 없으면 금융 챗봇에 어울리는 예시 문장을 스트리밍합니다.
    """
    if not text:
        text = "안녕하세요. 고객님의 현재 대출 잔액은 1억 5천만 원입니다. 추가로 궁금한 점이 있으시면 언제든지 말씀해주세요. 감사합니다."
    
    # 띄어쓰기를 기준으로 토큰화 (실제 LLM 토큰 단위와 유사하게 모방)
    tokens = [word + " " for word in text.split(" ")]
    # 마지막 토큰의 불필요한 공백 제거
    tokens[-1] = tokens[-1].strip()
    
    for token in tokens:
        await asyncio.sleep(0.1)  # LLM 생성 지연 시뮬레이션
        yield token

async def sentence_chunker(token_stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """
    LLM의 토큰 스트림을 받아 문장 단위로 버퍼링한 후 반환합니다.
    한국어 종결 어미나 기호(. ? ! \n)를 기준으로 자릅니다.
    """
    buffer = ""
    # 마침표, 물음표, 느낌표, 줄바꿈 뒤에 공백이나 문장 끝이 오는 경우를 문장 종결로 판단
    end_marks = re.compile(r'(?<=[.!?\n])(\s+|$)')
    
    async for token in token_stream:
        buffer += token
        
        # 버퍼 내에 문장 종결 기호가 포함되었는지 확인
        match = end_marks.search(buffer)
        if match:
            # 종결 기호까지를 하나의 문장으로 추출
            split_idx = match.end()
            sentence = buffer[:split_idx].strip()
            
            if sentence:
                yield sentence
                
            # 남은 부분을 다시 버퍼에 저장
            buffer = buffer[split_idx:].lstrip()
            
    # 스트림 종료 후 버퍼에 남은 텍스트가 있다면 반환
    if buffer.strip():
        yield buffer.strip()

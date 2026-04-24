"""
공통 로깅 모듈 (loguru 기반)
콘솔 출력 및 일자별 파일 저장을 담당합니다.
"""
import sys
from loguru import logger

# 기본 로거 제거 (중복 출력 방지)
logger.remove()

# 1. 콘솔 출력 설정 (개발 및 디버깅용)
# 색상을 지원하여 가독성을 높입니다.
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
)

# 2. 파일 출력 설정 (실무 환경, INFO 레벨 이상)
# 매일 자정(1 day) 기준으로 새 파일 생성, 최대 7일 보관
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
    encoding="utf-8"
)

# 3. 에러 전용 파일 분리 (모니터링 용이성)
logger.add(
    "logs/error.log",
    rotation="10 MB",
    retention="30 days",
    level="ERROR",
    encoding="utf-8"
)

def get_logger():
    """설정된 전역 로거 인스턴스를 반환합니다."""
    return logger

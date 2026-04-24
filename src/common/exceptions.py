from typing import Optional, Dict, Any

class BaseAppException(Exception):
    """
    애플리케이션 전역 최상위 예외 클래스입니다.
    FastAPI 에러 처리기에서 이 클래스를 낚아채서 일관된 응답을 만듭니다.
    """
    def __init__(
        self,
        status_code: int = 500,
        error_code: str = "INTERNAL_SERVER_ERROR",
        message: str = "시스템 내부 오류가 발생했습니다.",
        detail: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.detail = detail
        self.data = data
        super().__init__(self.message)

class TTSException(BaseAppException):
    """TTS(음성 합성) 관련 예외 발생 시 사용합니다."""
    def __init__(self, message: str, error_code: str = "TTS_ERROR", detail: Optional[str] = None):
        super().__init__(
            status_code=500, 
            error_code=error_code, 
            message=message, 
            detail=detail
        )

class ValidationException(BaseAppException):
    """입력 데이터가 유효하지 않을 때 사용합니다."""
    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(
            status_code=400, 
            error_code="INVALID_INPUT", 
            message=message, 
            detail=detail
        )

class AuthException(BaseAppException):
    """API 키 누락 등 인증 실패 시 사용합니다."""
    def __init__(self, message: str = "인증 정보가 유효하지 않습니다."):
        super().__init__(
            status_code=401, 
            error_code="UNAUTHORIZED", 
            message=message
        )

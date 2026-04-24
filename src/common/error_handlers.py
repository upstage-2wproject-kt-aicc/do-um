from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from common.exceptions import BaseAppException
from common.logger import logger

def register_error_handlers(app: FastAPI):
    """
    FastAPI 애플리케이션에 전역 에러 처리기를 등록합니다.
    모든 에러를 일관된 JSON 포맷으로 반환하도록 강제합니다.
    """

    @app.exception_handler(BaseAppException)
    async def app_exception_handler(request: Request, exc: BaseAppException):
        """우리가 정의한 커스텀 예외(BaseAppException)가 발생했을 때 호출됩니다."""
        # 로그에는 상세 정보를 남겨 디버깅을 돕습니다.
        logger.error(f"애플리케이션 에러 발생: [{exc.error_code}] {exc.message} | Detail: {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "detail": exc.detail
                }
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """코드의 버그 등 미처 처리하지 못한 모든 일반 예외를 처리합니다."""
        # 에러의 전체 트레이스백을 로그에 기록합니다.
        logger.exception(f"처리되지 않은 일반 에러 발생: {str(exc)}")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "시스템 내부 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                    "detail": str(exc)  # 실제 운영 시에는 보안을 위해 이 상세 내용을 감추는 설정을 추가할 수 있습니다.
                }
            }
        )

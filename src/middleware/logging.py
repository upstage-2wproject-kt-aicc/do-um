import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from common.logger import logger

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    모든 HTTP 요청의 시작과 끝을 로깅하는 미들웨어입니다.
    금융 챗봇에서 중요한 응답 지연 시간(Latency)을 측정합니다.
    """
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        
        # 1. 요청 기본 정보 추출
        path = request.url.path
        method = request.method
        # 쿼리 파라미터가 있다면 함께 로깅
        query_params = request.url.query
        full_path = f"{path}?{query_params}" if query_params else path
        
        logger.info(f"[REQ] {method} {full_path}")
        
        try:
            # 2. 다음 프로세스(엔드포인트) 호출
            response = await call_next(request)
            
            # 3. 처리 시간 계산 및 완료 로깅
            process_time = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"[RES] {method} {path} | Status: {response.status_code} | Latency: {process_time:.2f}ms"
            )
            
            # 응답 헤더에 처리 시간 추가 (디버깅용)
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
            
            return response
            
        except Exception as e:
            # 에러 발생 시에도 처리 시간을 기록합니다.
            process_time = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[ERR] {method} {path} | Latency: {process_time:.2f}ms | Error: {str(e)}"
            )
            # 에러는 다시 던져서 전역 에러 처리기(Error Handler)가 처리하게 합니다.
            raise e

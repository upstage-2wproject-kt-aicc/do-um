# Windows 실행 가이드 (Python 3.11 + uv)

이 문서는 Windows 사용자용 실행 절차를 정리한 문서다.
현재 저장소는 `pyproject.toml`의 `tool.uv.environments`가 `darwin`으로 제한되어 있으므로, Windows에서는 1회 수정이 필요하다.

## 0) 사전 준비
- OS: Windows 10/11
- Python: 3.11.x
- uv 설치 완료
- 저장소 루트에서 작업

확인 명령(PowerShell):
```powershell
py -3.11 --version
uv --version
```

## 1) 저장소 클론
```powershell
git clone https://github.com/upstage-2wproject-kt-aicc/do-um.git
cd do-um
```

## 2) `pyproject.toml` 1회 수정 (필수)
파일: `pyproject.toml`

현재(예시):
```toml
[tool.uv]
managed = true
environments = ["sys_platform == 'darwin' and python_full_version >= '3.11' and python_full_version < '3.12'"]
```

Windows 포함 예시(권장):
```toml
[tool.uv]
managed = true
environments = [
  "(sys_platform == 'darwin' or sys_platform == 'win32') and python_full_version >= '3.11' and python_full_version < '3.12'"
]
```

## 3) 의존성 설치
```powershell
$env:UV_CACHE_DIR=".uv-cache"
uv sync --python 3.11
```

## 4) 모델 디렉토리 배치
공유받은 NLU 모델 폴더를 아래 경로에 배치:
```text
src/nlu/my_aicc_nlu_model_klue
```

최소 포함 파일 예:
- `config.json`
- `tokenizer_config.json`
- `tokenizer.json`
- `model.safetensors`

## 5) 환경변수(.env) 설정
```powershell
Copy-Item .env.example .env
```

최소 필수:
- `LLM_SOLAR_API_KEY`
- `LLM_GPT_API_KEY`

실행 provider에 따라 추가:
- STT OpenAI 사용 시: `STT_PROVIDER=openai`
- TTS OpenAI 사용 시: `TTS_PROVIDER=openai`
- STT Google 사용 시: `STT_PROVIDER=google`, `GOOGLE_PROJECT_ID` (+ ADC 설정)
- TTS Azure 사용 시: `TTS_PROVIDER=azure`, `AZURE_SPEECH_KEY`, `AZURE_SERVICE_REGION`

## 6) 서버 실행
```powershell
$env:HF_HOME=".hf-cache"
$env:STT_PROVIDER="openai"
$env:TTS_PROVIDER="openai"
$env:UV_CACHE_DIR=".uv-cache"
uv run python -m src.main
```

서버 확인:
- `http://127.0.0.1:8000/`

## 7) 자주 발생하는 오류
1. `The current Python platform is not compatible with the lockfile's supported environments`
- 원인: `tool.uv.environments`에 `win32` 미반영
- 조치: 2단계 수정 후 `uv sync --python 3.11` 재실행

2. `Tokenizer class TokenizersBackend does not exist ...`
- 원인: 모델 메타데이터 불일치/오래된 모델 아티팩트
- 조치: 공유된 최신 모델 디렉토리로 교체

3. STT/TTS 호출 실패(401/403/NETWORK)
- 원인: `.env` 키 누락 또는 provider 설정 불일치
- 조치: 5단계의 provider별 필수 키 확인

## 8) 권장 운영 규칙
- 로컬 산출물 커밋 금지: `.venv/`, `.uv-cache/`, `.hf-cache/`, `data/`
- Python 버전은 팀 공통으로 3.11 유지

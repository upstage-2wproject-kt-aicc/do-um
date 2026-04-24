import os
import io
import time
import asyncio
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from tts.factory import TTSFactory
from common.schemas import LLMResponse
from common.exceptions import BaseAppException
from common.logger import logger

load_dotenv()

# ---------------------------------------------------------
# 1. Config
# ---------------------------------------------------------

PROVIDERS = {
    "OpenAI": {
        "service_name": "openai",
        "voices": ["alloy"],
    },
    "Azure": {
        "service_name": "azure",
        "voices": ["ko-KR-SunHiNeural"],
    },
    "Naver Clova": {
        "service_name": "naver",
        "voices": ["nara_call"],
    },
    "Google": {
        "service_name": "google",
        "voices": ["ko-KR-Wavenet-A"],
    }
}

SCRIPTS = {
    "환영 인사": "안녕하세요. NH농협 AICC 상담원입니다. 무엇을 도와드릴까요?",
    "금융 안내": "현재 고객님의 대출 잔액은 일억 이천삼백사십오만 원이며, 금리는 연 삼 점 오 퍼센트입니다.",
    "사과": "죄송합니다. 현재 시스템 점검으로 인해 상담이 지연되고 있습니다. 잠시만 기다려 주시겠습니까?"
}

# ---------------------------------------------------------
# 2. Generate Logic (Async)
# ---------------------------------------------------------

async def generate_single_provider(provider_name, text):
    """특정 공급자의 음성을 합성하고 결과를 반환합니다."""
    # Use Factory to get service dynamically
    provider_id = PROVIDERS[provider_name]["service_name"]
    service = TTSFactory.get_service(provider_id)
    
    resp = LLMResponse(session_id="streamlit_test", provider=provider_name, text=text, latency_ms=0)
    
    start_time = time.perf_counter()
    audio_bytes = b""
    
    try:
        async for chunk in service.stream(resp):
            audio_bytes += chunk.audio_bytes
            
        latency = (time.perf_counter() - start_time)
        return {
            "provider": provider_name,
            "audio": io.BytesIO(audio_bytes),
            "latency": latency,
            "success": len(audio_bytes) > 0
        }
    except BaseAppException as e:
        logger.error(f"[{provider_name}] 앱 에러: {e.message} ({e.error_code})")
        st.error(f"**{provider_name} 오류**: {e.message}")
        return None
    except Exception as e:
        logger.exception(f"[{provider_name}] 예상치 못한 오류 발생")
        st.error(f"**{provider_name} 시스템 오류**: {str(e)}")
        return None

async def generate_all_providers(selected_providers, text):
    """선택된 모든 공급자에 대해 비동기적으로 합성을 진행합니다."""
    tasks = [generate_single_provider(p, text) for p in selected_providers]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

# ---------------------------------------------------------
# 3. UI Layer
# ---------------------------------------------------------

def main():
    st.set_page_config(page_title="AICC TTS Comparison", page_icon="🎤", layout="wide")

    st.title("🎤 AICC TTS 비교 실험 플랫폼 (v2.0)")
    st.markdown("""
    이 플랫폼은 **Clean Architecture** 기반으로 리팩토링되었습니다. 
    각 모델은 독립된 서비스 객체로 동작하며, 공통 에러 핸들링 및 로깅 시스템이 적용되어 있습니다.
    """)

    # Sidebar
    st.sidebar.header("⚙️ 서비스 설정")

    selected_providers = st.sidebar.multiselect(
        "테스트할 TTS 모델 선택",
        list(PROVIDERS.keys()),
        default=["Azure", "Naver Clova"]
    )

    selected_script = st.sidebar.selectbox(
        "테스트 시나리오 선택",
        list(SCRIPTS.keys())
    )

    st.sidebar.divider()
    st.sidebar.info("💡 **Tip**: 각 모델의 설정을 변경하려면 `src/tts/service.py`를 수정하세요.")

    # Main Area
    text_input = st.text_area(
        "합성할 텍스트를 입력하세요",
        value=SCRIPTS[selected_script],
        height=150
    )

    if st.button("🚀 전체 모델 비교 실행", type="primary"):
        if not selected_providers:
            st.warning("최소 하나 이상의 모델을 선택해주세요.")
            return

        with st.spinner("음성 합성 중..."):
            # Streamlit에서 비동기 함수 실행
            results = asyncio.run(generate_all_providers(selected_providers, text_input))

        if not results:
            st.error("모든 모델의 합성에 실패했습니다. 로그를 확인해주세요.")
            return

        # 결과 출력 루프
        evaluation_data = []
        
        # 2개씩 한 줄에 배치
        cols = st.columns(2)
        for idx, result in enumerate(results):
            with cols[idx % 2]:
                st.subheader(f"🏢 {result['provider']}")
                
                if result["success"]:
                    st.audio(result["audio"])
                    st.caption(f"⏱ 지연 시간: **{result['latency']:.3f}초**")
                    
                    # 평가 섹션
                    with st.expander("⭐ 품질 평가 기록"):
                        p = st.slider("발음 정확도", 1, 5, 3, key=f"p_{result['provider']}")
                        n = st.slider("자연스러움", 1, 5, 3, key=f"n_{result['provider']}")
                        t = st.slider("신뢰감 (톤)", 1, 5, 3, key=f"t_{result['provider']}")
                        
                        evaluation_data.append({
                            "provider": result["provider"],
                            "latency": result["latency"],
                            "pronunciation": p,
                            "naturalness": n,
                            "tone": t
                        })
                else:
                    st.warning("오디오 생성 결과가 없습니다.")
                
                st.divider()

        # 결과 저장 및 다운로드
        if evaluation_data:
            st.success("✅ 모든 합성이 완료되었습니다.")
            df = pd.DataFrame(evaluation_data)
            
            st.download_button(
                "📊 결과 데이터(CSV) 다운로드",
                df.to_csv(index=False).encode('utf-8-sig'),
                file_name=f"tts_eval_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
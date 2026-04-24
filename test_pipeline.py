import os
from fastapi.testclient import TestClient
from src.main import app

def test_pipeline_stream():
    client = TestClient(app)
    
    # 1. 테스트용 더미 입력 데이터
    dummy_payload = {
        "session_id": "test-session-001",
        "original_query": "비밀번호를 분실했어요",
        "routing_info": {
            "intent": "질의형",
            "subdomain": "비밀번호",
            "router_confidence": 0.95,
            "domain": "보안"
        },
        "chat_history": [],
        "internal_context": [],
        "policy_rules": []
    }

    print("서버에 파이프라인 스트리밍 요청을 보냅니다...")
    # 2. 새로 만든 파이프라인 API 호출 (provider 파라미터 테스트)
    # ?provider=google 과 같이 쿼리 파라미터를 넘길 수 있습니다.
    response = client.post("/pipeline/stream?provider=google", json=dummy_payload)
    
    # 3. 결과 검증 및 저장
    if response.status_code == 200:
        print("API 호출 성공 (200 OK)")
        output_file = "output_test.mp3"
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        file_size = os.path.getsize(output_file)
        print(f"테스트 성공! {output_file} 파일이 생성되었습니다. (크기: {file_size} bytes)")
    else:
        print(f"API 호출 실패: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_pipeline_stream()

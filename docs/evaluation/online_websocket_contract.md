# Online Model Evaluation WebSocket Contract

이 문서는 상담사 화면이 STT WebSocket에서 workflow 응답과 모델별 평가 결과를 받는 방식입니다.

## Endpoint

- `ws://<host>/ws/stt`
- 클라이언트는 기존처럼 16kHz, mono, PCM 16-bit 오디오 청크를 전송합니다.
- 서버는 발화가 최종 확정되면 같은 WebSocket 연결로 JSON 이벤트를 순서대로 보냅니다.

## Event 1: workflow_result

첫 번째 이벤트는 고객에게 실제로 사용할 Solar 기반 workflow 응답입니다. 프론트는 이 이벤트를 받으면 상담사 화면에 고객 질문, NLU 분석, Solar 응답을 먼저 표시할 수 있습니다.

```json
{
  "type": "workflow_result",
  "session_id": "session-id",
  "transcript": "고객 발화 텍스트",
  "is_final": true,
  "nlu_analysis": {
    "status": "REQUIRE_LLM",
    "intent": "설명형",
    "metadata": {
      "domain": "금융상담",
      "subdomain": "대출/금리",
      "risk_level": "중간",
      "handoff_required": "N",
      "source_url": "https://example.com/source"
    },
    "guardrail_decision": "ALLOW",
    "guardrail_score": 0,
    "guardrail_reasons": [],
    "action": null
  },
  "workflow": {
    "session_id": "session-id",
    "results": [
      {
        "provider": "solar",
        "model": "solar-pro3",
        "answer": "TTS에 보낼 순수 답변",
        "latency_ms": 1200,
        "error": null
      }
    ],
    "final_answer_text": "TTS에 보낼 순수 답변",
    "pre_tts_text": "TTS에 보낼 순수 답변",
    "is_handoff_decided": false,
    "reference_links": ["https://example.com/source"],
    "nlu_evidence": {
      "selected_route": "faq",
      "route_reason": "default_faq"
    }
  },
  "action": null,
  "evaluation_status": "running"
}
```

### Frontend handling

- `workflow.final_answer_text`: 상담사가 먼저 확인할 기본 답변입니다.
- `workflow.reference_links`: 답변 본문에 직접 넣지 않은 근거 링크입니다.
- `evaluation_status=running`: 모델별 비교 평가가 아직 진행 중이라는 의미입니다.
- `workflow=null`이고 `evaluation_status=skipped`이면 NLU에서 직접 이관 또는 거절 처리된 케이스입니다.

## Event 2: evaluation_result

두 번째 이벤트는 Solar, GPT, Claude, Gemini 후보 답변과 LLM-as-a-Judge 평가 결과입니다. 프론트는 모델 버튼을 두고, 선택된 모델 하나의 패널만 보여주면 됩니다.

```json
{
  "type": "evaluation_result",
  "session_id": "session-id",
  "route": "faq",
  "route_reason": "default_faq",
  "customer_provider": "solar",
  "model_panels": [
    {
      "provider": "solar",
      "is_customer_answer": true,
      "answer": "Solar 후보 답변",
      "latency_ms": 1200,
      "token_usage": {
        "input_tokens": 800,
        "output_tokens": 120
      },
      "evaluation_status": "completed",
      "error": null,
      "score": 8.3,
      "quality_badge": "좋음",
      "summary": "근거 기반 안내가 안정적입니다.",
      "flags": [],
      "metrics": {
        "intent_fit": {
          "score": 8.0,
          "disagreement": 1
        },
        "accuracy": {
          "score": 8.0,
          "disagreement": 0
        }
      },
      "judge_models": ["gpt-5.5", "claude-opus-4-7", "gemini-3.1-pro-preview"],
      "details_ref": "evaluation_runs/online_sessions/session-id.json"
    }
  ],
  "timing_ms": {
    "evaluation_total": 13000
  }
}
```

### Frontend handling

- `model_panels[].provider`: 모델 버튼의 key입니다.
- `is_customer_answer=true`: 실제 고객/TTS 응답으로 사용한 모델입니다. 현재 기본값은 Solar입니다.
- `answer`: 상담사가 모델 버튼을 눌렀을 때 보여줄 후보 답변입니다.
- `score`: 1~10 원점수 평균입니다. 정규화하지 않습니다.
- `quality_badge`:
  - `좋음`: 8.0 이상
  - `주의`: 6.0 이상 8.0 미만
  - `확인 필요`: 6.0 미만 또는 생성 실패
- `flags`: judge가 표시한 위험 플래그입니다. 예: `unsupported_claim`, `missed_handoff`.
- `metrics`: 상세 점수입니다. 기본 패널에는 숨기고 펼침 영역에 두는 것을 권장합니다.
- `details_ref`: 서버에 저장된 전체 평가 JSON 경로입니다. 운영 DB가 붙으면 record id로 교체할 수 있습니다.

## Event 3: evaluation_error

평가 단계에서 오류가 나면 workflow 응답은 유지하고, 평가 오류 이벤트만 따로 보냅니다.

```json
{
  "type": "evaluation_error",
  "session_id": "session-id",
  "error": "RuntimeError"
}
```

## Notes

- LLM candidate 호출은 병렬로 시작합니다.
- Solar 응답이 끝나면 `workflow_result`를 먼저 보냅니다.
- GPT, Claude, Gemini 응답과 judge 평가가 끝나면 `evaluation_result`를 보냅니다.
- TTS에 읽힐 답변에는 원문 URL을 직접 넣지 않고, 링크는 `reference_links` 또는 `details_ref`를 통해 상담사 화면에서만 보여줍니다.
- 온라인 평가 저장 기본 위치는 `evaluation_runs/online_sessions/`이며 git에는 포함되지 않습니다.

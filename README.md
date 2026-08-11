# hyope-ai-server

독거노인 안부확인 전화를 실제로 거는 음성 AI 에이전트. xAI Grok Realtime과 Gemini Live 중 하나로
실시간 음성 대화를 진행하며, 정해진 임상 질문(우울·불안·신체건강·인지기능·복약/식사 등)을 자연스러운
대화 흐름 속에서 묻고, 답변을 분석해 통화 종료 후 결과를 메인 서버(hyope-backend, Spring)로 전달한다.

세 자매 레포 중 하나다: `hyope-backend`(Spring 메인 서버) · `hyope-frontend`(Flutter 앱) · 이 레포.

## 구조

```
agent/            순수 로직 — 네트워킹 없음
  brain.py          통화 state, 질문은행 조립, 통화 결과 파일 기록(write_call_result_outbox)
  tools.py          8개 툴(checkin_start, log_answer_analysis, double_check, flag_emergency,
                     search_nearby_resource, send_resource_info, update_recipient_profile,
                     end_call)의 스키마 + 실제 구현 + run_tool 디스패처
  prompts.py        시스템 프롬프트

integrations/      네트워킹 — 외부 API 호출
  dispatch.py        agent/tools.run_tool 앞단에서 네트워킹 필요한 툴 호출을 가로채는 허브
  geo.py             카카오 로컬 API — 주소 → 좌표 → 주변 노인복지시설 검색
  sms.py             ClawOps Messages API — 도움처 안내 문자 발송
  worker.py          runtime/ 폴링 → 채점 → Spring POST(백그라운드 스레드, server.py가 기동)

server.py          FastAPI 앱. WebSocket(브라우저 데모/실제 전화) 라우팅, ClawOps 웹훅, /call 트리거
gemini_loader.py   전화/브라우저 오디오 ↔ Gemini Live API 브릿지(server.py와 대칭되는 xAI 대안 경로)
sinks.py           오디오 트랜스코딩(G.711 μ-law ↔ PCM16/24k) + 모델 출력 → 클라이언트 어댑터
static/            questions.json(질문은행 원본), help_resources.json(전국 공통 도움처), index.html(브라우저 데모)
runtime/           통화 중 쌓이는 내구성 큐 파일들(git-ignored) — alerts/, call_results/
```

`agent/`는 네트워킹 금지 원칙을 지킨다 — 지오코딩·SMS 등 외부 호출이 필요한 툴도 `agent/tools.py`
자체는 값만 받아서 쓰고, 실제 API 호출은 `integrations/dispatch.py`가 `run_tool` 호출 전에 미리
해결해서 넘겨준다.

## 통화 흐름

1. **발신**: Spring이 `POST /call`(`X-API-KEY` 인증)로 `recipient_id`/`phone_number`/`profile`을 보내면,
   즉시 `{"status": "accepted"}`를 반환하고 백그라운드에서 ClawOps로 실제 발신.
2. **음성 세션**: 전화가 연결되면 `/ws/call-stream` WebSocket으로 오디오가 흐르고, `server.py`(xAI)
   또는 `gemini_loader.py`(Gemini)가 모델과 세션을 중개한다 — 모델은 8개 툴을 호출하며 질문은행을
   따라 대화를 진행한다. `/ws/browser`는 같은 흐름을 전화 없이 브라우저에서 테스트하기 위한 경로.
3. **통화 종료**: `agent/brain.py`의 `write_call_result_outbox()`가 통화 로그·판정 데이터를
   `runtime/call_results/outbox.jsonl`에 한 줄 append(내구성 큐 — 실제 POST와 분리).
4. **후처리**: `integrations/worker.py`가 같은 프로세스의 백그라운드 스레드(`server.py`의 lifespan에서
   기동)로 이 파일을 2초 간격 폴링 → 채점 서비스(`ASSESS_URL`, 없으면 placeholder 점수) 호출 →
   Spring `POST /internal/call-results`로 전송. 성공해야만 처리완료 원장(ledger)이 전진하므로,
   Spring이 잠시 죽어도 재시도로 결국 전달된다.
5. **위급 신호**: `flag_emergency` 툴이 high severity로 호출되면 별도로 `runtime/alerts/inbox.jsonl`에
   쌓이고, 같은 worker.py 루프가 Spring `POST /internal/emergency-alerts`로 즉시 전달한다.

## 채점 서비스 연동

`integrations/worker.py`의 `compute_assessment()`가 `ASSESS_URL`이 설정돼 있으면 별도 마이크로서비스
(`feature/serve` 브랜치의 `serve/` — KoELECTRA 감정분류기 + 척도 기반 위험도 산출)에 위임하고,
비어 있으면 자체 placeholder 점수(`risk_level: OBSERVE`)를 반환해 파이프라인이 항상 동작하게 한다.

## 실행

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 필수/선택 값 채우기
uvicorn server:app --reload --port 8000
```

브라우저 데모: `http://localhost:8000/`(`static/index.html`)에서 `/ws/browser`로 마이크 테스트 가능
(ClawOps/전화 없이 모델·질문은행만 확인).

`integrations/worker.py`는 `server.py` 기동 시 백그라운드 스레드로 자동 실행되므로 별도로 띄울 필요는
없다. 큐만 수동으로 한 번 비우고 싶을 때는:

```bash
python -m integrations.worker --once
```

## 환경 변수

`.env.example` 참고. 필수: `INTERNAL_API_KEY`, `XAI_API_KEY`, `PUBLIC_BASE_URL`,
`CLAWOPS_API_KEY`/`CLAWOPS_ACCOUNT_ID`/`CLAWOPS_SIGNING_KEY`/`CLAWOPS_FROM_NUMBER`. 그 외
`GEMINI_API_KEY`(Gemini 모델 사용 시), `KAKAO_API_KEY`(없으면 주변 시설 검색 skip, 전국 공통
핫라인만 안내), `SPRING_BASE_URL`, `ASSESS_URL`/`ASSESS_TIMEOUT`(채점 서비스)은 선택.

## 주요 엔드포인트

| Method | Path | 용도 |
|---|---|---|
| WS | `/ws/browser` | 브라우저 마이크로 모델 테스트(전화 없이) |
| WS | `/ws/call-stream` | ClawOps가 실제 전화 오디오를 중계하는 경로 |
| POST | `/call` | Spring → 발신 트리거(`X-API-KEY`) |
| POST | `/voiceml/answer` | ClawOps가 전화 연결 시 부르는 웹훅(서명 검증) |
| POST | `/webhooks/status` | ClawOps 통화 상태 콜백(연결 안 됨/끊김 등) |

## 관련 레포

- `hyope-backend` — Spring 메인 서버. `/call`을 트리거하고 `/internal/call-results`,
  `/internal/emergency-alerts`를 수신한다.
- `hyope-frontend` — Flutter 앱. 보호자/사회복지사가 통화 결과·지표를 확인.
- `feature/serve` 브랜치의 `serve/` — 별도 배포되는 감정분류 + 위험도 채점 마이크로서비스
  (`ASSESS_URL`로 연동, 무거운 torch/transformers 추론을 실시간 음성 경로에서 분리하기 위해 독립 서비스로 뺐다).

# * 통화 state 생성, 통화 결과 파일 기록.
# * 스테이지 상태머신(어떤 스테이지가 우선인지, 질문을 어떻게 고르는지)은 agent/stages.py에 있다 —
# * 이 파일은 그 외 콜 전반의 부가 정보(logic_data/emergencies/call_log_entries 등)와 Spring 전송용
# * 결과 파일 기록만 다룬다. 6개 툴의 실제 구현/스키마는 agent/tools.py에 있다.
# ! 네트워킹(FastAPI, WebSocket, HTTP 클라이언트) & 데이터 주고받기 x

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agent import stages

ROOT = Path(__file__).resolve().parent.parent

# ? inbox/outbox부분 로직 검토 필요
RUNTIME_DIR = ROOT / "runtime" # worker.py가 드레인하는 append 로그/원장류 임시 파일 전용 폴더
GUARDIAN_ALERTS_FILE = RUNTIME_DIR / "alerts" / "inbox.jsonl" # new+severity=high 감지 즉시 여기 한 줄 추가(agent/tools.py)
CALL_RESULT_OUTBOX = RUNTIME_DIR / "call_results" / "outbox.jsonl" # 통화 종료 시 여기 한 줄 추가 (worker.py가 Spring으로 드레인)

def spring_timestamp(dt: datetime | None = None) -> str:
    # * Spring의 LocalDateTime과 맞는 tz 없는 ISO-8601 문자열(예: "2026-06-18T10:00:00").

    dt = dt or datetime.now(ZoneInfo("Asia/Seoul"))
    return dt.replace(tzinfo=None).isoformat(timespec="seconds")

def new_alert_id() -> str:
    # A-1000, A-1001, ... — numbered by how many alerts are already filed.

    count = 0
    if GUARDIAN_ALERTS_FILE.exists():
        count = sum(1 for line in GUARDIAN_ALERTS_FILE.read_text().splitlines() if line.strip())
    return f"A-{1000 + count}"

def _default_profile() -> dict:
    # * 어르신 히스토리가 아직 없을 경우, 빈 파일 경로 초기화

    path = ROOT / "static" / "recipient_profile.json"
    return json.loads(path.read_text()) if path.exists() else {}

def new_call_state(call_id: str, profile: dict | None) -> dict:
    # * 한 통화의 상태 — 웹소켓 연결 동안만 존재한다. 스테이지 상태머신 자체(_stage_buffer)는
    # * agent/stages.py의 CallStageBuffer가 담당하고, 여기엔 그 외 콜 전반의 부가 정보만 둔다.
    # * profile은 통화 시작 시 넘겨받아 CallStageBuffer 생성 시 INTEREST/EVENT 질문은행 개인화에 쓰인다.

    profile = profile if profile is not None else _default_profile()
    return {
        "_stage_buffer": stages.get_or_create_buffer(call_id, profile),
        "logic_data": {},         # stage_type(value) -> {"judgment", "reason"}, filled by check_in
        "emergencies": [],        # [{"stage", "severity", "flagged_at"}, ...], filled by check_in(discomfort spawn, severity=high)
        "profile_updates": [],    # [{"action","kind","text"|"hobby_name"|"event_id","reason"}, ...] — filled by
                                   # update_recipient_profile, pushed to Spring at call end, next-call-only
        "call_log_entries": [],   # [{"sequence","question","answer","asked_at"}, ...] for Spring's call_log_entries
        "_last_offered_resources": None,  # check_external_api_necessity가 마지막으로 안내한
                                           # [{"name","phone","address"}, ...] 장소 목록
        "_unresolved_questions": [],  # [{"stage","question_id"}, ...] — fail_count 한계 초과로 포기된 질문
                                       # (사람이 나중에 검토하라는 표시 용도)
        "_last_agent_utterance": None,     # 직전 턴에 에이전트가 실제로 말한 문장 — 다음 어르신 답변과 페어링
        "_last_agent_utterance_at": None,  # 위 발화가 시작된 시각(ISO, tz 없음) — asked_at으로 씀
        "_call_started_at": None,  # browser_ws/call_stream_ws가 세션 열 때 채움 — call_log.started_at
    }

# --------------------------------------------------------------------------
# 통화 결과 파일 기록 (worker.py가 드레인)
# --------------------------------------------------------------------------

def write_call_result_outbox(state: dict, status: str) -> None:
    # * Spring 전송용 원본 데이터를 로컬에 남긴다(실제 POST는 worker.py가 폴링)
    # * 브라우저 데모는 skip

    metadata = state.get("_metadata") or {}
    recipient_id = metadata.get("recipient_id")
    if recipient_id is None:
        return

    # 통화가 조기 종료(어르신이 먼저 끊음, 위급상황 등)되면 일부 스테이지는 아예 못 다룰 수
    # 있다 — completed가 아닌 스테이지를 "미확인"으로 남겨 Spring/프론트까지 흘려보낸다
    # (채점엔 반영 안 함, 사람이 보고 판단하라는 표시 용도).
    buffer: stages.CallStageBuffer = state["_stage_buffer"]
    incomplete_categories = [
        s.stage.value for s in buffer.stages.values() if s.level != stages.StageLevel.COMPLETED
    ]

    CALL_RESULT_OUTBOX.parent.mkdir(parents=True, exist_ok=True)
    with open(CALL_RESULT_OUTBOX, "a") as f:
        f.write(json.dumps({
            "id": f"CR-{uuid.uuid4().hex[:12]}",  # worker.py의 처리완료 원장(ledger) 키
            "recipient_id": recipient_id,
            "call_log": {
                "started_at": state.get("_call_started_at"),
                "ended_at": spring_timestamp(),
                "status": status,
            },
            "call_log_entries": state["call_log_entries"],
            "logic_data": state["logic_data"],
            "emergencies": state["emergencies"],
            "profile_updates": state["profile_updates"],
            "incomplete_categories": incomplete_categories,
            "unresolved_questions": state.get("_unresolved_questions") or [],
            "filed_at": time.time(),
        }, ensure_ascii=False) + "\n")

def write_minimal_call_result(recipient_id, status: str) -> None:
    # * state 없이(발신 실패, 무응답/통화중 콜백처럼 통화가 시작도 못 한 경우) 빈 값으로 최소 결과만 기록한다.
    # * started_at을 None으로 두면 Spring CallResultService.saveAll()의 필수값 검증(둘 다 non-null)에
    # * 걸려서 이 결과가 영원히 저장 실패 → 재시도 루프에 빠진다. 시도한 시각을 그대로 쓴다(연결이 안
    # * 됐을 뿐 "시도"는 이 시점에 일어났으므로 의미상으로도 맞다).

    now = spring_timestamp()
    CALL_RESULT_OUTBOX.parent.mkdir(parents=True, exist_ok=True)
    with open(CALL_RESULT_OUTBOX, "a") as f:
        f.write(json.dumps({
            "id": f"CR-{uuid.uuid4().hex[:12]}",
            "recipient_id": recipient_id,
            "call_log": {"started_at": now, "ended_at": now, "status": status},
            "call_log_entries": [], "logic_data": {}, "emergencies": [], "profile_updates": [],
            "filed_at": time.time(),
        }, ensure_ascii=False) + "\n")

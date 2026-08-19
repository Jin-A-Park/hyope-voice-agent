
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

from agent import stages, tools

ROOT = Path(__file__).resolve().parent.parent

# ? inbox/outbox부분 로직 검토 필요
RUNTIME_DIR = ROOT / "runtime" # worker.py가 드레인하는 append 로그/원장류 임시 파일 전용 폴더
CALL_RESULT_OUTBOX = RUNTIME_DIR / "call_results" / "outbox.jsonl" # 통화 종료 시 여기 한 줄 추가 (worker.py가 Spring으로 드레인)
# 로컬 테스트용 — 실제 SMS 발송/Spring 연동과 무관하게, 통화마다 만들어진 요약 SMS 본문을 그대로
# 텍스트 파일로 남긴다. guardian_phone_number가 없거나 SMS 인증정보가 없어도(로컬 개발 환경)
# 요약 내용 자체는 항상 눈으로 확인할 수 있게 하기 위한 디버그 산출물 — Spring/worker.py와 무관.
CALL_SUMMARIES_DIR = RUNTIME_DIR / "call_summaries"

def spring_timestamp(dt: datetime | None = None) -> str:
    # * Spring의 LocalDateTime과 맞는 tz 없는 ISO-8601 문자열(예: "2026-06-18T10:00:00").

    dt = dt or datetime.now(ZoneInfo("Asia/Seoul"))
    return dt.replace(tzinfo=None).isoformat(timespec="seconds")

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
        "item_judgments": {},     # question_id -> {"judgment", "reason"}, filled by check_in — 문항
                                   # 단위라 "medication_entry"/"meal_entry"처럼 같은 스테이지 안 문항이
                                   # 여럿이어도 서로 안 덮어씀. serve/assess.py의 복약/식사 이행 판단용.
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

def _write_call_summary_debug_file(
    cr_id: str, recipient_id, recipient_name: str, guardian_phone_number: str | None, summary_sms_text: str,
) -> None:
    # * 로컬 테스트용 산출물 — Spring/ClawOps 연동 여부와 무관하게 통화가 끝날 때마다 요약 SMS
    # * 본문을 그대로 파일로 남긴다(runtime/call_summaries/CR-xxxx.txt). guardian_phone_number가
    # * 없어 실제로는 문자가 안 나가는 경우(로컬 개발 환경 등)에도 내용을 눈으로 확인할 수 있다.
    CALL_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    header = (
        f"call_result_id: {cr_id}\n"
        f"recipient_id: {recipient_id}\n"
        f"recipient_name: {recipient_name}\n"
        f"guardian_phone_number: {guardian_phone_number or '(없음 — 실제 SMS 발송 대상 아님)'}\n"
        f"generated_at: {spring_timestamp()}\n"
        "---\n"
    )
    (CALL_SUMMARIES_DIR / f"{cr_id}.txt").write_text(header + summary_sms_text, encoding="utf-8")


def write_call_result_outbox(state: dict, status: str) -> None:
    # * Spring 전송용 원본 데이터를 로컬에 남긴다(실제 POST는 worker.py가 폴링) — recipient_id가
    # * 없는 브라우저 데모는 Spring 아웃박스(아래 CALL_RESULT_OUTBOX 파일)는 건너뛰지만, 로컬
    # * 테스트용 요약 디버그 파일(_write_call_summary_debug_file)은 recipient_id 유무와 무관하게
    # * 항상 남긴다 — 사용자가 주로 브라우저 데모로 로컬 테스트를 하기 때문에, 여기서까지 건너뛰면
    # * 요약 기능을 로컬에서 확인할 방법이 없어진다.

    metadata = state.get("_metadata") or {}
    recipient_id = metadata.get("recipient_id")

    # 통화가 조기 종료(어르신이 먼저 끊음, 위급상황 등)되면 일부 스테이지는 아예 못 다룰 수
    # 있다 — completed가 아닌 스테이지를 "미확인"으로 남겨 Spring/프론트까지 흘려보낸다
    # (채점엔 반영 안 함, 사람이 보고 판단하라는 표시 용도).
    buffer: stages.CallStageBuffer = state["_stage_buffer"]
    incomplete_categories = [
        s.stage.value for s in buffer.stages.values() if s.level != stages.StageLevel.COMPLETED
    ]

    # ANXIETY/COGNITIVE처럼 반응형이라 한 번도 안 뜬 카테고리는, 짝이 되는 baseline
    # 카테고리(HEALTH/DEPRESSION)의 겸용 스크리닝 문항이 문제없음으로 끝났다면 "못 물어봤다"가
    # 아니라 "간접 스크리닝에서 이상 없었다"는 뜻이다 — serve/Spring으로 나가기 직전에 반영한다.
    tools.backfill_implicit_screening(state["logic_data"])

    # 통화 종료 시 보호자에게 보낼 요약 SMS 본문 — 보호자 전화번호는 통화 시작 시 Spring이
    # 넘겨준 profile에 실려 온다(check_external_api_necessity가 이미 같은 profile에서 address를
    # 읽는 것과 동일한 경로, integrations/dispatch.py 참고). 없으면(Spring이 아직 이 필드를 안
    # 보내주거나 보호자가 미등록인 경우) worker.py가 조용히 SMS 발송을 건너뛴다.
    # 예전 emergency-alerts 응답이 "010-1234-5678"처럼 하이픈 섞어 내려보냈던 것과 같은 이유로,
    # 여기서도 방어적으로 하이픈/공백을 제거해둔다(ClawOps가 정규화된 번호를 기대함).
    guardian_phone_number = (metadata.get("profile") or {}).get("guardian_phone_number")
    if guardian_phone_number:
        guardian_phone_number = guardian_phone_number.replace("-", "").replace(" ", "")
    recipient_name = (metadata.get("profile") or {}).get("recipient_name") or "대상자"
    summary_sms_text = tools.build_call_summary_text(state["logic_data"], buffer, metadata.get("profile") or {})
    profile_updates_text = tools.build_profile_updates_text(state["profile_updates"])
    if profile_updates_text:
        summary_sms_text += "\n\n[근황/취미 업데이트]\n" + profile_updates_text

    cr_id = f"CR-{uuid.uuid4().hex[:12]}"  # worker.py의 처리완료 원장(ledger) 키 — 요약 디버그 파일명도 같이 씀
    _write_call_summary_debug_file(cr_id, recipient_id, recipient_name, guardian_phone_number, summary_sms_text)

    if recipient_id is None:
        return  # 브라우저 데모/테스트 — Spring 아웃박스는 여기서 스킵(디버그 파일은 이미 위에서 남겼음)

    CALL_RESULT_OUTBOX.parent.mkdir(parents=True, exist_ok=True)
    with open(CALL_RESULT_OUTBOX, "a") as f:
        f.write(json.dumps({
            "id": cr_id,
            "recipient_id": recipient_id,
            "call_log": {
                "started_at": state.get("_call_started_at"),
                "ended_at": spring_timestamp(),
                "status": status,
            },
            "call_log_entries": state["call_log_entries"],
            "logic_data": state["logic_data"],
            "item_judgments": state.get("item_judgments") or {},
            "emergencies": state["emergencies"],
            "profile_updates": state["profile_updates"],
            "incomplete_categories": incomplete_categories,
            "unresolved_questions": state.get("_unresolved_questions") or [],
            "guardian_phone_number": guardian_phone_number,
            "recipient_name": recipient_name,
            "summary_sms_text": summary_sms_text,
            "filed_at": time.time(),
        }, ensure_ascii=False) + "\n")

def write_minimal_call_result(recipient_id, status: str) -> None:
    # * state 없이(발신 실패, 무응답/통화중 콜백처럼 통화가 시작도 못 한 경우) 빈 값으로 최소 결과만 기록한다.
    # * started_at을 None으로 두면 Spring CallResultService.saveAll()의 필수값 검증(둘 다 non-null)에
    # * 걸려서 이 결과가 영원히 저장 실패 → 재시도 루프에 빠진다. 시도한 시각을 그대로 쓴다(연결이 안
    # * 됐을 뿐 "시도"는 이 시점에 일어났으므로 의미상으로도 맞다).
    # * 통화가 시작도 못 했으니 요약할 내용이 없다 — guardian_phone_number도 없어 worker.py가
    # * 요약 SMS 발송을 자동으로 건너뛴다.

    now = spring_timestamp()
    CALL_RESULT_OUTBOX.parent.mkdir(parents=True, exist_ok=True)
    with open(CALL_RESULT_OUTBOX, "a") as f:
        f.write(json.dumps({
            "id": f"CR-{uuid.uuid4().hex[:12]}",
            "recipient_id": recipient_id,
            "call_log": {"started_at": now, "ended_at": now, "status": status},
            "call_log_entries": [], "logic_data": {}, "emergencies": [], "profile_updates": [],
            "guardian_phone_number": None, "recipient_name": None, "summary_sms_text": "",
            "filed_at": time.time(),
        }, ensure_ascii=False) + "\n")

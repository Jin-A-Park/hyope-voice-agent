
# * 통화 state, 질문은행 조립, 통화 결과 파일 기록.
# * 8개 툴의 실제 구현/스키마는 agent/tools.py에 있다 — 이 파일은 그게 기대는 state/질문은행 기반부.
# ! 네트워킹(FastAPI, WebSocket, HTTP 클라이언트) & 데이터 주고받기 x

from __future__ import annotations

import json
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agent.prompts import SYSTEM_PROMPT, GREETING_PROMPT




ROOT = Path(__file__).resolve().parent.parent

# ? inbox/outbox부분 로직 검토 필요
RUNTIME_DIR = ROOT / "runtime" # worker.py가 드레인하는 append 로그/원장류 임시 파일 전용 폴더
GUARDIAN_ALERTS_FILE = RUNTIME_DIR / "alerts" / "inbox.jsonl" # flag_emergency(severity=high) 감지 즉시 여기 한 줄 추가
CALL_RESULT_OUTBOX = RUNTIME_DIR / "call_results" / "outbox.jsonl" # 통화 종료 시 여기 한 줄 추가 (worker.py가 Spring으로 드레인)

def spring_timestamp(dt: datetime | None = None) -> str:
    # * Spring의 LocalDateTime과 맞는 tz 없는 ISO-8601 문자열(예: "2026-06-18T10:00:00").

    dt = dt or datetime.now(ZoneInfo("Asia/Seoul"))
    return dt.replace(tzinfo=None).isoformat(timespec="seconds")

def new_call_state(question_bank: dict) -> dict:
    # * 한 통화의 상태 — 웹소켓 연결 동안만 존재하며 id로 등록/조회하지 않는다.
    # * question_bank는 통화 시작 시 profile 반영해 한 번 만들어져 여기 실린다.

    return {
        "_question_bank": question_bank,
        "logic_data": {},         # category(KR) -> {"judgment", "reason"}, filled by log_answer_analysis
        "emergencies": [],        # [{"signal", "severity", "flagged_at"}, ...], filled by flag_emergency
        "asked_questions": {},    # category(KR) -> {question_id, ...} already asked this call
        "completed_categories": set(),  # category(KR) done (good, or its questions ran out)
        "asked_order": [],        # question_id in call order, across all categories — for
                                   # gap-conditional items like sis_recall (requires.min_gap_questions)
        "pending_retry": [],      # [{"category", "question_id"}, ...] — unknown-assessment questions
                                   # deferred to a final retry pass instead of hammered on immediately
        "double_check_counts": {},  # question_id -> times double_check'd without ever resolving —
                                     # caps the retry loop so a confused answer can't stall the call forever
        "profile_updates": [],    # [{"action","kind","text"|"hobby_name"|"event_id","reason"}, ...] — filled by
                                   # update_recipient_profile, pushed to Spring at call end, next-call-only
        "call_log_entries": [],   # [{"sequence","question","answer","asked_at"}, ...] for Spring's call_log_entries
        "_last_offered_resources": None,  # flag_emergency(severity!=high)가 마지막으로 안내한
                                           # [{"name","phone","address"}, ...] — send_resource_info가 참조
        "_recipient_profile_concern": False,  # "취미/근황" 카테고리에서 기억/조리력 concern이 한 번이라도
                                               # 나왔는지 — 카테고리 완료 시 "인지기능"을 앞당기는 데 씀
        "_resource_offered_categories": set(),  # log_answer_analysis가 도움처를 이미 제안한 카테고리 —
                                                 # 카테고리당 한 번만 제안하기 위한 중복 방지
        "_last_logged_entry_count": 0,  # log_answer_analysis 호출 시점의 call_log_entries 길이 — 그 사이
                                         # 어르신의 실제 새 발화 없이 또 호출되면(답을 지어낸 것) 거부하는 데 씀
        "_last_agent_utterance": None,     # 직전 턴에 에이전트가 실제로 말한 문장 — 다음 어르신 답변과 페어링
        "_last_agent_utterance_at": None,  # 위 발화가 시작된 시각(ISO, tz 없음) — asked_at으로 씀
        "_call_started_at": None,  # browser_ws/call_stream_ws가 세션 열 때 채움 — call_log.started_at
    }

# --------------------------------------------------------------------------
# 수신자 맞춤형 질문 은행 생성
# --------------------------------------------------------------------------

BASE_QUESTIONS: dict = json.loads((ROOT / "static" / "questions.json").read_text())

def _profile_to_block(profile: dict) -> dict:
    # * 메인 서버에서 받아온 어르신 히스토리를 임상 카테고리 포맷으로 변환
    # * build_question_bank에 같이 추가

    items = []
    for i, hobby in enumerate(profile.get("hobbies", []), start=1):
        items.append({
            "id": f"hobby_{i}",
            "question": f"평소 취미이신 '{hobby}', 요즘도 하고 계신지 편하게 여쭤보세요.",
        })
    for event in profile.get("recent_events", []):
        items.append({
            "id": event["id"],
            "question": f"최근에 있었다고 들은 '{event['text']}'에 대해 어떠셨는지 자연스럽게 여쭤보세요.",
        })

    return {
        "category": "취미/근황",
        "name": "개인 근황",
        "instructions": {"min_questions": 1},
        "items": items,
    }

def _prioritize_prerequisites(items: list[dict]) -> list[dict]:
    # * requires로 다른 문항의 전제가 되는 항목(예: sis_recall이 요구하는 sis_encode)은 셔플 후에도
    # * 카테고리 맨 앞으로 당긴다 — 뒤에 남는 문항 수가 충분해야 min_gap_questions를 채울 여지가
    # * 생긴다(안 그러면 전제 항목이 우연히 맨 뒤로 밀릴 때 gap을 채울 문항이 하나도 안 남는다).

    required_ids = {item["requires"]["item"] for item in items if item.get("requires")}
    prereqs = [i for i in items if i["id"] in required_ids]
    rest = [i for i in items if i["id"] not in required_ids]
    return prereqs + rest

def _default_profile() -> dict:
    # * 어르신 히스토리가 아직 없을 경우, 빈 파일 경로 초기화

    path = ROOT / "static" / "recipient_profile.json"
    return json.loads(path.read_text()) if path.exists() else {}


def build_question_bank(profile: dict | None) -> dict:
    # * 통화(recipient)별로 state["_question_bank"]에 정보 추가
    # * 안부 묻기로 시작 - 취미/근황을 다른 임상 카테고리보다 먼저 오게 처리

    profile = profile if profile is not None else _default_profile()
    questions = dict(BASE_QUESTIONS)
    block = _profile_to_block(profile)
    if block["items"]:
        questions = {"recipient_profile": block, **questions}

    # BASE_QUESTIONS는 모듈 전역이라 공유됨 — items를 직접 shuffle하면 다른 통화에도 영향을
    # 주니, 통화마다 새 리스트로 복사하면서 섞는다. 안 그러면 첫 질문이 매번 questions.json
    # 순서 그대로 고정돼서, 카테고리 안 문항 다양성이 사실상 없다.
    questions = {
        key: {**b, "items": _prioritize_prerequisites(random.sample(b["items"], len(b["items"])))}
        for key, b in questions.items()
    }

    categories = [b["category"] for b in questions.values()]
    category_items = {b["category"]: b["items"] for b in questions.values()}
    category_min_questions = {b["category"]: b["instructions"]["min_questions"] for b in questions.values()}
    all_question_ids = [item["id"] for items in category_items.values() for item in items]
    event_ids = [event["id"] for event in profile.get("recent_events", [])] # 근황 정보 - 나중에 지울 때 필요
    hobby_names = list(profile.get("hobbies", [])) # 취미 - 나중에 지울 때 필요 (hobby는 id가 없어 이름 자체가 식별자)

    return {
        "questions": questions,
        "categories": categories,
        "category_items": category_items,
        "category_min_questions": category_min_questions,
        "all_question_ids": all_question_ids,
        "event_ids": event_ids,
        "hobby_names": hobby_names,
    }

def _find_question(questions: dict, item_id: str) -> str | None:
    # * id로 질문 조회(예: pending_retry 질문 받아오거나, 질문 순서 제약 걸려있을 때 어떤 질문인지 알고싶을 때)

    for block in questions.values():
        for item in block["items"]:
            if item["id"] == item_id:
                return item["question"]
    return None

def build_question_bank_text(questions: dict) -> str:
    # * 질문 목록 텍스트화

    lines = [
        "다음은 각 영역에서 실제로 사용할 정해진 질문 목록입니다. "
        "자연스러운 대화 흐름에 맞게 표현을 다듬어도 되지만, 의미는 그대로 유지하세요.",
        "",
    ]
    for block in questions.values():
        min_q = block["instructions"]["min_questions"]
        lines.append(f"[{block['category']}] (최소 {min_q}문항)")
        for item in block["items"]:
            note = f" ({item['note']})" if item.get("note") else ""
            req = item.get("requires")
            if req:
                base_q = _find_question(questions, req["item"])
                note += (
                    f" (반드시 \"{base_q}\" 질문 이후 최소 {req['min_gap_questions']}개의 "
                    "다른 질문을 거친 뒤에 물어보세요)"
                )
            lines.append(f"  - [{item['id']}] {item['question']}{note}")
        lines.append("")
    return "\n".join(lines)

def build_full_instructions(question_bank: dict) -> str:
    # * 최종 instruction(프롬프트 + 질문 텍스트)

    return SYSTEM_PROMPT + "\n\n" + build_question_bank_text(question_bank["questions"])

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

    # 통화가 조기 종료(어르신이 먼저 끊음, 위급상황 등)되면 일부 카테고리는 아예 못 다룰 수
    # 있다 — completed_categories에 없는 카테고리를 "미확인"으로 남겨 Spring/프론트까지
    # 흘려보낸다(채점엔 반영 안 함, 사람이 보고 판단하라는 표시 용도).
    all_categories = state["_question_bank"]["categories"]
    incomplete_categories = [
        c for c in all_categories if c not in state["completed_categories"]
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
            "filed_at": time.time(),
        }, ensure_ascii=False) + "\n")

def write_minimal_call_result(recipient_id, status: str) -> None:
    # * state 없이(발신 실패, 무응답/통화중 콜백처럼 통화가 시작도 못 한 경우) 빈 값으로 최소 결과만 기록한다.

    CALL_RESULT_OUTBOX.parent.mkdir(parents=True, exist_ok=True)
    with open(CALL_RESULT_OUTBOX, "a") as f:
        f.write(json.dumps({
            "id": f"CR-{uuid.uuid4().hex[:12]}",
            "recipient_id": recipient_id,
            "call_log": {"started_at": None, "ended_at": spring_timestamp(), "status": status},
            "call_log_entries": [], "logic_data": {}, "emergencies": [], "profile_updates": [],
            "filed_at": time.time(),
        }, ensure_ascii=False) + "\n")

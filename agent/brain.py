
# * 프롬프트, 질문은행, 툴 구현, 통화 state.
# * 데이터를 가지고 "무엇을 물어보고, 어떻게 판단하고, 통화 결과를 어떤 모양으로 남길지"만 다룬다
# ! 네트워킹(FastAPI, WebSocket, HTTP 클라이언트), 데이터 주고받기 x

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agent.prompts import SYSTEM_PROMPT, GREETING_PROMPT





log = logging.getLogger("brain")

ROOT = Path(__file__).resolve().parent.parent

# ? inbox/outbox부분 로직 검토 필요
RUNTIME_DIR = ROOT / "runtime" # worker.py가 드레인하는 append 로그/원장류 임시 파일 전용 폴더
GUARDIAN_ALERTS_FILE = RUNTIME_DIR / "alerts" / "inbox.jsonl" # 위급_조기종료 시 여기 한 줄 추가
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
        "_last_agent_utterance": None,     # 직전 턴에 에이전트가 실제로 말한 문장 — 다음 어르신 답변과 페어링
        "_last_agent_utterance_at": None,  # 위 발화가 시작된 시각(ISO, tz 없음) — asked_at으로 씀
        "_call_started_at": None,  # browser_ws/call_stream_ws가 세션 열 때 채움 — call_log.started_at
    }

# --------------------------------------------------------------------------
# 수신자 맞춤형 질문 은행 생성
# --------------------------------------------------------------------------

BASE_QUESTIONS: dict = json.loads((ROOT / "static" / "questions.json").read_text())
JUDGMENTS: list[str] = ["양호", "우려", "알수없음", "위급"]

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

    categories = [b["category"] for b in questions.values()]
    category_items = {b["category"]: b["items"] for b in questions.values()}
    all_question_ids = [item["id"] for items in category_items.values() for item in items]
    event_ids = [event["id"] for event in profile.get("recent_events", [])] # 근황 정보 - 나중에 지울 때 필요
    hobby_names = list(profile.get("hobbies", [])) # 취미 - 나중에 지울 때 필요 (hobby는 id가 없어 이름 자체가 식별자)

    return {
        "questions": questions,
        "categories": categories,
        "category_items": category_items,
        "all_question_ids": all_question_ids,
        "event_ids": event_ids,
        "hobby_names": hobby_names,
    }

def build_tools(question_bank: dict) -> list[dict]:
    # * 툴 스키마

    categories = question_bank["categories"]
    all_question_ids = question_bank["all_question_ids"]
    event_ids = question_bank["event_ids"] or ["_none"]  # 빈 enum은 JSON Schema 위반이라 더미로 채움
    hobby_names = question_bank["hobby_names"] or ["_none"]

    return [
    {
        "type": "function",
        "name": "checkin_start",
        "description": "웰니스 체크인 통화를 시작합니다. 인사 후 첫 질문으로 넘어가기 전 가장 먼저 호출하십시오.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "type": "function",
        "name": "log_answer_analysis",
        "description": "어르신의 답변을 분석하고 결과를 저장합니다. 어르신이 질문에 실질적으로 답했을 때만 호출하십시오. 호출 결과로 그 카테고리에서 아직 안 물어본 질문 목록과 다음에 할 일이 안내되니 그대로 따르십시오. 통화 종료 의사가 감지된 경우에는 이 함수 대신 end_call을 호출하십시오. 낙상, 자해, 응급 위험 신호가 감지된 경우에는 이 함수와 별도로 flag_emergency도 함께 호출하십시오.",
        "parameters": {
            "type": "object",
            "properties": {
            "category": {
                "type": "string",
                "enum": categories
            },
            "question_id": {
                "type": "string",
                "enum": all_question_ids,
                "description": "방금 어르신께 여쭤본 질문의 id (질문은행에 [id]로 표시된 값)"
            },
            "assessment": {
                "type": "string",
                "enum": JUDGMENTS
            },
            "reason": {
                "type": "string",
                "description": "판단 근거를 한 문장으로 요약"
            }
            },
            "required": ["category", "question_id", "assessment", "reason"]
        }
    },
    {
        "type": "function",
        "name": "double_check",
        "description": "어르신의 답변 의도가 불명확하거나 질문과 무관해 정상적으로 이해할 수 없을 때 호출합니다. 호출 결과가 계속 다시 물어보라고 하면 그렇게 하고, 이제 그만 포기하고 넘어가라고 안내하면 log_answer_analysis(assessment=\"알수없음\")를 호출하십시오.",
        "parameters": {
            "type": "object",
            "properties": {
            "category": {
                "type": "string",
                "enum": categories
            },
            "question_id": {
                "type": "string",
                "enum": all_question_ids,
                "description": "이해가 안 돼서 다시 물어보려는 질문의 id"
            }
            },
            "required": ["category", "question_id"]
        }
    },
    {
        "type": "function",
        "name": "flag_emergency",
        "description": "낙상, 자해, 급성 통증, 도움 요청 등 응급 상황을 암시하는 명시적 또는 암묵적 신호가 감지되면 즉시 호출하십시오. 확신이 없어도 의심되는 즉시 호출하는 것을 우선하십시오.",
        "parameters": {
            "type": "object",
            "properties": {
            "signal": {
                "type": "string",
                "description": "감지된 위험 신호에 대한 짧은 설명"
            },
            "severity": {
                "type": "string",
                "enum": ["low", "medium", "high"]
            }
            },
            "required": ["signal", "severity"]
        }
    },
    {
        "type": "function",
        "name": "update_recipient_profile",
        "description": "어르신이 질문은행에 없던 새 취미/근황을 언급하면 action=\"add\"로 기록하십시오. "
                       "기존에 기록된 취미나 최근 특이사항(recent_events)에 대해 어르신이 '그건 이제 "
                       "안 해요/끝났다/지난 일이다'처럼 더 이상 유효하지 않다는 뉘앙스를 보이면, 그 자리에서 "
                       "다시 묻지 말고 action=\"remove\"로 그 항목을 제거하십시오 — 취미는 kind=\"hobby\"와 "
                       "함께 hobby_name을, 최근 특이사항은 kind=\"event\"와 함께 event_id를 넣으십시오. "
                       "이 기록은 다음 통화부터 반영되며, 지금 진행 중인 통화의 질문 흐름에는 영향을 주지 "
                       "않습니다.",
        "parameters": {
            "type": "object",
            "properties": {
            "action": {"type": "string", "enum": ["add", "remove"]},
            "kind": {"type": "string", "enum": ["hobby", "event"]},
            "text": {
                "type": "string",
                "description": "action=add일 때 새로 기록할 내용"
            },
            "hobby_name": {
                "type": "string",
                "enum": hobby_names,
                "description": "action=remove이고 kind=\"hobby\"일 때 제거할 취미명 (질문은행에 나온 표현 그대로)"
            },
            "event_id": {
                "type": "string",
                "enum": event_ids,
                "description": "action=remove이고 kind=\"event\"일 때 제거할 recent_event의 id (질문은행에 [id]로 표시된 값)"
            },
            "reason": {
                "type": "string",
                "description": "판단 근거를 한 문장으로 요약"
            }
            },
            "required": ["action", "kind"]
        }
    },
    {
        "type": "function",
        "name": "end_call",
        "description": "다음 두 경우에 호출하십시오: (1) 어르신이 '됐어요', '끊을게요'처럼 명시적 종료 의사를 밝혔을 때 (2) 다섯 카테고리를 모두 종료 조건(good/unknown 판정 또는 해당 카테고리 질문 소진)에 따라 마치고, 미뤄뒀던 unknown 질문 재시도까지 log_answer_analysis의 next_step 안내대로 끝냈을 때. 카테고리나 재시도가 아직 남아있는데 질문 하나에 답만 한 경우는 호출하지 마십시오.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }
    ]

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

_KOREAN_WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]

def current_date_context() -> str:
    # * 실제 날짜 정답
    
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    weekday = _KOREAN_WEEKDAYS[now.weekday()]
    return (
        f"오늘은 {now.year}년 {now.month}월 {now.day}일 {weekday}요일입니다(한국 시간 기준). "
        "지남력 관련 질문(오늘 요일/몇 월/올해 몇 년도)의 답을 판단할 때는 반드시 이 날짜를 "
        "기준으로 정확하게 비교하세요. 정답을 어르신에게 절대로 노출하지 마세요."
    )

def build_full_instructions(question_bank: dict) -> str:
    # * 최종 instruction(프롬프트 + 날짜 정답 + 질문 텍스트 한 번에 합치기)

    return (
        SYSTEM_PROMPT + "\n\n" + current_date_context() + "\n\n"
        + build_question_bank_text(question_bank["questions"])
    )

# --------------------------------------------------------------------------
# 툴 실행
# --------------------------------------------------------------------------

def checkin_start(state: dict) -> str:
    return json.dumps({"ok": True}, ensure_ascii=False)

def _gap_satisfied(state: dict, item: dict) -> bool:
    # * requires(min_gap_questions) 있는 항목은 모델이 턴 수를 카운트하게 두지 않고 asked_order로 직접 확인한다.

    req = item.get("requires")
    if not req:
        return True
    order = state["asked_order"]
    if req["item"] not in order:
        return False
    gap = len(order) - order.index(req["item"]) - 1
    return gap >= req["min_gap_questions"]

def log_answer_analysis(state: dict, category: str, question_id: str, assessment: str, reason: str) -> str:
    # * 답변 판정을 기록하고, 그 카테고리에서 다음에 뭘 물어야 할지(next_step)를 계산해 돌려준다.

    category_items = state["_question_bank"]["category_items"]

    state["logic_data"][category] = {"judgment": assessment, "reason": reason}
    state["asked_order"].append(question_id)

    # 재시도(버퍼에 있던 질문)였다면 이번 결과와 상관없이 버퍼에서 뺀다 — 재시도는 한 번만.
    state["pending_retry"] = [
        p for p in state["pending_retry"]
        if not (p["category"] == category and p["question_id"] == question_id)
    ]

    asked = state["asked_questions"].setdefault(category, set())
    asked.add(question_id)
    remaining = [
        item["question"] for item in category_items.get(category, [])
        if item["id"] not in asked and _gap_satisfied(state, item)
    ]
    random.shuffle(remaining)  # questions.json 순서대로 매번 똑같이 물어보지 않게

    # sis_encode 같은 안내문(type: statement)은 good이어도 카테고리를 끝내면 안 된다 — 뒤이은 실제 문항이 스킵됨.
    asked_item = next((i for i in category_items.get(category, []) if i["id"] == question_id), None)
    is_statement = bool(asked_item and asked_item.get("type") == "statement")

    # unknown은 바로 재질문하지 않고 pending_retry에 담아 한 바퀴 돈 뒤 다시 묻는다 — 헷갈림 방지.
    if (is_statement or assessment in ("우려", "위급")) and remaining:
        why = "방금 건 채점 대상이 아닌 안내문일 뿐" if is_statement else f"'{category}'는 아직 good이 아닙니다"
        return json.dumps({
            "ok": True,
            "category_status": "진행중",
            "remaining_questions_in_category": remaining,
            "next_step": f"{why}. remaining_questions_in_category 중 하나로 이어서 물어보세요"
                         "(같은 질문 반복 금지).",
        }, ensure_ascii=False)

    if assessment == "알수없음":
        state["pending_retry"].append({"category": category, "question_id": question_id})

    # 카테고리가 지금 닫혀도, 아직 안 물어본 gap-conditional 항목(sis_recall 등)은 pending_retry에 담아 끝에 다시 묻는다.
    already_queued = {(p["category"], p["question_id"]) for p in state["pending_retry"]}
    for item in category_items.get(category, []):
        if item["id"] not in asked and item.get("requires") and (category, item["id"]) not in already_queued:
            state["pending_retry"].append({"category": category, "question_id": item["id"]})

    state["completed_categories"].add(category)
    all_categories = state["_question_bank"]["categories"]
    pending_categories = [c for c in all_categories if c not in state["completed_categories"]]

    if pending_categories:
        next_step = (
            f"'{category}' 카테고리는 끝났습니다. 쿠션어로 부드럽게 운을 뗀 뒤 다음 카테고리 "
            f"'{pending_categories[0]}'로 넘어가세요(예: '이제 {pending_categories[0]} 관련해서 "
            "좀 여쭤봐도 될까요?')."
        )
    elif state["pending_retry"]:
        retry = state["pending_retry"][0]
        retry_question = _find_question(state["_question_bank"]["questions"], retry["question_id"])
        next_step = (
            "모든 카테고리를 한 바퀴 돌았습니다. 다만 답을 알 수 없었던 질문이 남아있으니 "
            f"'{retry['category']}' 카테고리의 질문을 자연스럽게 한 번 더 여쭤보세요: \"{retry_question}\""
        )
    else:
        next_step = "모든 카테고리와 재시도까지 마쳤습니다. 짧게 마무리 인사를 하고 end_call을 호출하세요."

    return json.dumps({
        "ok": True,
        "category_status": "완료",
        "next_step": next_step,
    }, ensure_ascii=False)

DOUBLE_CHECK_LIMIT = 2  # 이 횟수를 넘기면 계속 되묻지 말고 unknown으로 넘어가게 한다

def double_check(state: dict, category: str, question_id: str) -> str:
    # * 답변 의도가 불명확할 때 호출됨 — DOUBLE_CHECK_LIMIT 넘게 반복되면 그만 포기하고 unknown으로 넘어가라고 안내한다.
    
    counts = state["double_check_counts"]
    counts[question_id] = counts.get(question_id, 0) + 1

    if counts[question_id] > DOUBLE_CHECK_LIMIT:
        return json.dumps({
            "ok": True,
            "next_step": f"이 질문은 이미 {counts[question_id] - 1}번 다시 여쭤봤습니다. 더 반복하지 "
                         f"말고 log_answer_analysis(category=\"{category}\", "
                         f"question_id=\"{question_id}\", assessment=\"알수없음\")을 호출한 뒤 "
                         "다음으로 넘어가세요.",
        }, ensure_ascii=False)

    return json.dumps({
        "ok": True,
        "next_step": "같은 질문을 더 쉬운 말로 풀어서 다시 여쭤보세요.",
    }, ensure_ascii=False)

def flag_emergency(state: dict, signal: str, severity: str) -> str:
    # * 낙상/자해 등 위험 신호를 기록만 해둔다 — 실제 알림 파일 기록은 end_call에서 emergencies 유무로 처리.
    state["emergencies"].append({
        "signal": signal,
        "severity": severity,
        "flagged_at": time.time(),
    })
    return json.dumps({"ok": True}, ensure_ascii=False)


def update_recipient_profile(
    state: dict, action: str, kind: str,
    text: str | None = None, hobby_name: str | None = None, event_id: str | None = None,
    reason: str | None = None,
) -> str:
    # * 이번 통화의 질문 흐름은 안 건드리고 profile_updates에만 쌓아둔다 — 다음 통화 profile부터 반영됨.

    state["profile_updates"].append({
        "action": action, "kind": kind, "text": text, "hobby_name": hobby_name,
        "event_id": event_id, "reason": reason,
    })
    return json.dumps({"ok": True}, ensure_ascii=False)

def new_alert_id() -> str:
    # TODO 위급_조기종료면 보호자 알림을 파일로 남긴다(실제 서버로 전송해야함).
    #A-1000, A-1001, ... — numbered by how many alerts are already filed."""

    count = 0
    if GUARDIAN_ALERTS_FILE.exists():
        count = sum(1 for line in GUARDIAN_ALERTS_FILE.read_text().splitlines() if line.strip())
    return f"A-{1000 + count}"

def end_call(state: dict) -> str:
    # * 통화 종료

    emergencies = state["emergencies"]
    reason = "위급_조기종료" if emergencies else "정상_종료"

    if emergencies:
        GUARDIAN_ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(GUARDIAN_ALERTS_FILE, "a") as f:
            f.write(json.dumps({
                "id": new_alert_id(),
                "alert": " / ".join(e["signal"] for e in emergencies),
                "logic_data": state["logic_data"],
                "metadata": state.get("_metadata") or {},  # recipient_id/phone_number from POST /call
                "filed_at": time.time(),
            }, ensure_ascii=False) + "\n")

    return json.dumps({
        "call_ended": True,
        "reason": reason,
        "next_step": "지금 바로 다음 답변으로 부드러운 작별 인사를 건네고 통화를 마치세요.",
    }, ensure_ascii=False)

def write_call_result_outbox(state: dict, status: str) -> None:
    # * Spring 전송용 원본 데이터를 로컬에 남긴다(실제 POST는 worker.py가 폴링)
    # * 브라우저 데모는 skip

    metadata = state.get("_metadata") or {}
    recipient_id = metadata.get("recipient_id")
    if recipient_id is None:
        return

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


def run_tool(state: dict, name: str, args: dict) -> str:
    # * 툴 실행
    
    log.info("tool call: %s(%s)", name, json.dumps(args, ensure_ascii=False))
    try:
        if name == "checkin_start":
            result = checkin_start(state)
        elif name == "log_answer_analysis":
            result = log_answer_analysis(state, **args)
        elif name == "double_check":
            result = double_check(state, **args)
        elif name == "flag_emergency":
            result = flag_emergency(state, **args)
        elif name == "update_recipient_profile":
            result = update_recipient_profile(state, **args)
        elif name == "end_call":
            result = end_call(state)
        else:
            result = json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)
    except Exception as exc:
        log.exception("tool %s failed", name)
        result = json.dumps({"error": str(exc)}, ensure_ascii=False)

    return result

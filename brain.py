"""웰니스 체크인 에이전트의 "뇌" — 프롬프트, 질문은행, 툴 구현, 통화 state.

server.py가 xAI Realtime/claw-ops/Spring과 실제로 데이터를 주고받는 쪽이라면,
여기는 그 데이터를 가지고 "무엇을 물어보고, 어떻게 판단하고, 통화 결과를 어떤
모양으로 남길지"만 다룬다 — 네트워킹(FastAPI, WebSocket, HTTP 클라이언트)은
이 파일에 전혀 없다. server.py가 이 모듈을 가져다 쓰는 방향으로만 의존한다
(반대 방향 의존성 없음 — 나중에 다른 transport를 붙이거나 단위 테스트하기 쉽게).
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger("brain")

GUARDIAN_ALERTS_FILE = Path("alerts/inbox.jsonl")      # 위급_조기종료 시 여기 한 줄 추가
CALL_RESULT_OUTBOX = Path("call_results_outbox.jsonl")  # 통화 종료 시 여기 한 줄 추가 (worker.py가 Spring으로 드레인)


def spring_timestamp(dt: datetime | None = None) -> str:
    """Spring의 LocalDateTime과 맞는 tz 없는 ISO-8601 문자열(예: "2026-06-18T10:00:00").

    한국 시간 기준으로 값은 만들되, Java LocalDateTime엔 타임존 개념이 없어서 tzinfo를
    실어 보내면(예: "+09:00") Spring 쪽 파싱이 깨진다 — 그래서 마지막에 벗겨낸다.
    """
    dt = dt or datetime.now(ZoneInfo("Asia/Seoul"))
    return dt.replace(tzinfo=None).isoformat(timespec="seconds")


#실제 전화 연결이되면 생성, 툴 호출될 때마다 retrieve되거나 값이 업데이트됨
def new_call_state(question_bank: dict) -> dict:
    """One call's state — lives only as long as its websocket connection

    (a realtime call has no cascade-style shared session to persist across
    separate HTTP requests, so there's nothing to register or look up by id).

    question_bank는 이 통화의 profile(취미/근황)을 반영해 통화 시작 시 한 번
    build_question_bank()로 만들어져 여기 실려온다 — log_answer_analysis 등이
    더 이상 모듈 전역 CATEGORIES/CATEGORY_ITEMS를 보지 않고 이걸 본다.
    """
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
        "profile_updates": [],    # [{"action","kind","text"|"event_id","reason"}, ...] — filled by
                                   # update_recipient_profile, pushed to Spring at call end, next-call-only
        "call_log_entries": [],   # [{"sequence","question","answer","asked_at"}, ...] for Spring's call_log_entries
        "_last_agent_utterance": None,     # 직전 턴에 에이전트가 실제로 말한 문장 — 다음 어르신 답변과 페어링
        "_last_agent_utterance_at": None,  # 위 발화가 시작된 시각(ISO, tz 없음) — asked_at으로 씀
        "_call_started_at": None,  # call_ws/claw_stream_ws가 세션 열 때 채움 — call_log.started_at
    }


SYSTEM_PROMPT = (
    "당신은 독거노인 어르신들의 안부와 건강 상태를 확인하는 전화 통화를 진행하는 "
    "다정하고 차분한 상담원입니다. 통화는 음성으로만 이루어지므로 마크다운이나 목록 없이 "
    "짧고 자연스러운 한두 문장으로만 말하세요. \n\n"

    "greeting에 대한 어르신 답변이 온 후, "
    "통화는 취미/근황, 우울, 불안, 신체 건강, 인지 기능, 복약/식사 여섯 가지 영역을 순서대로 "
    "다루며, 각 영역마다 정해진 질문들을 자연스러운 대화 흐름 속에서 물어봅니다. 취미/근황은 "
    "임상 평가가 아니라 자연스러운 안부 대화이니, 인사 직후 편안한 분위기로 시작해 본 질문(우울 "
    "이하)으로 자연스럽게 넘어가는 다리 역할로 다루세요 — 그래도 log_answer_analysis 추적과 "
    "카테고리 완료 규칙은 다른 영역과 똑같이 적용됩니다. 어르신이 질문에 실질적으로 답했다면 "
    "그 답변을 해당 영역의 임상적 관점에서 해석하여 category, question_id(방금 물어본 질문의 "
    "id — 질문은행에 [id]로 표시되어 있습니다), assessment, reason 값을 채운 뒤 log_answer_analysis를 "
    "호출하세요. assessment는 답변 내용이 뚜렷한 문제를 시사하지 않으면 good, 경미하거나 애매한 "
    "우려가 있으면 concern, 판단하기 어려우면 unknown, 즉각적인 위험이 느껴지면 urgent로 판단합니다.\n\n"

    "카테고리를 넘어갈 때는 곧바로 다음 질문으로 들어가지 말고, 매번 쿠션어로 부드럽게 운을 뗀 "
    "뒤 시작하세요. 예를 들어 '이제 건강 관련해서 좀 여쭤봐도 될까요?', '괜찮으시다면 마음 상태도 "
    "여쭤볼게요' 같은 식입니다. 카테고리마다 다른 표현을 쓰고, 매번 기계적으로 똑같은 문구를 "
    "반복하지 마세요.\n\n"

    "어떤 질문을 이미 물어봤는지는 당신의 기억이 아니라 log_answer_analysis 호출 결과가 정답입니다 "
    "— 결과에 담긴 remaining_questions_in_category(그 카테고리에서 아직 안 물어본 질문 목록)와 "
    "next_step 지시를 그대로 따르세요. category_status가 완료로 나오면 그 카테고리는 더 묻지 말고 "
    "다음 카테고리로 넘어가고, 진행중으로 나오면 remaining_questions_in_category 중 하나로 이어서 "
    "확인하세요. 같은 질문을 다시 묻지 마세요.\n\n"

    "assessment가 concern이나 urgent면 같은 카테고리에서 계속 파고들지만, unknown(판단하기 "
    "어려움)이면 그 자리에서 바로 다시 묻지 마세요 — 짧게 반응만 하고 다음 카테고리로 넘어가세요. "
    "그 질문은 서버가 기억해뒀다가, 모든 카테고리를 한 바퀴 다 돈 뒤 next_step으로 다시 여쭤보라고 "
    "안내해 줄 것입니다. 그때는 안내된 질문을 자연스럽게 한 번 더 여쭤보세요.\n\n"

    "질문은행에 '채점 대상 아님'이라고 표시된 안내성 문장(예: 지남력 검사에서 단어 세 개를 "
    "안내하는 부분)도 말한 직후 반드시 log_answer_analysis를 호출해 question_id를 기록하세요 "
    "— assessment는 good, reason은 무엇을 말했는지(예: 사용한 단어 세 개) 한 줄로 남기면 됩니다. "
    "그래야 같은 안내를 다시 반복하지 않고, 그 단어를 다시 여쭤보기까지 필요한 간격도 서버가 "
    "추적해서 remaining_questions_in_category에 반영해 줍니다.\n\n"

    "어르신의 답변이 질문과 무관하거나 의도를 파악하기 어렵다면 log_answer_analysis를 호출하지 "
    "말고 대신 double_check(category, question_id)을 호출한 뒤, 같은 질문을 더 쉬운 말로 풀어서 "
    "다시 여쭤보세요. double_check 결과가 이제 그만 포기하고 넘어가라고 안내하면, 억지로 계속 "
    "되묻지 말고 그 지시대로 log_answer_analysis를 unknown으로 호출한 뒤 다음으로 넘어가세요. "
    "인지 기능 관련 질문에서는 어르신이 답을 맞혔는지 틀렸는지를 절대로 직접 알려주지 마세요. "
    "정답 여부와 상관없이 '네, 잘 말씀해주셨어요' 같은 중립적이고 격려하는 반응만 보이고 "
    "다음 질문으로 자연스럽게 넘어가세요.\n\n"

    "어르신의 발화 중 낙상, 자해, 극심한 통증, 도움을 요청하는 뉘앙스가 조금이라도 느껴지면 "
    "확신이 없더라도 즉시 flag_emergency를 호출하세요. 이는 다른 판단과 별개로 이루어지며, "
    "같은 턴에 log_answer_analysis와 함께 호출해도 됩니다.\n\n"

    "'취미/근황' 대화 중 어르신이 질문은행에 없던 새로운 취미나 최근 일을 스스로 언급하면, "
    "update_recipient_profile(action=\"add\")로 짧게 기록해두세요(대화를 끊지 말고 자연스럽게 "
    "반응한 뒤 호출). 반대로 이미 기록된 최근 특이사항에 대해 어르신이 '그건 이제 끝났다', "
    "'벌써 지난 일이다'처럼 더는 유효하지 않다는 뉘앙스를 보이면, 그 자리에서 다시 캐묻지 말고 "
    "update_recipient_profile(action=\"remove\")로 그 항목을 제거하세요. 이 기록들은 이번 통화의 "
    "질문 흐름에는 영향을 주지 않고 다음 통화부터 반영됩니다.\n\n"

    "어르신이 질문에 짜증을 내거나 대화를 귀찮아하는 기색을 보이더라도 부드럽게 공감하며 "
    "달래되, 아직 다루지 않은 필수 영역의 질문은 건너뛰지 말고 표현 방식만 더 가볍고 짧게 "
    "바꿔서 계속 진행하세요.\n\n"

    "다섯 카테고리를 모두 위 규칙대로 마치고, 미뤄뒀던 unknown 질문 재시도까지 끝났다면(log_answer_analysis의 "
    "next_step이 그렇게 안내합니다), 짧게 마무리 인사를 건넨 뒤 end_call을 호출해 통화를 종료하세요. "
    "다만 어르신이 '이제 됐어요', '끊을게요'처럼 명시적으로 통화 종료 의사를 밝히면 남은 카테고리나 "
    "재시도가 있어도 다른 어떤 질문도 더 하지 말고 그 즉시 end_call을 호출하세요."
)
GREETING_PROMPT = (
    "통화가 연결되면 가장 먼저 부드럽고 편안한 목소리로 본인이 누구인지와 "
    "어떤 목적으로 전화드렸는지를 짧게, 격식 없이 설명하세요. "
    "예를 들어 '안녕하세요, 어르신 잘 지내고 계신지 안부 여쭤보려고 전화드렸어요' 정도의 "
    "자연스러운 톤의 1~2 문장이면 됩니다. "
)

# 우울 <-> "depression_gds5" 등 실제 질문 목록. 리얼타임 모델은 스스로 질문을
# 골라야 하므로, 전체 질문은행을 텍스트로 풀어 SYSTEM_PROMPT에 통째로 붙여 넣는다.
# 임상 카테고리(우울/불안/신체 건강/인지 기능/복약·식사)는 어르신이 바뀌어도 고정이라
# 모듈 레벨에 한 번만 로드 — "취미/근황"만 통화별로 달라서 build_question_bank()가
# 매 통화 이걸 기반으로 합쳐서 만든다.
BASE_QUESTIONS: dict = json.loads(Path("static/questions.json").read_text())

JUDGMENTS: list[str] = ["양호", "우려", "알수없음", "위급"]


def _profile_to_block(profile: dict) -> dict:
    """profile(hobbies/recent_events, Spring이 POST /call에 실어 보냄)을

    questions.json의 카테고리와 똑같은 모양(category/instructions/items)으로 바꿔,
    다른 임상 카테고리와 완전히 같은 방식(log_answer_analysis 추적, 반복 방지,
    완료 판정)으로 다뤄지게 한다. 실제 문장은 모델이 자연스럽게 다듬도록
    "여쭤보세요" 지시문 형태로만 준다.

    recent_events의 id는 profile이 이미 갖고 있는 값을 그대로 question_id로
    재사용한다 — update_recipient_profile(action="remove")가 이 id로 항목을
    가리켜야 하는데, 매 통화 새로 hobby_1/event_1처럼 찍으면 통화마다 id가
    달라져서 Spring의 원본 recent_event.id와 어긋난다.
    """
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
    """Spring이 profile을 안 보낸 경우(브라우저 데모 /ws/call)의 폴백.

    static/recipient_profile.json은 실제 배포에선 어르신별로 달라지고 개인정보라
    gitignore 대상 — 로컬 데모/개발용 고정 프로필로만 쓰인다.
    """
    path = Path("static/recipient_profile.json")
    return json.loads(path.read_text()) if path.exists() else {}


def build_question_bank(profile: dict | None) -> dict:
    """통화 하나에 쓸 질문은행을 만든다 — QUESTIONS/CATEGORIES/CATEGORY_ITEMS/

    ALL_QUESTION_IDS를 예전엔 모듈 전역으로 한 번만 고정했는데, 이제 통화(recipient)마다
    profile이 달라지니 매 통화 새로 만들어서 state["_question_bank"]에 담아 쓴다.

    취미/근황을 다른 임상 카테고리보다 먼저 오게 앞에 꽂는다 — 인사 직후 부드러운
    안부 대화로 시작해서 자연스럽게 본 질문으로 넘어가려는 의도(딕셔너리는 삽입
    순서를 유지하므로 이 순서가 곧 categories 순서, 곧 진행 순서가 된다).
    """
    profile = profile if profile is not None else _default_profile()
    questions = dict(BASE_QUESTIONS)
    block = _profile_to_block(profile)
    if block["items"]:
        questions = {"recipient_profile": block, **questions}

    categories = [b["category"] for b in questions.values()]
    category_items = {b["category"]: b["items"] for b in questions.values()}
    all_question_ids = [item["id"] for items in category_items.values() for item in items]
    # update_recipient_profile(action="remove")의 event_id enum용 — hobby는 삭제 대상이 아님.
    event_ids = [event["id"] for event in profile.get("recent_events", [])]

    return {
        "questions": questions,
        "categories": categories,
        "category_items": category_items,
        "all_question_ids": all_question_ids,
        "event_ids": event_ids,
    }


def build_tools(question_bank: dict) -> list[dict]:
    categories = question_bank["categories"]
    all_question_ids = question_bank["all_question_ids"]
    event_ids = question_bank["event_ids"] or ["_none"]  # 빈 enum은 JSON Schema 위반이라 더미로 채움

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
                       "기존에 기록된 최근 특이사항(recent_events)에 대해 어르신이 '그건 이제 끝났다/지난 "
                       "일이다'처럼 더 이상 유효하지 않다는 뉘앙스를 보이면, 그 자리에서 다시 묻지 말고 "
                       "action=\"remove\"로 그 항목을 제거하십시오. 이 기록은 다음 통화부터 반영되며, "
                       "지금 진행 중인 통화의 질문 흐름에는 영향을 주지 않습니다.",
        "parameters": {
            "type": "object",
            "properties": {
            "action": {"type": "string", "enum": ["add", "remove"]},
            "kind": {"type": "string", "enum": ["hobby", "event"]},
            "text": {
                "type": "string",
                "description": "action=add일 때 새로 기록할 내용"
            },
            "event_id": {
                "type": "string",
                "enum": event_ids,
                "description": "action=remove일 때 제거할 recent_event의 id (질문은행에 [id]로 표시된 값)"
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

# --------------------------------------------------------------------------
# 질문은행 -> 프롬프트 텍스트
# --------------------------------------------------------------------------


def _find_question(questions: dict, item_id: str) -> str | None:
    for block in questions.values():
        for item in block["items"]:
            if item["id"] == item_id:
                return item["question"]
    return None


def build_question_bank_text(questions: dict) -> str:
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
    """지남력(sis_day/month/year) 문항을 채점할 때 모델이 오늘 날짜를 스스로 추측하지

    않게 실제 날짜를 박아 넣는다 — 실시간 음성 모델은 학습 시점 지식만 갖고 있어서
    이게 없으면 '오늘이 무슨 요일인지' 자체를 모른 채로 채점하게 된다.
    """
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    weekday = _KOREAN_WEEKDAYS[now.weekday()]
    return (
        f"오늘은 {now.year}년 {now.month}월 {now.day}일 {weekday}요일입니다(한국 시간 기준). "
        "지남력 관련 질문(오늘 요일/몇 월/올해 몇 년도)의 답을 판단할 때는 반드시 이 날짜를 "
        "기준으로 정확하게 비교하세요. 정답을 어르신에게 절대로 노출하지 마세요."
    )


def build_full_instructions(question_bank: dict) -> str:
    """통화마다 새로 만든다 — 날짜가 바뀌어도, profile이 달라도 항상 맞게 반영되도록."""
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
    """sis_recall-style items (requires: {item, min_gap_questions}) aren't ready

    until enough OTHER questions have come between them and their prerequisite —
    tracked via asked_order instead of trusting the model to count turns itself.
    """
    req = item.get("requires")
    if not req:
        return True
    order = state["asked_order"]
    if req["item"] not in order:
        return False
    gap = len(order) - order.index(req["item"]) - 1
    return gap >= req["min_gap_questions"]


def log_answer_analysis(state: dict, category: str, question_id: str, assessment: str, reason: str) -> str:
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

    # sis_encode 같은 "채점 대상 아님" 안내문(type: statement)은 진짜 답변이 아니므로
    # good을 찍어도 카테고리를 끝내면 안 된다 — 안 그러면 그 뒤에 이어질 실제 문항들
    # (sis_day/month/year, sis_recall)이 통째로 스킵된다.
    asked_item = next((i for i in category_items.get(category, []) if i["id"] == question_id), None)
    is_statement = bool(asked_item and asked_item.get("type") == "statement")

    # concern/urgent, 그리고 방금 물은 게 statement였던 경우는 같은 카테고리에서 계속
    # 진행한다. unknown(판단 불가)은 그 자리에서 다시 묻지 않고 카테고리를 일단 마친 걸로
    # 보되, pending_retry에 담아 모든 카테고리를 한 바퀴 돈 뒤 한 번 더 여쭤보게 한다 —
    # 못 알아들은 질문을 바로 또 물으면 어르신이 더 헷갈릴 수 있어서, 대화가 자연스럽게
    # 넘어간 뒤에 다시 시도하는 편이 낫다.
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

    # sis_recall처럼 "나중에 다시 여쭤보겠다"고 예고한 gap-conditional 항목은, good이
    # 떠서 카테고리가 지금 바로 닫히더라도 그냥 버려지면 안 된다(약속을 어기게 됨) —
    # 아직 안 물어봤다면 pending_retry에 담아 통화 끝에 반드시 다시 물어보게 한다.
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
    state["emergencies"].append({
        "signal": signal,
        "severity": severity,
        "flagged_at": time.time(),
    })
    return json.dumps({"ok": True}, ensure_ascii=False)


def update_recipient_profile(
    state: dict, action: str, kind: str,
    text: str | None = None, event_id: str | None = None, reason: str | None = None,
) -> str:
    """이번 통화의 질문 흐름·enum(state["_question_bank"])은 건드리지 않는다 — 그냥

    state["profile_updates"]에 쌓아두면, 통화 끝에 Spring으로 보내는 결과에 실려서
    다음 통화의 profile부터 반영된다(설계상 결정 — 라이브 세션 스키마 변경 없음).
    """
    state["profile_updates"].append({
        "action": action, "kind": kind, "text": text, "event_id": event_id, "reason": reason,
    })
    return json.dumps({"ok": True}, ensure_ascii=False)


# 위급_조기종료면 보호자 알림을 파일로 남긴다(느린 발송은 worker.py 몫).
def new_alert_id() -> str:
    """A-1000, A-1001, ... — numbered by how many alerts are already filed."""
    count = 0
    if GUARDIAN_ALERTS_FILE.exists():
        count = sum(1 for line in GUARDIAN_ALERTS_FILE.read_text().splitlines() if line.strip())
    return f"A-{1000 + count}"


def end_call(state: dict) -> str:
    emergencies = state["emergencies"]
    reason = "위급_조기종료" if emergencies else "정상_종료"

    if emergencies:
        GUARDIAN_ALERTS_FILE.parent.mkdir(exist_ok=True)
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
    """통화 종료 시 Spring에 보낼 원본 데이터를 로컬에 남긴다 — GUARDIAN_ALERTS_FILE과

    똑같은 내구성 패턴: 실제 POST(그리고 채점 모델 호출)는 worker.py가 폴링하며
    처리하게 빼서, Spring이 잠깐 안 받아줘도 통화 종료 경로에서 데이터를 잃지 않는다.
    브라우저 데모(/ws/call)처럼 Spring이 트리거하지 않은 통화는 recipient_id가 없어
    보낼 대상이 없으니 그냥 건너뛴다.
    """
    metadata = state.get("_metadata") or {}
    recipient_id = metadata.get("recipient_id")
    if recipient_id is None:
        return

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
    """state 없이(발신 자체 실패, claw-ops 무응답/통화중 콜백처럼 통화가 시작도 못 한

    경우) 최소한의 결과만 기록한다. write_call_result_outbox와 같은 CALL_RESULT_OUTBOX에
    쓰지만, call_log_entries/logic_data/emergencies/profile_updates가 애초에 없으니
    빈 값으로 채운다.
    """
    with open(CALL_RESULT_OUTBOX, "a") as f:
        f.write(json.dumps({
            "id": f"CR-{uuid.uuid4().hex[:12]}",
            "recipient_id": recipient_id,
            "call_log": {"started_at": None, "ended_at": spring_timestamp(), "status": status},
            "call_log_entries": [], "logic_data": {}, "emergencies": [], "profile_updates": [],
            "filed_at": time.time(),
        }, ensure_ascii=False) + "\n")


def run_tool(state: dict, name: str, args: dict) -> str:
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

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dispatcher")

app = FastAPI(title="HYOPE AI voice agent")

#GUARDIAN_ALERTS_FILE = Path("alerts/inbox.jsonl")    # 위급_조기종료 시 여기 한 줄 추가


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in; "
            f"model names and voices live at https://docs.x.ai."
        )
    return value


def realtime_url() -> str:
    base = os.environ.get("XAI_REALTIME_BASE_URL", "wss://api.x.ai/v1/realtime")
    model = os.environ.get("REALTIME_MODEL", "grok-voice-think-fast-1.0")
    return f"{base}?model={model}"


def turn_detection_config() -> dict:

    return {
        "type": "server_vad",
        "threshold": float(os.environ.get("VAD_THRESHOLD", "0.5")),
        "prefix_padding_ms": int(os.environ.get("PREFIX_MS", "300")),
        "silence_duration_ms": int(os.environ.get("SILENCE_MS", "900")),
        "idle_timeout_ms": int(os.environ.get("IDLE_TIMEOUT_MS", "30000")),
    }

'''
def synthesize_speech(text: str) -> bytes:
    """xAI의 네이티브 TTS 엔드포인트(mp3 bytes 반환) — index2.html이 타이핑한 텍스트를

    실제 음성으로 바꿔 /ws/call에 "말하듯" 흘려보낼 때만 쓴다. 리얼타임 통화 자체의
    출력 음성은 이거랑 무관하게 realtime API가 직접 스트리밍한다.
    """
    resp = requests.post(
        "https://api.x.ai/v1/tts",
        headers={
            "Authorization": f"Bearer {require_env('XAI_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={"text": text, "voice_id": os.environ.get("TTS_VOICE", "ara"), "language": "auto"},
    )
    resp.raise_for_status()
    return resp.content
'''

#실제 전화 연결이되면 생성, 툴 호출될 때마다 retrieve되거나 값이 업데이트됨
def new_call_state() -> dict:
    """One call's state — lives only as long as its websocket connection

    (a realtime call has no cascade-style shared session to persist across
    separate HTTP requests, so there's nothing to register or look up by id).
    """
    return {
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
QUESTIONS: dict = json.loads(Path("static/questions.json").read_text())


def _load_recipient_profile_block() -> dict:
    """어르신의 취미/최근 2주 특이사항(static/recipient_profile.json, 실제 배포에선

    어르신별로 달라지고 개인정보라 gitignore 대상)을 questions.json의 카테고리와
    똑같은 모양(category/instructions/items)으로 바꿔, 다른 임상 카테고리와 완전히
    같은 방식(log_answer_analysis 추적, 반복 방지, 완료 판정)으로 다뤄지게 한다.
    실제 문장은 모델이 자연스럽게 다듬도록 "여쭤보세요" 지시문 형태로만 준다.
    """
    path = Path("static/recipient_profile.json")
    profile = json.loads(path.read_text()) if path.exists() else {}

    items = []
    for i, hobby in enumerate(profile.get("hobbies", []), start=1):
        items.append({
            "id": f"hobby_{i}",
            "question": f"평소 취미이신 '{hobby}', 요즘도 하고 계신지 편하게 여쭤보세요.",
        })
    for i, event in enumerate(profile.get("recent_events", []), start=1):
        items.append({
            "id": f"event_{i}",
            "question": f"최근에 있었다고 들은 '{event}'에 대해 어떠셨는지 자연스럽게 여쭤보세요.",
        })

    return {
        "category": "취미/근황",
        "name": "개인 근황",
        "instructions": {"min_questions": 1},
        "items": items,
    }


# 취미/근황을 다른 임상 카테고리보다 먼저 오게 앞에 꽂는다 — 인사 직후 부드러운
# 안부 대화로 시작해서 자연스럽게 본 질문으로 넘어가려는 의도(딕셔너리는 삽입
# 순서를 유지하므로 이 순서가 곧 CATEGORIES 순서, 곧 진행 순서가 된다).
_profile_block = _load_recipient_profile_block()
if _profile_block["items"]:
    QUESTIONS = {"recipient_profile": _profile_block, **QUESTIONS}

# 5가지 상태 확인 영역과 카테고리별 최종판단 값. CATEGORIES는 questions.json에서
# 그대로 뽑아써서 두 곳의 목록이 어긋날 일이 없게 한다. 모델이 부르는
# category/assessment 값이 곧 logic_data에 쓰는 키다.
CATEGORIES: list[str] = [block["category"] for block in QUESTIONS.values()]
JUDGMENTS: list[str] = ["양호", "우려", "알수없음", "위급"]

# category(KR) -> 그 영역의 질문 목록, id 전체 목록 — "이미 물어본 질문 다시 안 묻기"를
# 모델의 대화 기억이 아니라 서버 상태로 추적하는 데 쓴다(log_answer_analysis 참고).
CATEGORY_ITEMS: dict[str, list[dict]] = {block["category"]: block["items"] for block in QUESTIONS.values()}
ALL_QUESTION_IDS: list[str] = [item["id"] for items in CATEGORY_ITEMS.values() for item in items]

TOOLS: list[dict] = [
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
                "enum": CATEGORIES
            },
            "question_id": {
                "type": "string",
                "enum": ALL_QUESTION_IDS,
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
                "enum": CATEGORIES
            },
            "question_id": {
                "type": "string",
                "enum": ALL_QUESTION_IDS,
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
        "name": "end_call",
        "description": "다음 두 경우에 호출하십시오: (1) 어르신이 '됐어요', '끊을게요'처럼 명시적 종료 의사를 밝혔을 때 (2) 다섯 카테고리를 모두 종료 조건(good/unknown 판정 또는 해당 카테고리 질문 소진)에 따라 마치고, 미뤄뒀던 unknown 질문 재시도까지 log_answer_analysis의 next_step 안내대로 끝냈을 때. 카테고리나 재시도가 아직 남아있는데 질문 하나에 답만 한 경우는 호출하지 마십시오.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }
]

# --------------------------------------------------------------------------
# 질문은행 -> 프롬프트 텍스트
# --------------------------------------------------------------------------


def _find_question(item_id: str) -> str | None:
    for block in QUESTIONS.values():
        for item in block["items"]:
            if item["id"] == item_id:
                return item["question"]
    return None


def build_question_bank_text() -> str:
    lines = [
        "다음은 각 영역에서 실제로 사용할 정해진 질문 목록입니다. "
        "자연스러운 대화 흐름에 맞게 표현을 다듬어도 되지만, 의미는 그대로 유지하세요.",
        "",
    ]
    for block in QUESTIONS.values():
        min_q = block["instructions"]["min_questions"]
        lines.append(f"[{block['category']}] (최소 {min_q}문항)")
        for item in block["items"]:
            note = f" ({item['note']})" if item.get("note") else ""
            req = item.get("requires")
            if req:
                base_q = _find_question(req["item"])
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


def build_full_instructions() -> str:
    """세션마다 새로 만든다 — 날짜가 바뀌어도 서버를 재시작하지 않고 항상 맞게 반영되도록."""
    return SYSTEM_PROMPT + "\n\n" + current_date_context() + "\n\n" + build_question_bank_text()

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
        item["question"] for item in CATEGORY_ITEMS.get(category, [])
        if item["id"] not in asked and _gap_satisfied(state, item)
    ]
    random.shuffle(remaining)  # questions.json 순서대로 매번 똑같이 물어보지 않게

    # sis_encode 같은 "채점 대상 아님" 안내문(type: statement)은 진짜 답변이 아니므로
    # good을 찍어도 카테고리를 끝내면 안 된다 — 안 그러면 그 뒤에 이어질 실제 문항들
    # (sis_day/month/year, sis_recall)이 통째로 스킵된다.
    asked_item = next((i for i in CATEGORY_ITEMS.get(category, []) if i["id"] == question_id), None)
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
    for item in CATEGORY_ITEMS.get(category, []):
        if item["id"] not in asked and item.get("requires") and (category, item["id"]) not in already_queued:
            state["pending_retry"].append({"category": category, "question_id": item["id"]})

    state["completed_categories"].add(category)
    pending_categories = [c for c in CATEGORIES if c not in state["completed_categories"]]

    if pending_categories:
        next_step = (
            f"'{category}' 카테고리는 끝났습니다. 쿠션어로 부드럽게 운을 뗀 뒤 다음 카테고리 "
            f"'{pending_categories[0]}'로 넘어가세요(예: '이제 {pending_categories[0]} 관련해서 "
            "좀 여쭤봐도 될까요?')."
        )
    elif state["pending_retry"]:
        retry = state["pending_retry"][0]
        retry_question = _find_question(retry["question_id"])
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
                "metadata": state.get("_metadata") or {},  # recipientId/guardianPhoneNumber from POST /call
                "filed_at": time.time(),
            }, ensure_ascii=False) + "\n")

    return json.dumps({
        "call_ended": True,
        "reason": reason,
        "next_step": "지금 바로 다음 답변으로 부드러운 작별 인사를 건네고 통화를 마치세요.",
    }, ensure_ascii=False)


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
        elif name == "end_call":
            result = end_call(state)
        else:
            result = json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)
    except Exception as exc:
        log.exception("tool %s failed", name)
        result = json.dumps({"error": str(exc)}, ensure_ascii=False)

    return result

async def _pump_downstream(client_ws: WebSocket, upstream) -> None:
    """브라우저 -> 이 서버로 오는 메시지를 xAI realtime ws로 그대로 전달한다."""
    while True:
        msg = await client_ws.receive_json()
        kind = msg.get("type")

        if kind == "audio":
            await upstream.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": msg["audio"],
            }))
        elif kind == "text":
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            await upstream.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }))
            await upstream.send(json.dumps({"type": "response.create"}))
        elif kind == "hangup":
            return


async def _pump_upstream(upstream, client_ws: WebSocket, state: dict, session_id: str) -> None:
    """xAI realtime ws에서 오는 이벤트를 처리하고, 재생용 오디오/자막/칩을 브라우저로 보낸다."""
    ending = False
    # 툴 호출이 있었던 응답은 response.done 이후 우리가 곧바로 response.create로
    # 이어서 대답을 시킨다 — 그 사이에 클라이언트가 새 입력을 보내면 response.create가
    # 겹쳐 응답이 통째로 비어버린다(실제로 겪은 버그). 그래서 "이 응답이 이어질
    # 예정이다"를 추적해, 진짜로 할 말이 끝난 response.done에서만 turn_done을 보낸다.
    awaiting_continuation = False

    # 지연시간 계측: 구간 시작 시각만 들고 있다가, 그 구간을 끝내는 이벤트가 오면
    # 그 시각과의 차이를 ms로 로그에 남긴다(perf_counter라 단조 증가만 보장하면 됨).
    timing = {
        "voice_in": None,   # 어르신 발화 STT 완료 시각 — ~툴 선정까지 구간의 시작점
        "tool_called": None,  # 가장 최근 툴 호출 시각 — ~다음 답변 생성 시작까지 구간의 시작점
    }

    # 실제 대화 내용 로깅: 에이전트 발화는 delta로 쪼개져 오므로 한 턴이 끝날 때까지
    # 모았다가 한 줄로 찍는다. 어르신 발화는 STT 결과가 이미 완성된 문장으로 온다.
    agent_text_parts: list[str] = []

    def _mark_answer_started() -> None:
        """response.output_audio(.transcript).delta가 오는 시점 = LLM 답변 생성 시작."""
        if timing["tool_called"] is None:
            return
        elapsed_ms = (time.perf_counter() - timing["tool_called"]) * 1000
        log.info("[%s] 툴 호출 -> LLM 답변 생성: %.0f ms", session_id, elapsed_ms)
        timing["tool_called"] = None

    async for raw in upstream:
        event = json.loads(raw)
        etype = event.get("type", "")

        if etype == "response.created":
            awaiting_continuation = False

        elif etype == "response.output_audio.delta":
            _mark_answer_started()
            await client_ws.send_json({"type": "audio", "audio": event["delta"]})

        elif etype == "response.output_audio_transcript.delta":
            _mark_answer_started()
            delta = event.get("delta", "")
            agent_text_parts.append(delta)
            await client_ws.send_json({
                "type": "transcript", "role": "agent", "delta": delta,
            })

        elif etype == "conversation.item.input_audio_transcription.completed":
            timing["voice_in"] = time.perf_counter()
            transcript = event.get("transcript", "")
            log.info("\n[%s] 어르신: %s", session_id, transcript)  # 앞 줄바꿈으로 이전 턴과 구분
            await client_ws.send_json({
                "type": "transcript", "role": "elder", "text": transcript,
            })

        elif etype == "response.function_call_arguments.done":
            t_selected = time.perf_counter()
            name = event["name"]
            call_id = event["call_id"]
            args = json.loads(event.get("arguments") or "{}")

            if timing["voice_in"] is not None:
                log.info(
                    "[%s] 음성 들어감 -> 툴 선정(%s): %.0f ms",
                    session_id, name, (t_selected - timing["voice_in"]) * 1000,
                )

            t_called = time.perf_counter()
            log.info(
                "[%s] 툴 선정 -> 툴 호출(%s): %.1f ms",
                session_id, name, (t_called - t_selected) * 1000,
            )

            result = run_tool(state, name, args)
            log.info(
                "[%s] 툴 실행 소요(%s): %.1f ms",
                session_id, name, (time.perf_counter() - t_called) * 1000,
            )

            await upstream.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                },
            }))

            # checkin_start만 예외: 여기서 response.create를 또 걸면 모델이 인사와 첫
            # 본 질문을 한 응답 안에 이어붙여 말해버린다(실제로 겪은 문제). 이 턴은 그냥
            # 끝내고 turn_done을 보내, 어르신이 실제로 뭐라고 답한 다음에야 첫 질문이
            # 나가게 만든다. response.create를 안 거니 뒤이은 답변 생성도 없어 여기서는
            # "툴 호출 -> 답변 생성" 구간을 재지 않는다.
            if name != "checkin_start":
                timing["tool_called"] = t_called
                await upstream.send(json.dumps({"type": "response.create"}))
                awaiting_continuation = True  # 방금 response.create를 새로 걸었다

            await client_ws.send_json({"type": "tool_used", "name": name})
            if name == "end_call":
                ending = True  # 작별 인사(다음 response)까지는 듣고 나서 끊는다

        elif etype == "response.done":
            if awaiting_continuation:
                continue  # 곧 이어지는 응답이 있다 — 아직 클라이언트에 turn_done을 보낼 때가 아니다
            if agent_text_parts:  # 이 턴에서 실제로 뭔가 말했으면 한 줄로 모아서 찍는다
                log.info("[%s] 에이전트: %s", session_id, "".join(agent_text_parts))
                agent_text_parts.clear()
            await client_ws.send_json({"type": "turn_done"})  # 다음 발화용 새 말풍선을 열라는 신호
            if ending:
                await client_ws.send_json({"type": "call_ended"})
                return

        elif etype == "error":
            log.error("upstream error: %s", event)
            await client_ws.send_json({
                "type": "error",
                "message": event.get("error", {}).get("message", "unknown error"),
            })


@app.websocket("/ws/call")
async def call_ws(client_ws: WebSocket) -> None:
    await client_ws.accept()
    session_id = f"session-{uuid.uuid4().hex[:8]}"  # 로그 상관관계용 id — 조회 용도로 쓰이진 않는다
    state = new_call_state()
    log.info("call started: %s", session_id)

    try:
        async with websockets.connect(
            realtime_url(),
            additional_headers={"Authorization": f"Bearer {require_env('XAI_API_KEY')}"},
        ) as upstream:
            await upstream.send(json.dumps({
                "type": "session.update",
                "session": {
                    "voice": os.environ.get("TTS_VOICE", "ara"),
                    "instructions": build_full_instructions(),
                    "turn_detection": turn_detection_config(),
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "tools": TOOLS,
                },
            }))

            # GREETING_PROMPT을 한 번만 슬쩍 끼워 넣어 통화의 첫 마디를 인사말로 유도한다
            # (SYSTEM_PROMPT에 영구히 섞어 넣으면 이후 턴에도 계속 "인사만 하라"고 남는다).
            await upstream.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": GREETING_PROMPT}],
                },
            }))
            await upstream.send(json.dumps({"type": "response.create"}))

            await client_ws.send_json({"type": "ready"})

            upstream_task = asyncio.create_task(_pump_upstream(upstream, client_ws, state, session_id))
            downstream_task = asyncio.create_task(_pump_downstream(client_ws, upstream))

            done, pending = await asyncio.wait(
                {upstream_task, downstream_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                    log.exception("call %s pump failed", session_id, exc_info=exc)

    except WebSocketDisconnect:
        log.info("client disconnected: %s", session_id)
    except Exception:
        log.exception("call %s failed", session_id)
        try:
            await client_ws.send_json({"type": "error", "message": "서버 오류로 통화가 종료되었습니다."})
        except Exception:
            pass
    finally:
        try:
            await client_ws.close()
        except Exception:
            pass
        log.info("call ended: %s", session_id)


'''
@app.post("/tts")
async def tts_endpoint(request: Request) -> Response:
    """index2.html용 텍스트 -> 음성 변환. 입력한 문장을 mp3로 돌려주면, 브라우저가

    그걸 PCM으로 디코드해 실제 마이크 입력과 같은 경로(/ws/call의 "audio" 메시지)로
    흘려보낸다 — 그래야 서버 VAD/STT를 포함한 진짜 음성 파이프라인의 지연시간을 잰다.
    """
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return Response(content="empty text", status_code=400, media_type="text/plain")

    try:
        audio = synthesize_speech(text)
    except requests.HTTPError as exc:
        log.exception("tts failed")
        return Response(content=str(exc), status_code=502, media_type="text/plain")

    return Response(content=audio, media_type="audio/mpeg")
'''

# Serve static/index.html at / (and index2.html at /index2.html) — mounted
# last so /ws/call and /tts win the route match first.
app.mount("/", StaticFiles(directory="static", html=True), name="static")

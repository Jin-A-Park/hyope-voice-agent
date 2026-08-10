
# * 8개 툴(checkin_start, log_answer_analysis, double_check, flag_emergency, search_nearby_resource,
# * send_resource_info, update_recipient_profile, end_call)의 스키마(build_tools)와 실제 구현, run_tool 디스패처.
# * 실행 중인 통화의 state(agent/brain.py의 new_call_state)를 읽고 쓰지만, 어떤 질문을 물을지 조립하는
# * 로직(질문은행)은 agent/brain.py에 있다 — 여기는 "각 툴이 실제로 뭘 하는지"만 다룬다.
# ! 네트워킹(FastAPI, WebSocket, HTTP 클라이언트) & 데이터 주고받기 x — 지오코딩/SMS 등 외부 호출이
# ! 필요한 툴(flag_emergency, log_answer_analysis, search_nearby_resource, send_resource_info)은
# ! integrations/dispatch.py가 이 파일의 run_tool을 부르기 전에 미리 해결해서 값만 넘겨준다.

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from agent import brain

log = logging.getLogger("tools")

JUDGMENTS: list[str] = ["양호", "우려", "알수없음", "위급"]

# flag_emergency(severity=medium/low)에서 지오코딩(integrations/geo.py)이 실패했거나 학대/정신건강/
# 응급처럼 nearby_resource를 아예 안 쓰는 카테고리일 때 쓰는 정적 폴백.
HELP_RESOURCES: dict = json.loads((brain.ROOT / "static" / "help_resources.json").read_text())

# log_answer_analysis가 "우려" 판정을 기록했을 때 능동적으로 도움처를 추천할 질문 id -> 카카오
# 검색 키워드. 낙상 관련 FRAIL 문항(fall_gate/fall_unsteady/fall_worry)은 이미 flag_emergency의
# "낙상" 영역과 겹치므로 여기 넣지 않는다.
RESOURCE_TRIGGER_KEYWORDS: dict[str, str] = {
    "gds5_2": "경로당",             # 지루함 — 우울 GDS-5
    "meal": "무료급식소",            # 식사 못 함 — 복약/식사
    "health_self_rating": "내과",   # 자기평가 건강 — "병원"은 동물병원/치과의원 등도 걸려서 너무 넓음
    "frail_illness": "내과",        # FRAIL 질환
    "frail_fatigue": "내과",        # FRAIL 피로
}

def should_offer_resource(state: dict, category: str, question_id: str, assessment: str) -> str | None:
    # * log_answer_analysis가 방금 기록한 답변이 능동 도움처 추천 대상인지 판단해 카카오 검색
    # * 키워드를 돌려준다(대상 아니면 None) — flag_emergency와 달리 모델이 아니라 서버가 트리거를
    # * 결정한다. 카테고리당 한 번만 제안한다(state["_resource_offered_categories"]).

    if assessment != "우려":
        return None
    if category in state["_resource_offered_categories"]:
        return None
    return RESOURCE_TRIGGER_KEYWORDS.get(question_id)

# --------------------------------------------------------------------------
# 툴 스키마
# --------------------------------------------------------------------------

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
        "description": "낙상, 자해, 급성 통증, 도움 요청 등 응급 상황을 암시하는 명시적 또는 암묵적 신호가 감지되면 즉시 호출하십시오. 확신이 없어도 의심되는 즉시 호출하는 것을 우선하십시오. severity=high는 즉시 보호자에게 알림이 가고, medium/low는 보호자 알림 없이 호출 결과에 담긴 도움 받을 곳 정보를 그 자리에서 어르신께 자연스럽게 안내해야 합니다.",
        "parameters": {
            "type": "object",
            "properties": {
            "signal": {
                "type": "string",
                "description": "감지된 위험 신호에 대한 짧은 설명"
            },
            "severity": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "high: 즉각적인 위험(낙상 직후, 의식 저하, 심한 자해/자살 신호 등) — 보호자 즉시 알림. "
                               "medium/low: 우려되지만 즉각적 위험은 아님 — 통화 중 도움 받을 곳만 안내."
            }
            },
            "required": ["signal", "severity"]
        }
    },
    {
        "type": "function",
        "name": "search_nearby_resource",
        "description": "어르신이 병원, 복지관, 무료급식소, 취미/여가 활동 장소 등 근처에서 갈 만한 곳을 직접 "
                       "물어보시면 예외 없이 즉시 호출하십시오(필수 질문 흐름 중이든, 통화 시작 직후 잡담 "
                       "중이든 상관없음). '나중에 찾아보겠다'처럼 말로만 답하고 호출을 건너뛰지 마십시오.",
        "parameters": {
            "type": "object",
            "properties": {
            "search_keyword": {
                "type": "string",
                "description": "어르신이 하신 말을 실제 지도 검색에 걸리는 구체적인 장소 유형으로 바꿔서 "
                               "넣으십시오(어르신의 표현을 그대로 옮기지 마십시오) — 예: '속이 안 좋다/아프다'는 "
                               "'내과'로(그냥 '병원'은 동물병원까지 같이 걸립니다), '심심하다/놀 데 없나'는 "
                               "'경로당' 또는 '공원'으로, '복지기관'은 '노인복지관'으로."
            }
            },
            "required": ["search_keyword"]
        }
    },
    {
        "type": "function",
        "name": "send_resource_info",
        "description": "flag_emergency, log_answer_analysis, search_nearby_resource 응답으로 방금 이름만 "
                       "안내해드린 도움처 중 하나를, 어르신이 더 궁금해하시거나 문자로 연락처를 받고 싶다고 "
                       "하셨을 때 호출하십시오. 어르신이 아직 관심을 보이지 않았는데 먼저 호출하지 마십시오.",
        "parameters": {
            "type": "object",
            "properties": {
            "resource_name": {
                "type": "string",
                "description": "문자로 보내드릴 곳의 이름 — 방금 안내드린 이름 그대로 넣으십시오."
            }
            },
            "required": ["resource_name"]
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

_KOREAN_WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]  # 지남력 채점(log_answer_analysis)에서 씀

def _score_orientation_answer(state: dict, question_id: str) -> str | None:
    # 지남력 문항(sis_day/month/year)은 모델에게 정답을 준 적이 없으니 모델이 낸 assessment를
    # 믿지 않고, 서버가 방금 어르신이 한 말(call_log_entries 마지막 항목)을 실제 날짜와 직접
    # 대조해서 assessment를 대신 계산한다.
    if question_id not in ("sis_day", "sis_month", "sis_year"):
        return None
    if not state["call_log_entries"]:
        return None

    answer_text = state["call_log_entries"][-1]["answer"]
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    if question_id == "sis_day":
        correct = _KOREAN_WEEKDAYS[now.weekday()] in answer_text
    elif question_id == "sis_month":
        correct = f"{now.month}월" in answer_text
    else:
        correct = str(now.year) in answer_text
    return "양호" if correct else "우려"

def log_answer_analysis(
    state: dict, category: str, question_id: str, assessment: str, reason: str,
    nearby_resource: list[dict] | None = None,
) -> str:
    # * 답변 판정을 기록하고, 그 카테고리에서 다음에 뭘 물어야 할지(next_step)를 계산해 돌려준다.
    # * nearby_resource는 모델이 주는 게 아니라, integrations/dispatch.py가 should_offer_resource()로
    # * 판단해 지오코딩까지 마친 뒤 넘겨주는 값이다(flag_emergency와 같은 패턴, 여기선 네트워킹 안 함).

    category_items = state["_question_bank"]["category_items"]

    # sis_encode 같은 안내문(type: statement)은 애초에 어르신 답변이 필요 없다(듣자마자 바로 기록).
    asked_item = next((i for i in category_items.get(category, []) if i["id"] == question_id), None)
    is_statement = bool(asked_item and asked_item.get("type") == "statement")

    # 어르신의 실제 새 발화(call_log_entries 증가) 없이 또 호출됐다면, 모델이 답을 못 들은 채로
    # 지어낸 것이다 — response.create가 tool 호출 직후 바로 걸리다 보니(server.py) 실제 사용자
    # 오디오를 기다리지 않고 모델 혼자 질문+답변+다음 질문을 이어붙여 말해버리는 사고가 있었다.
    if not is_statement and len(state["call_log_entries"]) <= state["_last_logged_entry_count"]:
        return json.dumps({
            "ok": False,
            "next_step": "아직 어르신의 실제 답변을 듣지 못했습니다. log_answer_analysis를 호출하지 "
                         "말고, 방금 질문에 대한 어르신의 답변을 기다리세요.",
        }, ensure_ascii=False)
    state["_last_logged_entry_count"] = len(state["call_log_entries"])

    override = _score_orientation_answer(state, question_id)
    if override is not None:
        assessment = override
        reason = f"(서버가 실제 날짜와 대조) {reason}"

    state["logic_data"][category] = {"judgment": assessment, "reason": reason}
    state["asked_order"].append(question_id)

    if category == "취미/근황" and assessment in ("우려", "위급"):
        state["_recipient_profile_concern"] = True

    resource_suffix = ""
    if nearby_resource:
        state["_last_offered_resources"] = nearby_resource
        state["_resource_offered_categories"].add(category)
        names = ", ".join(r["name"] for r in nearby_resource)
        resource_suffix = (
            f" 그리고 이런 곳들의 이름만 자연스럽게 안내하세요(전화번호나 주소는 말하지 마세요): "
            f"{names}. 어르신이 더 궁금해하시거나 연락처를 원하시면 문자로 보내드리겠다고 제안하고, "
            "동의하시면 방금 안내한 이름 그대로 send_resource_info를 호출하세요."
        )

    # 재시도(버퍼에 있던 질문)였다면 이번 결과와 상관없이 버퍼에서 뺀다 — 재시도는 한 번만.
    state["pending_retry"] = [
        p for p in state["pending_retry"]
        if not (p["category"] == category and p["question_id"] == question_id)
    ]

    # sis_recall처럼 방금 로그된 질문을 전제(requires)로 하는 문항은, gap이 우연히 같은
    # 카테고리 안에서 채워지길 기다리지 않고 바로 pending_retry에 넣어 통화 막판 재시도 때
    # 확실히 다뤄지게 한다 — 매번 지연 방식이 결정적이라 로직이 단순해진다.
    already_queued = {(p["category"], p["question_id"]) for p in state["pending_retry"]}
    for dependent in category_items.get(category, []):
        req = dependent.get("requires")
        if req and req["item"] == question_id and (category, dependent["id"]) not in already_queued:
            state["pending_retry"].append({"category": category, "question_id": dependent["id"]})

    asked = state["asked_questions"].setdefault(category, set())
    asked.add(question_id)
    remaining = [
        item["question"] for item in category_items.get(category, [])
        if item["id"] not in asked and _gap_satisfied(state, item)
    ]
    random.shuffle(remaining)  # questions.json 순서대로 매번 똑같이 물어보지 않게

    # instructions.min_questions는 예전엔 프롬프트 표시용일 뿐 실제로 집행되지 않아서, 카테고리가
    # 최소 문항 수와 상관없이 첫 질문 하나만 답하면 바로 닫혀버리는 문제가 있었다 — 여기서 집행한다.
    min_questions = state["_question_bank"]["category_min_questions"].get(category, 1)
    not_enough = len(asked) < min_questions

    # unknown은 바로 재질문하지 않고 pending_retry에 담아 한 바퀴 돈 뒤 다시 묻는다 — 헷갈림 방지.
    if (is_statement or assessment in ("우려", "위급") or not_enough) and remaining:
        if is_statement:
            why = "방금 건 채점 대상이 아닌 안내문일 뿐"
        elif not_enough:
            why = f"'{category}'는 아직 최소 {min_questions}문항을 못 채웠습니다"
        else:
            why = f"'{category}'는 아직 good이 아닙니다"
        return json.dumps({
            "ok": True,
            "category_status": "진행중",
            "remaining_questions_in_category": remaining,
            "next_step": f"{why}. remaining_questions_in_category 중 하나로 이어서 물어보세요"
                         f"(같은 질문 반복 금지).{resource_suffix}",
        }, ensure_ascii=False)

    if assessment == "알수없음":
        state["pending_retry"].append({"category": category, "question_id": question_id})

    state["completed_categories"].add(category)
    all_categories = state["_question_bank"]["categories"]

    if category == "취미/근황" and state["_recipient_profile_concern"]:
        # 근황 잡담 중 기억/조리력 concern이 있었으면, 남은 카테고리 중 인지기능을 다음 순서로 당긴다.
        if "인지기능" in all_categories:
            all_categories.remove("인지기능")
            all_categories.insert(0, "인지기능")

    pending_categories = [c for c in all_categories if c not in state["completed_categories"]]

    if pending_categories:
        next_step = (
            f"'{category}' 카테고리는 끝났습니다. 쿠션어로 부드럽게 운을 뗀 뒤 다음 카테고리 "
            f"'{pending_categories[0]}'로 넘어가세요(예: '이제 {pending_categories[0]} 관련해서 "
            "좀 여쭤봐도 될까요?')."
        )
    elif state["pending_retry"]:
        retry = state["pending_retry"][0]
        retry_question = brain._find_question(state["_question_bank"]["questions"], retry["question_id"])
        next_step = (
            "모든 카테고리를 한 바퀴 돌았습니다. 다만 답을 알 수 없었던 질문이 남아있으니 "
            f"'{retry['category']}' 카테고리의 질문을 자연스럽게 한 번 더 여쭤보세요: \"{retry_question}\""
        )
    else:
        next_step = "모든 카테고리와 재시도까지 마쳤습니다. 짧게 마무리 인사를 하고 end_call을 호출하세요."

    next_step += resource_suffix

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

def emergency_category(signal: str) -> str:
    # 학대/정신건강/응급처럼 전문 위기개입이 필요한 신호는 "근처 노인복지관"보다 전국 전문
    # 핫라인이 훨씬 적절하다. integrations/dispatch.py도 이 함수로 "default"인지 미리 확인해서, 그
    # 경우가 아니면 어차피 안 쓸 지오코딩 API 호출 자체를 생략한다(불필요한 호출 방지).
    if any(k in signal for k in ("학대", "폭행", "방임", "때리", "맞")):
        return "학대"
    if any(k in signal for k in ("자살", "죽고 싶", "우울", "희망이 없", "외롭")):
        return "정신건강"
    if any(k in signal for k in ("쓰러", "의식", "심장", "숨", "응급")):
        return "응급"
    return "default"

def default_category_keyword(signal: str) -> str:
    # emergency_category()가 "default"(학대/정신건강/응급 전용 핫라인 대상이 아닌 일반 신호)로
    # 분류한 signal 안에서, 신체 불편/통증 관련 신호면 "내과"를, 그 외 일반적인 도움 요청이면
    # 기존처럼 "노인복지관"을 카카오 검색 키워드로 쓴다. "병원"은 카카오 검색에서 동물병원/
    # 치과의원/한의원 등도 같이 걸려서 사람 대상 일반 진료로는 너무 넓다(실사용 중 확인됨).
    if any(k in signal for k in ("아프", "통증", "머리", "속이 안", "어지럽", "몸살", "메스꺼")):
        return "내과"
    return "노인복지관"

def _help_resource_for(signal: str, nearby_resource: list[dict] | None) -> list[dict]:
    category = emergency_category(signal)
    if category == "default" and nearby_resource:
        return nearby_resource
    return HELP_RESOURCES.get(category, HELP_RESOURCES["default"])

def flag_emergency(
    state: dict, signal: str, severity: str, nearby_resource: list[dict] | None = None,
) -> str:
    # * 낙상/자해 등 위험 신호를 기록한다. severity로 다음 행동이 갈린다 — high는 감지된 그
    # * 즉시(통화가 언제 끝나는지와 무관하게) 보호자 알림 파일에 기록하고, medium/low는
    # * 보호자 알림 없이 통화 중 바로 어르신께 도움 받을 곳을 안내하게 한다.
    # * nearby_resource는 모델이 주는 게 아니라, 통화 중 이 tool이 실제로 호출되는 그 순간에
    # * integrations/geo.py가 지오코딩해서 넘겨주는 값이다(여기선 네트워킹 안 함).
    state["emergencies"].append({
        "signal": signal,
        "severity": severity,
        "flagged_at": time.time(),
    })

    if severity == "high":
        # end_call까지 기다리면 모델이 남은 질문을 계속 진행하는 동안 보호자 알림이 몇 분씩
        # 늦어질 수 있다 — 통화 종료 여부와 상관없이 감지 즉시 기록한다.
        brain.GUARDIAN_ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(brain.GUARDIAN_ALERTS_FILE, "a") as f:
            f.write(json.dumps({
                "id": new_alert_id(),
                "alert": signal,
                "logic_data": state["logic_data"],
                "metadata": state.get("_metadata") or {},  # recipient_id/phone_number from POST /call
                "filed_at": time.time(),
            }, ensure_ascii=False) + "\n")
        next_step = "보호자에게 알리는 절차가 진행됩니다. 침착하게 어르신을 안심시키는 말을 건네세요."
    else:
        resources = _help_resource_for(signal, nearby_resource)
        state["_last_offered_resources"] = resources
        resource_text = ", ".join(r["name"] for r in resources)
        next_step = (
            "심각한 위급 상황은 아니니 보호자 알림 없이, 어르신께 이런 곳에 도움을 요청할 수 있다고 "
            f"자연스럽게 안내하세요. 이름만 말하고 전화번호나 주소는 말하지 마세요: {resource_text}. "
            "어르신이 더 궁금해하시거나 연락처를 원하시면 문자로 보내드리겠다고 제안하고, 동의하시면 "
            "방금 안내한 이름 그대로 send_resource_info를 호출하세요."
        )

    return json.dumps({"ok": True, "next_step": next_step}, ensure_ascii=False)


def search_nearby_resource(state: dict, search_keyword: str, nearby_resource: list[dict] | None = None) -> str:
    # * 어르신이 필수 질문과 무관하게 직접 물어봤을 때(예: "근처에 병원 있어요?") 모델이 스스로
    # * 판단해서 호출하는 툴 — flag_emergency와 같은 패턴. nearby_resource는 모델이 주는 게 아니라
    # * integrations/dispatch.py가 search_keyword로 지오코딩해서 넘겨주는 값이다(여기선 네트워킹 안 함).

    if not nearby_resource:
        return json.dumps({
            "ok": True,
            "next_step": "죄송하지만 지금은 근처 정보를 확인하기 어렵다고 자연스럽게 안내하세요.",
        }, ensure_ascii=False)

    state["_last_offered_resources"] = nearby_resource
    names = ", ".join(r["name"] for r in nearby_resource)
    next_step = (
        f"이런 곳들의 이름만 자연스럽게 안내하세요(전화번호나 주소는 말하지 마세요): {names}. "
        "어르신이 더 궁금해하시거나 연락처를 원하시면 문자로 보내드리겠다고 제안하고, 동의하시면 "
        "방금 안내한 이름 그대로 send_resource_info를 호출하세요."
    )
    return json.dumps({"ok": True, "next_step": next_step}, ensure_ascii=False)


def find_offered_resource(state: dict, resource_name: str) -> dict | None:
    # * flag_emergency가 마지막으로 안내한 도움처 중 이름으로 하나를 찾는다.
    # * 정확 일치를 우선하고, 안 되면 부분 일치(모델이 이름을 살짝 다르게 되풀이해도 관대하게 허용).

    offered = state.get("_last_offered_resources") or []
    name = (resource_name or "").strip()
    for r in offered:
        if r["name"] == name:
            return r
    for r in offered:
        if name and (name in r["name"] or r["name"] in name):
            return r
    return None

def send_resource_info(state: dict, resource_name: str, sms_sent: bool = False) -> str:
    # * sms_sent는 모델이 주는 게 아니라, integrations/dispatch.py가 실제로 문자를 보내본 결과를 넣어준다
    # * (여기선 발송 자체는 안 함).

    resource = find_offered_resource(state, resource_name)
    if resource is None:
        return json.dumps({
            "ok": False,
            "next_step": "방금 안내해드린 곳 중 어디를 말씀하시는 건지 다시 자연스럽게 여쭤보세요.",
        }, ensure_ascii=False)

    if sms_sent:
        next_step = "문자로 보내드렸다고 안내하세요."
    else:
        next_step = (
            "문자 전송에 실패했으니, 대신 지금 전화로 알려드리세요: "
            f"{resource['name']} {resource['phone']}."
        )
    return json.dumps({"ok": True, "next_step": next_step}, ensure_ascii=False)


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
    # A-1000, A-1001, ... — numbered by how many alerts are already filed.

    count = 0
    if brain.GUARDIAN_ALERTS_FILE.exists():
        count = sum(1 for line in brain.GUARDIAN_ALERTS_FILE.read_text().splitlines() if line.strip())
    return f"A-{1000 + count}"

def end_call(state: dict) -> str:
    # * 통화 종료 — 응급 알림은 flag_emergency가 감지 즉시 이미 처리하므로 완전히 별개다.
    # * 여기는 그냥 통화를 마무리하라는 신호만 준다.

    return json.dumps({
        "call_ended": True,
        "next_step": "지금 바로 다음 답변으로 부드러운 작별 인사를 건네고 통화를 마치세요.",
    }, ensure_ascii=False)


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
        elif name == "search_nearby_resource":
            result = search_nearby_resource(state, **args)
        elif name == "send_resource_info":
            result = send_resource_info(state, **args)
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

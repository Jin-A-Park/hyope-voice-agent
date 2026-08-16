
# * 6개 툴(start_call, check_in, check_external_api_necessity, end_call, update_recipient_profile,
# * send_resource_info)의 스키마(build_tools)와 실제 구현, run_tool 디스패처. 실행 중인 통화의
# * state(agent/brain.py의 new_call_state)를 읽고 쓰지만, 스테이지 상태머신 자체(어떤 스테이지가
# * 우선인지, 질문을 어떻게 고르는지)는 agent/stages.py에 있다 — 여기는 "각 툴이 실제로 뭘 하는지"만
# * 다룬다.
# * check_in이 예전의 check_stage_status/check_question_validity/if_answer_valid/trigger_next_turn
# * 4개를 대체한다 — "LLM이 자칭한 stage가 서버가 실제로 추적 중인 스테이지와 같은지 확인 -> 그 stage
# * 관점에서 방금 답변을 판정 -> 다음 질문까지 한 번에 반환"을 한 호출로 끝낸다(agent/tools.py의
# * _advance/_apply_assessment 참고). 인사말은 상태 판정이 필요 없어 GREETING이 StageType은 아니지만,
# * 규칙(prompts.GREETING_PROMPT)을 모델에게 실어 보낼 통로로 start_call은 남아있다 — 이 툴은
# * CallStageBuffer를 전혀 건드리지 않는다. 어르신이 그 인사에 답하면, 통화의 첫 check_in 호출
# * (인자 없이 호출됨)만 판정을 건너뛰고 부트스트랩으로 동작해 첫 스테이지 질문을 내준다.
# ! 네트워킹(FastAPI, WebSocket, HTTP 클라이언트) & 데이터 주고받기 x — 지오코딩/SMS 등 외부 호출이
# ! 필요한 check_external_api_necessity/send_resource_info는 integrations/dispatch.py가 이 파일의
# ! run_tool을 부르기 전에 미리 해결해서(nearby_resource/sms_sent) 값만 넘겨준다.

from __future__ import annotations

import json
import logging
import time

from agent import prompts, stages

log = logging.getLogger("tools")

ASSESSMENTS: list[str] = ["문제있음", "부적절", "문제없음"]
SEVERITIES: list[str] = ["low", "medium", "high"]
STAGE_VALUES: list[str] = [s.value for s in stages.StageType]

# --------------------------------------------------------------------------
# 스테이지 <-> 자연어 라벨 — 화제 전환 멘트는 고정 문구를 그대로 읽게 하지 않고, 이 라벨만 알려준 뒤
# 모델이 매번 자연스럽게(다양하게) 직접 만들어 말하게 한다.
# --------------------------------------------------------------------------

_STAGE_LABEL: dict[stages.StageType, str] = {
    stages.StageType.EVENT: "최근 근황",
    stages.StageType.INTEREST: "취미",
    stages.StageType.ANXIETY: "불안",
    stages.StageType.DEPRESSION: "우울",
    stages.StageType.HEALTH: "건강",
    stages.StageType.MEDICATION: "복약",
    stages.StageType.MEAL: "식사",
    stages.StageType.EMERGENCY: "응급",
    stages.StageType.COGNITIVE: "인지기능",
    stages.StageType.CLOSING: "마무리",
}


def _stage_label(stage: stages.StageType) -> str:
    return _STAGE_LABEL.get(stage, stage.value)


# --------------------------------------------------------------------------
# "일상" 통합 화제 전환 — 매 통화 필수인 baseline 6개(근황/취미/복약/식사/우울/건강)는 원래도
# 전부 물어보게 되어있다(REACTIVE_ONLY_STAGES가 아니라서 건너뛸 수 없음). 문제는 이 6개 사이를
# 넘어갈 때마다 _stage_label로 "우울"/"건강" 같은 구체적 라벨을 모델에게 그대로 던져줘서,
# 모델이 그 임상적 카테고리명을 육성으로 말해버릴 위험이 있었다는 것 — 이 묶음 안에서는 화제
# 전환을 "일상" 한 번으로만 하고(맨 처음 진입할 때 한 번), 그 뒤로 6개 사이를 오갈 때는 새로
# 화제를 여는 것처럼 브릿지하지 않고 그냥 자연스럽게 이어서 묻는다 — 어차피 어르신 입장에선 계속
# 같은 "일상 얘기" 흐름이다. 응급/불안/인지기능처럼 반응형으로만 뜨는 카테고리는 이 대상이 아니다
# (사용자가 명시적으로 이 6개만 지정함) — 그쪽은 기존처럼 구체적 라벨로 브릿지한다.
# --------------------------------------------------------------------------

DAILY_LIFE_STAGES: set[stages.StageType] = set(stages.INITIAL_PRIORITY.keys())


def _question_goal(q: stages.QuestionCandidate) -> str:
    # * improvise=True(entry — 검증된 척도가 아니라 그냥 자연스럽게 떠보려고 지은 문구)면 text를
    # * 그대로 읽을 대본이 아니라 "이런 걸 파악하라"는 목표로만 주고, 모델이 지금까지 나눈 대화
    # * 흐름에 자연스럽게 얹어 직접 질문을 지어내게 한다 — GDS-5/GAD-2/SIS 같은 본 문항(검증된
    # * 척도라 문구를 임의로 바꾸면 안 됨)은 여전히 text를 그대로 참고해서 묻는다.
    if q.improvise:
        return (
            f"목표: \"{q.text}\". 이 문장을 그대로 옮기지 말고, 지금까지 나눈 대화에 자연스럽게 "
            "이어 붙여 네가 직접 질문을 새로 만들어 편하게 물어보세요."
        )
    return f"다음 질문: \"{q.text}\"를 참고해서 자연스럽게 이어서 질문하세요."


def _question_next_step(buffer: stages.CallStageBuffer, stage: stages.StageType, q: stages.QuestionCandidate) -> str:
    goal = _question_goal(q)
    if stage in DAILY_LIFE_STAGES:
        if not buffer.entered_daily_life:
            buffer.entered_daily_life = True
            return f"'일상' 쪽으로 자연스러운 한마디로 화제를 전환한 뒤, {goal}"
        return goal
    return f"'{_stage_label(stage)}' 쪽으로 자연스러운 한마디로 화제를 전환한 뒤, {goal}"


# --------------------------------------------------------------------------
# 툴 스키마
# --------------------------------------------------------------------------

def build_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "name": "start_call",
            "description": "통화 시작 후 인삿말 생성 툴입니다. 통화가 연결되면 가장 먼저, 오직 한 번만 호출합니다. ",
            "parameters": {"type": "object", "properties": {}, "required": []}
        },
        {
            "type": "function",
            "name": "check_in",
            "description": "어르신의 답변을 분석해야 할 때 필요한 함수로, 분석 후 "
                        "그 자리에서 다음 질문까지 함께 돌려주는 핵심 함수입니다. 어르신이 통화 종료 의사를 "
                        "명시적으로 밝히면 이 함수 대신 end_call을 호출하십시오.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stage": {
                        "type": "string", "enum": STAGE_VALUES,
                        "description": "지금 판정하려는 스테이지. 통화 시작 직후 첫 호출이면 생략.",
                    },
                    "assessment": {
                        "type": "string", "enum": ASSESSMENTS,
                        "description": "어르신 답변의 '내용'에 우려할 만한 문제/증상이 있는지에 대한 판정. "
                                    "'문제있음': 답변 내용 자체에서 우울/불안/통증/수면장애/인지저하 등 우려되는 증상이나 부정적 신호가 감지됨. "
                                    "'부적절': 답변이 질문과 동떨어지거나 이해를 못한 것처럼 보이는 동문서답. "
                                    "'문제없음': 답변 내용에 우려할 만한 게 특별히 없음. ",
                    },
                    "reason": {"type": "string", "description": "판단 근거 한 문장."},
                    "discomfort_flags": {
                        "type": "array",
                        "description": "답변 안에서 다른 스테이지에 속하는 우려 신호가 감지된 경우에만 "
                                    "채우십시오 — 한 답변에서 여러 카테고리(예: 건강+인지기능)가 동시에 "
                                    "의심되면 항목을 여러 개 넣으십시오.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "signal": {
                                    "type": "string",
                                    "description": "불편 표현/요청(배고픔, 심심함, 외로움 등)이나 인지 저하 "
                                                "신호(예: 방금 나눈 대화 내용 자체를 기억 못함, 조금 전에 "
                                                "물어본 걸 다시 물어봄, 앞뒤가 안 맞는 이야기) 등 감지된 "
                                                "내용에 대한 짧은 설명. 주의: '잘 안 들려요', '뭐라고 "
                                                "하셨어요?', '다시 말씀해 주세요'처럼 단순히 소리가 안 "
                                                "들리거나 놓쳐서 되묻는 것은 인지 저하 신호가 아닙니다 — "
                                                "이런 경우 category=\"cognitive\"로 신고하지 말고 그냥 "
                                                "다시 또렷하게 말해주세요.",
                                },
                                "category": {
                                    "type": "string", "enum": stages.DISCOMFORT_CATEGORY_VALUES,
                                    "description": "그 문제가 속한 스테이지 유형.",
                                },
                                "severity": {
                                    "type": "string", "enum": SEVERITIES,
                                    "description": "그 불편/요청의 심각도.",
                                },
                                "certain": {
                                    "type": "boolean",
                                    "description": "severity=\"high\"일 때만 의미가 있습니다. 어르신이 "
                                                "쓰러짐·호흡곤란처럼 명확하게 위급함을 밝힌 경우 true. "
                                                "애매한 뉘앙스만으로 심각도를 추측한 경우 false — 그러면 "
                                                "보호자에게 바로 알리지 않고, 서버가 대신 확인 질문을 한 번 "
                                                "물어보라고 안내합니다. 생략하면 true로 취급됩니다.",
                                },
                            },
                            "required": ["signal", "category", "severity"],
                        },
                    },
                },
                "required": [],
            }
        },
        {
            "type": "function",
            "name": "check_external_api_necessity",
            "description": "근처 도움처(병원/응급실/보건소/상담센터 등)를 검색하는 툴입니다. 두 경우에만 "
                        "호출하십시오 — (1) check_in이 action=\"emergency\"를 반환해 안심시키는 말과 "
                        "함께 병원/응급실 정보를 안내해도 될지 여쭤봤고 어르신이 동의했을 때, (2) "
                        "check_in의 next_step이 우려되는 내용과 관련해 도움처를 안내해도 될지 여쭤보라고 "
                        "안내했고 어르신이 동의했을 때. 두 경우 다 어르신이 먼저 동의해야만 호출하십시오.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stage_type": {"type": "string", "description": "우려 신호가 감지된 스테이지의 유형(e.g., '응급', '건강')"},
                    "answer_requirement": {"type": "string", "description": "답변이 요구하는 바(e.g., '응급실 검색', '병원 검색', '상담센터 검색')"},
                    "keyword": {
                        "type": "string",
                        "description": "검색어(e.g., '응급실', '병원', '보건소', '정신건강복지센터'). 어르신이 "
                                    "도움처 안내에 동의하지 않았으면 이 필드는 채우지 말고 생략하십시오.",
                    },
                },
                "required": ["stage_type", "answer_requirement"],
            }
        },
        {
            "type": "function",
            "name": "end_call",
            "description": "어르신이 '됐어', '끊어'처럼 통화 종료 의사를 명시적으로 밝혔을 때 호출합니다. "
                        "호출 전 짧은 작별 인사를 먼저 건네세요.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        },
        {
            "type": "function",
            "name": "update_recipient_profile",
            "description": "어르신이 질문은행에 없던 새 취미/근황을 언급하면 action=\"add\"로 기록하십시오. "
                        "기존에 기록된 취미나 최근 특이사항(recent_events)에 대해 어르신이 '그건 이제 "
                        "안 해요/끝났다/지난 일이다'처럼 더 이상 유효하지 않다는 뉘앙스를 보이면, 그 자리에서 "
                        "다시 묻지 말고 action=\"remove\"로 그 항목을 제거하십시오 — 취미는 kind=\"hobby\"와 "
                        "함께 hobby_name을, 최근 특이사항은 kind=\"event\"와 함께 event_id를 넣으십시오. ",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "remove"]},
                    "kind": {"type": "string", "enum": ["hobby", "event"]},
                    "text": {"type": "string", "description": "action=add일 때 새로 기록할 내용"},
                    "hobby_name": {"type": "string", "description": "action=remove이고 kind=\"hobby\"일 때 제거할 취미명(어르신이 말한 표현 그대로)"},
                    "event_id": {"type": "string", "description": "action=remove이고 kind=\"event\"일 때 제거할 근황 항목(질문에서 언급된 표현 그대로)"},
                    "reason": {"type": "string", "description": "판단 근거를 한 문장으로 요약"},
                },
                "required": ["action", "kind"],
            }
        },
        {
            "type": "function",
            "name": "send_resource_info",
            "description": "check_external_api_necessity로 방금 이름만 안내해드린 도움처 중 하나를, "
                        "어르신이 더 궁금해하시거나 문자로 연락처를 받고 싶다고 하셨을 때 호출하십시오 ",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_name": {
                        "type": "string",
                        "description": "문자로 보내드릴 곳의 이름 — 방금 안내드린 이름 그대로 넣으십시오.",
                    },
                },
                "required": ["resource_name"],
            }
        },
    ]

def _write_dashboard_alert(state: dict, stage_state: stages.StageState) -> None:
    # * new+severity=high 감지 즉시(통화 종료 여부와 무관하게) 보호자에게 알려야 한다는 결정을
    # * 내린다 — emergency 카테고리만 대상이다. 다른 카테고리(health/depression/anxiety/cognitive)는
    # * high여도 여기서는 조용히 넘어가고, 통화 종료 후 정상적인 리포트/위험도 알림 경로(assessment
    # * 저장 시 Spring이 만드는 NEW_REPORT/HIGH_RISK)로만 반영된다. 호출부 3곳(스폰 즉시, 확인 라운드
    # * 확정 시, low->high 승격 시) 전부 여기서 한 번에 막는다 — 호출부마다 따로 카테고리를 검사하면
    # * 하나라도 빠뜨렸을 때 다른 카테고리도 즉시 알림이 새나갈 수 있다.
    #
    # * 실제 통보(Spring POST + 보호자 SMS 즉시 시도, 실패 시에만 파일 폴백)는 네트워킹이 필요해서
    # * 여기서 직접 안 하고 대기열에만 쌓아둔다 — integrations/dispatch.py가 run_tool() 호출 직후
    # * 이 대기열을 비우며 처리한다(agent/ 쪽 네트워킹 금지 원칙, hyope-call-agent의 flag_emergency
    # * 즉시-통보 패턴과 동일하게 맞춘다).
    if stage_state.stage != stages.StageType.EMERGENCY:
        return
    state["emergencies"].append({
        "stage": stage_state.stage.value,
        "severity": stage_state.severity,
        "flagged_at": time.time(),
    })
    state.setdefault("_pending_guardian_alerts", []).append({
        "alert": f"{_stage_label(stage_state.stage)} 관련 심각도 높음",
        "logic_data": dict(state["logic_data"]),
        "filed_at": time.time(),
    })


# DEPRESSION/HEALTH의 entry 질문은 우울+인지, 건강+불안을 겸해서 스크리닝하도록 설계됐다
# (agent/stages.py의 INITIAL_PRIORITY 주석 참고) — COGNITIVE/ANXIETY는 반응형(discomfort_flags로만
# 스폰)이라 한 번도 안 뜨면 logic_data에 기록이 아예 없다. 이걸 무조건 "미확인"으로 두면
# 안 된다 — 짝이 되는 카테고리가 결국 "문제없음"으로 끝났다면 그건 "못 물어봤다"가 아니라
# "간접 스크리닝에서 이상 없었다"는 뜻이다. 통화 종료 시 문제없음으로 소급 반영한다.
_IMPLICIT_SCREEN_PAIR: dict[stages.StageType, stages.StageType] = {
    stages.StageType.COGNITIVE: stages.StageType.DEPRESSION,
    stages.StageType.ANXIETY: stages.StageType.HEALTH,
}


def backfill_implicit_screening(logic_data: dict) -> None:
    # * agent/brain.py의 write_call_result_outbox가 Spring/serve로 보내기 직전에 호출한다 —
    # * 통화 도중에 하면 안 된다(나중 턴에 진짜 discomfort_flag가 뜰 여지를 미리 닫아버리면 안 됨).
    for reactive, baseline in _IMPLICIT_SCREEN_PAIR.items():
        if reactive.value in logic_data:
            continue  # 실제로 반응형 신호가 떠서 이미 판정된 경우 — 건드리지 않는다
        baseline_judgment = (logic_data.get(baseline.value) or {}).get("judgment")
        if baseline_judgment == "문제없음":
            logic_data[reactive.value] = {
                "judgment": "문제없음",
                "reason": f"{_stage_label(baseline)} 문항에서 겸해서 스크리닝됨 — 신호 없음",
            }


# --------------------------------------------------------------------------
# 1. 다음 스테이지/질문 선택 — check_in의 부트스트랩/일반 판정 경로가 공유한다.
# --------------------------------------------------------------------------

def _complete_stage_and_advance(
    state: dict, buffer: stages.CallStageBuffer, stage_state: stages.StageState, gave_up: bool = False,
) -> dict:
    # * validate 은행소진 또는 failed 포기로 스테이지를 completed 처리하고, closing이면 작별
    # * 인사+end_call로, 아니면 다음 스테이지의 브릿지+진입 질문을 같은 턴에 이어 붙인다.

    stage_state.level = stages.StageLevel.COMPLETED
    stage_state.priority = -1
    if gave_up and stage_state.current_question is not None:
        state.setdefault("_unresolved_questions", []).append(
            {"stage": stage_state.stage.value, "question_id": stage_state.current_question.id}
        )

    if stage_state.stage == stages.StageType.CLOSING:
        return {
            "ok": True, "action": "end_call",
            "next_step": "짧게 작별 인사를 건네고 통화를 마치세요.",
        }

    # pick_next_stage()가 후보를 못 찾으면 closing을 스스로 재활성화해서 돌려주므로 여기선 None을
    # 신경 쓸 필요가 없다 — closing 자신이 이미 completed인 경우(통화가 이미 끝났어야 하는 상태)만 예외.
    next_stage = stages.pick_next_stage(buffer)
    if next_stage is None:
        return {"ok": False, "error": "no stage left to advance to"}
    buffer.set_active_stage(next_stage)
    next_state = buffer.current()
    q = next_state.next_question(buffer.turn_count)
    next_state.mark_asked(q, buffer.turn_count)
    return {
        "ok": True, "action": "ask_question",
        "stage_type": next_state.stage.value, "stage_status": "도전",
        "question": q.text, "note": q.note,
        "next_step": _question_next_step(buffer, next_state.stage, q),
    }


def _advance(state: dict, buffer: stages.CallStageBuffer) -> dict:
    # * 다음으로 물어볼 스테이지/질문을 고른다 — 기존 check_stage_status 본문과 동일한 분기.
    next_stage = stages.pick_next_stage(buffer)
    if next_stage is None:  # pick_next_stage()는 closing까지 스스로 재활성화하므로 여기 도달하지 않는다
        return {"ok": False, "error": "no stage left to advance to"}
    buffer.set_active_stage(next_stage)
    stage_state = buffer.current()
    log.info(
        "[stage] picked %s (level=%s, priority=%s, severity=%s, turn=%s)",
        stage_state.stage.value, stage_state.level.value, stage_state.priority,
        stage_state.severity, buffer.turn_count,
    )

    if stage_state.level == stages.StageLevel.TRY:
        # * 안심+병원 안내로 바로 라우팅하는 이 특수 분기는 EMERGENCY 카테고리 전용이다 — 다른
        # * 카테고리(건강/우울/불안/인지기능)가 discomfort_flags로 severity="high"를 직접 보고받아도
        # * 여기 안 걸리고 아래 q = stage_state.next_question(...)으로 빠져서, high 티어 문항을 정상
        # * 대로 물어본다("심각도 상인 질문으로 진입하면 문제없음이 돌아올 때까지 다 물어봐야 한다" —
        # * _apply_assessment가 문제있음이면 계속 validate로 순환시키고 문제없음이 나오면 그때 종료).
        # * EMERGENCY는 confirmed=True(모델이 처음부터 확신했거나, 확인 라운드를 이미 거쳐 확정된
        # * 경우)면 보호자 알림은 이미 스폰/확정 시점에 나갔으므로 그저 정상 질문 흐름 대신 안심시키는
        # * 응급 대응으로 라우팅만 한다. confirmed=False(모델이 certain=false로 스폰한 경우)면 아직
        # * 알림을 안 보낸 상태이므로, 여기서 확인 질문을 하나 강제한 뒤 그 답변으로 check_in을 다시
        # * 호출하게 한다 — 그 판정(check_in의 2.5단계)이 나와야만 실제로 알림이 나간다.
        if stage_state.severity == "high" and stage_state.stage == stages.StageType.EMERGENCY:
            if not stage_state.confirmed:
                return {
                    "ok": True, "action": "clarify_emergency",
                    "stage_type": stage_state.stage.value,
                    "next_step": "위급 신호일 가능성이 있으나 아직 확신할 정도는 아닙니다. 침착하게 "
                                 "어르신의 지금 상태를 구체적으로 확인하는 질문을 하나 자연스럽게 하세요 "
                                 "(예: 지금 어디가 어떻게 불편하신지, 스스로 움직이실 수 있는지 등 — 절대 "
                                 "'위급 상황인지 확인하겠다'는 식으로 말하지 마세요). 그 답변을 듣고, 정말 "
                                 "위급하다고 판단되면 문제있음으로, 아니라면 문제없음으로 판정하세요.",
                }
            return {
                "ok": True, "action": "emergency",
                "stage_type": stage_state.stage.value,
                "next_step": "심각한 위급 신호가 감지되어 보호자에게 알리는 절차가 이미 진행됐습니다. "
                             "침착하게 어르신을 안심시킨 뒤, 근처 병원이나 응급실 정보를 안내해드릴지 "
                             "여쭤보세요. 원하시면 check_external_api_necessity를 호출하세요. 만약 "
                             "어르신이 필요 없다고 답하시면, 짧게 안심시키는 말을 건넨 뒤 반드시 "
                             "end_call을 호출해 통화를 마치세요 — 어르신이 바로 119나 주변 도움을 "
                             "요청하실 수 있도록 통화를 오래 붙잡지 마세요.",
            }
        q = stage_state.next_question(buffer.turn_count)
        if q is None:  # 정상 흐름에선 도달하지 않지만, 방어적으로 다음 스테이지로 넘긴다
            return _complete_stage_and_advance(state, buffer, stage_state)
        stage_state.mark_asked(q, buffer.turn_count)
        return {
            "ok": True, "action": "ask_question",
            "stage_type": stage_state.stage.value, "stage_status": "도전",
            "question": q.text, "note": q.note,
            "next_step": _question_next_step(buffer, stage_state.stage, q),
        }

    if stage_state.level == stages.StageLevel.VALIDATE:
        q = stage_state.next_question(buffer.turn_count)
        if q is None:
            return _complete_stage_and_advance(state, buffer, stage_state)
        stage_state.mark_asked(q, buffer.turn_count)
        return {
            "ok": True, "action": "ask_question",
            "stage_type": stage_state.stage.value, "stage_status": "검증",
            "question": q.text, "note": q.note,
            "next_step": (
                f"\"{q.text}\"를 참고해서 자연스럽게 이어서 질문하세요. 그 답변에 대한 다음 "
                "check_in 호출의 assessment는 오직 이번 질문에 대한 답변 '내용'만 보고 판정하세요 — "
                "이 카테고리에서 앞서 확인된 다른 문제가 아직 남아있다고 해서 이번 답변까지 자동으로 "
                "문제있음으로 표시하지 마세요. 이번 답변 자체가 괜찮다는 내용이면 문제없음으로 "
                "판정하세요."
            ),
        }

    if stage_state.level == stages.StageLevel.FAILED:
        q = stage_state.next_question(buffer.turn_count)
        if q is None:
            return _complete_stage_and_advance(state, buffer, stage_state, gave_up=True)
        stage_state.mark_asked(q, buffer.turn_count)
        return {
            "ok": True, "action": "ask_question",
            "stage_type": stage_state.stage.value, "stage_status": "실패",
            "question": q.text, "note": q.note,
            "next_step": f"아까 정확한 답변을 얻지 못했다고 짧게 고지한 뒤 \"{q.text}\"를 다시 여쭤보세요.",
        }

    return {"ok": False, "error": f"unexpected level: {stage_state.level.value}"}


def start_call(state: dict) -> str:

    return json.dumps({
        "ok": True,
        "next_step": prompts.GREETING_PROMPT,
    }, ensure_ascii=False)


# --------------------------------------------------------------------------
# 2. 답변 판정 + 다음 질문 — 예전 check_stage_status/check_question_validity/if_answer_valid/
#    trigger_next_turn 4개를 대체한다. GREETING은 StageType이 아니라서(인사말은 start_call이 규칙만
#    실어 보내는 순수 발화일 뿐, 스테이지 상태머신은 관여하지 않는다) 통화의 첫 check_in 호출에는
#    판정할 것이 아직 없다 — buffer.turn_count == 0(스테이지가 하나도 안 뽑힌 시점)이면 판정을
#    건너뛰고 곧바로 첫 스테이지 질문으로 넘어간다(부트스트랩).
# --------------------------------------------------------------------------

def _apply_assessment(buffer: stages.CallStageBuffer, stage_state: stages.StageState, assessment: str) -> bool:
    # * 판정에 따라 level을 전이시키고, 이 판정으로 스테이지가 완료됐으면 True를 반환한다.
    old_level, old_priority = stage_state.level, stage_state.priority

    if assessment == "문제있음":
        # * EMERGENCY(별도 확정 플로우가 있음)를 뺀 나머지 카테고리가 severity="medium"/"high"로
        # * 확인 중이면, 관련 질문은 최소/최대 2개로 캡을 건다 — 2개까지 물어봤는데도 계속
        # * "문제있음"이면 그게 확인의 최대 한도이므로 더 캐묻지 않고 그대로 완료 처리해서 다음
        # * 스테이지로 넘어간다("문제없음"이 오면 그 전에 이미 else 분기에서 즉시 완료되므로 여기까지
        # * 안 온다). FIXED_SEQUENCE_STAGES(COGNITIVE)는 severity="high"(진짜 SIS 배터리)일 때만
        # * 캡 없이 전체 소진까지 진행한다 — discomfort_flags로 severity="low"/"medium"만 받고
        # * 반응형으로 스폰된 경우(예: 애매한 발언 하나로 인지기능이 의심된 경우)까지 SIS 배터리
        # * 전체를 강제로 다 물어보면 안 된다(캡 없이 5~6개까지 캐물은 실제 사례로 발견된 버그).
        capped_tier = (
            stage_state.severity in ("medium", "high")
            and stage_state.stage != stages.StageType.EMERGENCY
            and not (
                stage_state.severity == "high"
                and stage_state.stage in stages.FIXED_SEQUENCE_STAGES
            )
        )
        if capped_tier:
            stage_state.high_severity_asks += 1
        if capped_tier and stage_state.high_severity_asks >= 2:
            stage_state.level, stage_state.priority = stages.StageLevel.COMPLETED, -1
            completed = True
        else:
            stage_state.level, stage_state.priority = stages.StageLevel.VALIDATE, 1
            completed = False

    elif assessment == "부적절":
        stage_state.level, stage_state.priority = stages.StageLevel.FAILED, 0
        if stage_state.current_question is not None:
            stage_state.fail_current_question()
        completed = False

    else:  # 문제없음
        # COGNITIVE(SIS) 같은 고정 배터리 스테이지는 진짜 SIS 검사 중(severity="high")일 때만 문항
        # 하나가 문제없어도 나머지 문항(sis_recall/day/month/year 등)을 건너뛰지 않는다 — 은행이
        # 실제로(또는 min_gap 대기 없이) 소진됐을 때만 완료 처리하고, 그 전엔 문제있음과 똑같이
        # validate로 계속 진행시킨다. severity="low"/"medium"인 반응형 스폰(가벼운 의심만으로 뜬
        # 경우)까지 이 "전체 소진" 규칙을 적용하면 안 된다 — 위 capped_tier와 대칭.
        still_has_items = (
            stage_state.stage in stages.FIXED_SEQUENCE_STAGES
            and stage_state.severity == "high"
            and (
                stage_state.next_question(buffer.turn_count) is not None
                or stage_state.has_pending_gap_items(buffer.turn_count)
            )
        )
        if still_has_items:
            stage_state.level, stage_state.priority = stages.StageLevel.VALIDATE, 1
            completed = False
        else:
            stage_state.level, stage_state.priority = stages.StageLevel.COMPLETED, -1
            completed = True

    q = stage_state.current_question
    log.info(
        "[stage] %s: %s(%s) -> %s(%s) [assessment=%s, question=%s]",
        stage_state.stage.value, old_level.value, old_priority,
        stage_state.level.value, stage_state.priority, assessment,
        q.id if q is not None else None,
    )
    return completed


def _spawn_discomfort_flags(
    state: dict, buffer: stages.CallStageBuffer, stage_state: stages.StageState,
    discomfort_flags: list[dict] | None,
) -> None:
    # * discomfort_flags 배열을 순회하며 각 신호를 해당 스테이지에 스폰한다 — assessment 값과 무관하게
    # * 항상 평가. 현재 스테이지 자신과 같은 카테고리는 이미 _apply_assessment에서 failed/validate로
    # * 재시도되므로 건너뛴다. signal/severity가 진짜인지는 모델의 판단을 그대로 신뢰한다 — 서버는
    # * 별도로 재검증하지 않는다.
    # * logic_data는 여기서 바로 채운다 — 예전엔 스폰된 스테이지를 재평가(reassess)하는 왕복 호출로
    # * 채웠지만, 지금은 그 왕복이 없으므로(TRY로 흡수, agent/stages.py의 StageLevel 참고) 신호 자체를
    # * 그 스테이지의 판정으로 기록한다.
    # * severity="high"는 그 스테이지가 실제로 언제 뽑히는지(_advance)와 무관하게 감지 즉시 보호자에게
    # * 알린다 — pick 시점까지 미루면, 한 턴에 severity=high가 여러 개 동시에 들어왔을 때 늦게 뽑히는
    # * 쪽의 알림이 그만큼 지연되기 때문이다. 단, certain=false(모델이 확신 못함)면 즉시 알리지 않고
    # * confirmed=False로 스폰만 해둔다 — _advance가 이 스테이지를 뽑으면 확인 질문을 하나 강제하고,
    # * 그 판정(check_in의 확정 처리, 아래 참고)이 나온 뒤에야 실제로 알림이 나간다. was_pending이면
    # * 이미 그 확인 라운드를 한 번 거친 재신고이므로, 이번엔 certain 값과 무관하게 확정으로 취급한다
    # * (모델이 계속 certain=false로 미루면서 알림을 영영 안 보내는 걸 서버가 허용하지 않는다).
    for flag in discomfort_flags or []:
        signal = flag.get("signal")
        category = flag.get("category")
        severity = flag.get("severity")
        certain = flag.get("certain", True)
        if not (signal and category and category != stage_state.stage.value):
            continue

        # emergency 카테고리는 low/medium 단계로 완만하게 확인하지 않는다 — 위급 의심 자체가
        # questions.json에 severity="high" 문항만 있는 이유이기도 하다. 모델이 severity를 낮게
        # 보내도(예: 아직 확신 없다는 뜻으로 "low") 서버가 무조건 high로 올려서, low 티어 질문으로
        # 완곡하게 에두르지 않고 곧장 "지금 위급한 상태신가요?" 같은 직접적인 질문으로 간다.
        if category == stages.StageType.EMERGENCY.value:
            severity = "high"

        target_state = buffer.stages[stages.StageType(category)]
        was_pending = target_state.level != stages.StageLevel.COMPLETED and not target_state.confirmed
        target_state.level = stages.StageLevel.TRY
        target_state.priority = 1
        target_state.severity = severity
        # 재스폰(이전에 한 번 완료됐다가 discomfort_flags로 다시 뜨는 경우) 시 이전 라운드의
        # high_severity_asks가 남아있으면 새 라운드 첫 질문부터 곧바로 2개 캡에 걸려버린다 — 새
        # 라운드는 항상 0부터 다시 센다.
        target_state.high_severity_asks = 0
        state["logic_data"][category] = {"judgment": "문제있음", "reason": signal}
        log.info(
            "[stage] %s spawned as try(priority=1, severity=%s, certain=%s) via discomfort from %s (signal=%r)",
            category, severity, certain, stage_state.stage.value, signal,
        )
        if severity == "high":
            if certain or was_pending:
                target_state.confirmed = True
                _write_dashboard_alert(state, target_state)
            else:
                target_state.confirmed = False


def check_in(
    state: dict, stage: str | None = None,
    assessment: str | None = None, reason: str | None = None,
    discomfort_flags: list[dict] | None = None,
) -> str:
    buffer: stages.CallStageBuffer = state["_stage_buffer"]

    # 0) 부트스트랩 — 아직 스테이지가 하나도 안 뽑힌 시점(인사말에 대한 첫 답)이라 "판정"할 stage는
    # 없지만, discomfort_flags는 처리해야 한다 — 안 그러면 어르신이 인사에 대한 답으로 처음부터
    # "너무 우울했다"처럼 실제 우려 신호를 말해도 그냥 통째로 버려진다(실제 발견된 버그). active_stage가
    # 아직 임의 기본값(EVENT)이라 그 카테고리 자체를 자기 자신으로 착각해 걸러내는 예외 케이스가
    # 있을 수 있지만, 그 정도는 감수한다 — 나머지 카테고리는 정상적으로 스폰되어 첫 스테이지
    # 선택(_advance)에 곧바로 반영된다.
    if buffer.turn_count == 0:
        _spawn_discomfort_flags(state, buffer, buffer.current(), discomfort_flags)
        return json.dumps(_advance(state, buffer), ensure_ascii=False)

    # 1) discomfort_flags는 지금 판정 중인 스테이지가 뭐든 상관없이(다른 카테고리를 가리키는
    # 신호라) 먼저 처리한다 — stage 검증보다 앞에 둔 이유: 모델이 stage를 잘못 보고해서 아래에서
    # invalid_stage로 튕겨나가더라도, 같이 실려온 진짜 우려/위급 신호까지 함께 버려지면 안 된다.
    # 실제로 "지금 판정 중인 스테이지(예: meal)와 무관한 화제(예: 다리 통증)가 나오면 discomfort_flags로
    # 신고하라"는 지시를 모델이 stage 자체를 그 화제로 착각해서 잘못 부르는 경우가 있었다 — 그럴 때도
    # discomfort_flags 안의 신호는 유효하므로 여기서 먼저 살린다.
    _spawn_discomfort_flags(state, buffer, buffer.current(), discomfort_flags)

    # 2) LLM이 자칭한 stage가 서버가 실제로 추적 중인 현재 스테이지와 같은지 검사.
    try:
        stage_type = stages.StageType(stage)
    except ValueError:
        # * stage를 아예 안 넣었거나(부트스트랩 이후 턴에서도 생략) 잘못된 값을 넣은 경우 —
        # * next_step 없이 에러만 던지면 모델이 복구할 방법을 몰라서 어르신 답변 없이 그냥 또
        # * 호출하거나(같은 에러 반복), 심하면 이유를 지어내기 시작한다(실제 사례: 없던 "전화가
        # * 안 들린다" 발언을 지어내 discomfort_flags로 신고). 아래 mismatch 분기처럼 실제
        # * 스테이지 값을 알려줘서 다음 호출에서 바로 고칠 수 있게 한다.
        return json.dumps({
            "ok": False, "error": "invalid_stage",
            "next_step": f"stage 값이 없거나 잘못됐습니다. 지금 실제로 진행 중인 스테이지는 "
                         f"'{buffer.active_stage.value}'입니다 — 어르신이 방금 하신 답변을 그 값으로 "
                         "다시 판정해서 check_in을 호출하세요.",
        }, ensure_ascii=False)
    if stage_type != buffer.active_stage:
        return json.dumps({
            "ok": False, "error": "invalid_stage",
            "next_step": f"지금 실제로 진행 중인 스테이지는 '{buffer.active_stage.value}'입니다.",
        }, ensure_ascii=False)
    stage_state = buffer.current()

    state["logic_data"][stage_type.value] = {"judgment": assessment, "reason": reason}
    # * logic_data와 별개로 문항(question_id) 단위로도 남긴다 — 카테고리 단위인 logic_data와 달리,
    # * "복약/식사"처럼 한 카테고리 안에 문항이 여럿(medication_entry/meal_entry)이어도 서로 안
    # * 덮어쓴다. serve/assess.py의 _adherence()가 이 필드로 복약/식사 이행 여부를 판단한다.
    if stage_state.current_question is not None:
        state.setdefault("item_judgments", {})[stage_state.current_question.id] = {
            "judgment": assessment, "reason": reason,
        }

    # 3) 판정 -> level 전이
    # closing은 다른 스테이지와 다르게 다룬다 — "더 궁금한 점 있으세요?"에 "없어요"(문제없음)가
    # 아닌 다른 답(새로운 이야기든 동문서답이든)이 오면, 그 얘기를 자연스럽게 받아준 뒤에도 정말
    # 더 없는지 같은 질문을 다시 물어야 한다. closing_entry는 문항이 하나뿐이라 일반
    # _apply_assessment를 타면(다음 문항이 없어) 바로 강제 완료돼버리므로, fail_current_question()으로
    # 직접 재시도시킨다 — QUESTION_FAIL_LIMIT을 넘기면 그만 포기하고 끝내서 무한루프를 막는다.
    if stage_type == stages.StageType.CLOSING and assessment != "문제없음":
        stage_state.fail_current_question()
        stage_completed = False
    else:
        stage_completed = _apply_assessment(buffer, stage_state, assessment)

    # 3.5~3.7은 서로 배타적이다(elif로 묶는다) — 한 턴에 severity가 한 단계만 전이하게 하기 위해서다.
    # 안 그러면 예를 들어 3.7이 방금 severity를 None->low로 올렸는데 바로 이어서 3.6이 같은 턴의
    # 같은 "문제있음" 판정을 또 보고 low->high까지 한 턴에 훌쩍 건너뛰어 버린다 — low 티어로 한 번
    # 더 확인하자는 원래 의도(성급하게 확정하지 않기)가 무너진다.
    #
    # 3.5) 이 스테이지가 confirmed=False로 스폰됐던(직전 턴에 certain=false라 알림을 보류한) 위급
    # 신호였다면, 지금이 그 확인 질문에 대한 답변이다 — 여기서 확정한다. "문제있음"이면 정말
    # 위급했다는 뜻이라 지금 알림을 보낸다. 아니면 확인 결과 위급이 아니었다는 뜻이라 알림 없이
    # severity를 지워 더 이상 긴급 후보로 취급되지 않게 한다.
    if stage_state.severity == "high" and not stage_state.confirmed:
        stage_state.confirmed = True
        if assessment == "문제있음":
            _write_dashboard_alert(state, stage_state)
        else:
            stage_state.severity = None

    # 3.6) severity="low"로 스폰된 스테이지(의심은 되지만 아직 불확실한 상태 — questions.json의 low
    # 티어 문항으로 가볍게 물어보던 중)였는데, 그 low 문항에 대한 답까지 "문제있음"으로 판정되면 —
    # 즉 의심(스폰 시점 또는 3.7)에 이어 또 한 번("또") 문제가 확인된 것 — severity를 high로 즉시
    # 올린다. 이러면 next_question()이 다음부터 high 티어 문항을 물어보고, 이 승격 자체가 곧
    # 확정이므로(severity="high"+confirmed=True로 스폰된 것과 동급) 바로 보호자 알림도 나간다.
    elif stage_state.severity == "low" and assessment == "문제있음":
        stage_state.severity = "high"
        stage_state.confirmed = True
        _write_dashboard_alert(state, stage_state)

    # 3.7) baseline으로 승격된 스테이지(DEPRESSION/HEALTH)는 severity 없이(None) 시작해서, 인지/불안
    # 신호까지 겸해서 받는 넓은 is_entry 문항 하나만 우선 묻는다. 그 entry 답변에서 "문제있음"이
    # 나오면, 아직 확신할 단계는 아니니 곧장 high로 올리지 않고 먼저 low로 내려서 그 카테고리의
    # 부드러운 후속 문항들로 한 번 더 확인한다 — 거기서 또 문제가 확인되면 다음 턴엔 3.6이 이어받는다.
    elif (
        stage_state.current_question is not None
        and stage_state.current_question.is_entry
        and stage_state.severity is None
        and assessment == "문제있음"
    ):
        stage_state.severity = "low"

    # 3.8) EMERGENCY는 자체 확정 플로우(action="emergency")가 따로 있으니 대상이 아니다 — 나머지
    # 카테고리가 severity="medium"/"high"(capped_tier와 동일 범위 — 3.6번은 low->high만 처리해서
    # medium은 계속 medium인 채로 캡에 걸릴 수 있는데, 그 경우에도 권유는 나가야 한다)로 확인 중일
    # 때, 그 티어의 "첫" 질문 답변까지 또 "문제있음"으로 나오면(high_severity_asks==1) 응급 상황과
    # 다르게(강제 절차 없이) 관련 도움처를 물어보라고 안내한다. next_step 뒤에 덧붙이면 모델이
    # "다음 질문으로 이어가라"는 지시에 밀려 권유 자체를 건너뛰는 경우가 있었다 — 다음 질문 지시보다
    # 먼저 오도록 앞에 둔다. 실제로 도움처를 호출할지는 어르신이 원하는지에 달렸지만, 여쭤보는
    # 행위 자체는 건너뛰지 않도록 문구를 단정적으로 바꿨다.
    recommend_note = ""
    if (
        stage_type != stages.StageType.EMERGENCY
        and stage_state.severity in ("medium", "high")
        and stage_state.high_severity_asks == 1
        and assessment == "문제있음"
    ):
        recommend_note = (
            " 그 다음, 방금 확인된 내용과 관련해 근처에 도움받을 수 있는 곳(예: 병원, 보건소, "
            "상담센터 등)이 있는지 찾아봐드릴지 반드시 한 번 여쭤보세요 — 어르신이 원하면 "
            "check_external_api_necessity를 호출해서 안내하고, 원치 않는다고 답하면 그때 아래 "
            "다음 질문으로 넘어가세요."
        )
        # pick_next_stage()가 동률 랜덤으로 다른 스테이지(예: health)의 질문을 골라버리면, 권유
        # 문구("방금 확인된 내용")와 그 뒤에 이어 붙는 "다음 질문"이 서로 다른 카테고리를 가리키게
        # 되어 말이 안 섞인다(실제 사례: 우울 권유 문구 뒤에 보행 질문이 붙어 나옴) — 이번 턴만큼은
        # 같은 스테이지가 우선하도록 priority를 다른 VALIDATE 스테이지(기본값 1)보다 높여둔다.
        stage_state.priority = 2

    # 4) closing이 방금 완료됐으면 바로 작별 인사 안내
    if stage_completed and stage_type == stages.StageType.CLOSING:
        return json.dumps({
            "ok": True, "action": "end_call",
            "next_step": "공감성 발언 대신 짧은 작별 인사를 건네고 통화를 마치세요.",
        }, ensure_ascii=False)

    # 5) 다음 스테이지/질문 고르기 — trigger_next_turn 왕복 없이 이번 응답에 바로 실린다.
    result = _advance(state, buffer)
    if "next_step" in result:
        result["next_step"] = (
            "공감성 발언을 짧게 건넵니다."
            + recommend_note
            + " 그리고 해당 instruction을 실행하세요: " + result["next_step"]
        )
    return json.dumps(result, ensure_ascii=False)


# --------------------------------------------------------------------------
# 3. 외부 API 호출 로직
# --------------------------------------------------------------------------

def check_external_api_necessity(
    state: dict, stage_type: str, answer_requirement: str,
    keyword: str | None = None, nearby_resource: list[dict] | None = None,
) -> str:
    # * nearby_resource는 모델이 주는 게 아니라, integrations/dispatch.py가 keyword로 지오코딩한 결과를
    # * 넣어준다(여기선 네트워킹 안 함). 스테이지 상태는 여기서 바꾸지 않는다 — 상태 전이는 오직
    # * check_in만 한다(single-writer 원칙). 문자 발송은 이 함수가 아니라 전용 툴
    # * send_resource_info가 담당한다(방금 안내한 이름을 실제 목록과 대조 검증하기 위해 분리됨).

    # * EMERGENCY는 도움처 안내가 이 카테고리에서 할 수 있는 마지막 조치다 — 어르신이 문자를
    # * 원하지 않으면 곧바로 통화를 마쳐서 어르신이 바로 119나 주변에 도움을 요청할 수 있게 한다.
    # * 다른 카테고리(건강/우울/불안/인지, "장소 추천" 권유 플로우)는 도움처 안내 후에도 baseline
    # * 질문이 남아있을 수 있으니 통화를 끊으면 안 된다 — EMERGENCY일 때만 이 지시를 얹는다.
    buffer: stages.CallStageBuffer = state["_stage_buffer"]
    end_call_note = (
        " 어르신이 문자를 원하지 않으시면, 짧은 작별 인사와 함께 반드시 end_call을 호출해 통화를 "
        "마치세요."
    ) if buffer.active_stage == stages.StageType.EMERGENCY else ""

    if nearby_resource:
        state["_last_offered_resources"] = nearby_resource
        names = ", ".join(r["name"] for r in nearby_resource)
        return json.dumps({
            "ok": True,
            "next_step": f"이런 곳들의 이름만 자연스럽게 안내하세요(전화번호나 주소는 말하지 마세요): "
                         f"{names}. 어르신이 더 궁금해하시거나 연락처를 원하시면 문자로 보내드리겠다고 "
                         f"제안하세요.{end_call_note}",
        }, ensure_ascii=False)

    if keyword:
        # 검색을 시도했지만(keyword가 있었음) integrations/dispatch.py가 결과를 못 채워준 경우 —
        # KAKAO_API_KEY 미설정이나 지오코딩 실패. "외부 정보 불필요"와는 다른 상황이니 구분해서 안내.
        return json.dumps({
            "ok": True,
            "next_step": "죄송하지만 지금은 근처 정보를 확인하기 어렵다고 자연스럽게 안내하세요."
                         + (
                             " 위급 상황이니 바로 119에 직접 연락하시라고 안내한 뒤, 반드시 end_call을 "
                             "호출해 통화를 마치세요."
                             if buffer.active_stage == stages.StageType.EMERGENCY else ""
                         ),
        }, ensure_ascii=False)

    return json.dumps({
        "ok": True,
        "next_step": "어르신이 병원/응급실 안내를 원하지 않는 것으로 보입니다. 괜찮다고 자연스럽게 "
                     "답하고 대화를 이어가세요."
                     + (
                         " 짧게 안심시키는 말을 건넨 뒤 반드시 end_call을 호출해 통화를 마치세요."
                         if buffer.active_stage == stages.StageType.EMERGENCY else ""
                     ),
    }, ensure_ascii=False)


# --------------------------------------------------------------------------
# 4. 통화 종료
# --------------------------------------------------------------------------

def end_call(state: dict) -> str:
    # * 통화 종료 — 응급 알림은 check_in의 discomfort spawn 단계가 감지 즉시 이미 처리하므로 완전히 별개다.
    return json.dumps({
        "call_ended": True,
        "next_step": "지금 바로 다음 답변으로 부드러운 작별 인사를 건네고 통화를 마치세요.",
    }, ensure_ascii=False)


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


# --------------------------------------------------------------------------
# 5. 도움처 문자 발송
# --------------------------------------------------------------------------

def find_offered_resource(state: dict, resource_name: str) -> dict | None:
    # * check_external_api_necessity가 마지막으로 안내한 도움처 중 이름으로 하나를 찾는다.
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
    # * sms_sent는 모델이 주는 게 아니라, integrations/dispatch.py가 실제로 문자를 보내본 결과를
    # * 넣어준다(여기선 발송 자체는 안 함). 이 후속 발화는 원래 스테이지 질문에 대한 새로운 답이
    # * 아니라 "문자로 받고 싶다"는 부가 요청이라 check_in(재판정)을 거치지 않고 바로 다음
    # * 질문으로 넘어간다 — 원래 답변의 판정은 이미 그 이전 turn의 check_in에서 끝난 상태다.
    resource = find_offered_resource(state, resource_name)
    if resource is None:
        return json.dumps({
            "ok": False,
            "next_step": "방금 안내해드린 곳 중 어디를 말씀하시는 건지 다시 자연스럽게 여쭤보세요.",
        }, ensure_ascii=False)

    if sms_sent:
        prefix = "문자로 보내드렸다고 안내한 뒤, "
    else:
        prefix = (
            f"문자 전송에 실패했으니, 대신 지금 전화로 알려드리세요: {resource['name']} {resource['phone']}. "
            "그런 다음 "
        )

    buffer: stages.CallStageBuffer = state["_stage_buffer"]

    # * EMERGENCY는 이 문자(또는 전화 안내) 조치가 이 카테고리에서 할 수 있는 마지막 조치다 —
    # * 어르신이 바로 119나 주변에 도움을 요청할 수 있도록 여기서 통화를 마친다. _advance()로
    # * 다음 질문을 잇지 않는다(안 그러면 다른 baseline 질문이 응급 직후에 이어지는 부자연스러운
    # * 흐름이 된다). 다른 카테고리("장소 추천" 권유 플로우)는 baseline 질문이 남아있을 수 있으니
    # * 정상적으로 다음 질문으로 이어간다 — 그쪽만 아래 기존 로직을 그대로 탄다.
    if buffer.active_stage == stages.StageType.EMERGENCY:
        return json.dumps({
            "ok": True, "action": "end_call",
            "next_step": prefix + "곧바로 짧은 작별 인사를 건네고 반드시 end_call을 호출해 통화를 "
                         "마치세요.",
        }, ensure_ascii=False)

    # * check_external_api_necessity는 discomfort로 스폰된 TRY 스테이지에서도 호출될 수 있는데,
    # * 그 스테이지는 여기 오기까지 한 번도 check_in의 판정을 못 받아 여전히 TRY로 남아있다 —
    # * 그대로 _advance()를 부르면 pick_next_stage()가 severity 있는 TRY를 다시 최우선으로
    # * 뽑아버려서, 모델이 방금 끝낸 안내를 처음부터 반복하게 된다. 정보 전달(문자 발송 시도)로
    # * 그 우려가 해소됐다고 보고 여기서 직접 완료 처리한다.
    current = buffer.current()
    if current.level == stages.StageLevel.TRY:
        current.level = stages.StageLevel.COMPLETED
        current.priority = -1
    result = _advance(state, buffer)
    if "next_step" in result:
        result["next_step"] = prefix + "곧바로 이어서 " + result["next_step"]
    return json.dumps(result, ensure_ascii=False)


# SYSTEM_PROMPT(통화 전체에 한 번만 실리는 시스템 지시문) 자체를 없앴다 — 대신 이 규칙(정보가
#부족하면 함부로 판정하지 말고 먼저 확인하라)을 매 턴 next_step에 직접 실어 보낸다. run_tool은
# 모든 툴 결과가 모델에게 돌아가기 전 마지막으로 거치는 지점이라, 여기서 재차 박아 넣어 코드로
# 강제한다 — 통화가 길어져도 턴마다 다시 실리므로 초반 지시처럼 비중이 흐려지지 않는다.
_BASIC_REMINDER = (
    "툴 호출에 필요한 정보가 충족되지 않으면 함수 호출 없이 그냥 평소처럼 대화해서 "
    "확신이 서면 그때 지금까지 나눈 내용을 종합해서 판정한다. "
    "그리고 다음에 또 함수(예: check_in)를 호출해야 하는 상황이면, 먼저 몇 마디 말부터 꺼내고 "
    "그 다음 함수를 호출하지 말고, 함수부터 호출해서 결과를 받은 뒤 그 결과를 바탕으로 한 번에 "
    "말하라 — 함수 호출 전에 미리 대답의 일부를 말해버리면 이후 그 결과에 따라 또 한 번 "
    "말하게 되어 같은 내용을 두 번 말하는 것처럼 들린다. "
    "단, check_in은 반드시 어르신의 새로운 답변이 있을 때만 호출한다 — 방금 이 결과로 다음 질문을 "
    "받았다면, 그 질문을 실제로 소리 내어 물어보고 어르신이 답할 때까지 기다린 뒤에만 check_in을 "
    "다시 호출하라. 어르신의 새 답변 없이 check_in을 연속으로 또 호출하거나, 아직 안 물어본 "
    "질문의 assessment를 미리 짐작해서 채워 넣지 마라."
)


def _basic_internal_plan(result_json: str) -> str:
    try:
        data = json.loads(result_json)
    except (TypeError, ValueError):
        return result_json
    if isinstance(data, dict) and data.get("next_step"):
        data["next_step"] = data["next_step"] + _BASIC_REMINDER
        return json.dumps(data, ensure_ascii=False)
    return result_json


def run_tool(state: dict, name: str, args: dict) -> str:
    log.info("tool call: %s(%s)", name, json.dumps(args, ensure_ascii=False))
    try:
        if name == "start_call":
            result = start_call(state)
        elif name == "check_in":
            result = check_in(state, **args)
        elif name == "check_external_api_necessity":
            result = check_external_api_necessity(state, **args)
        elif name == "end_call":
            result = end_call(state)
        elif name == "update_recipient_profile":
            result = update_recipient_profile(state, **args)
        elif name == "send_resource_info":
            result = send_resource_info(state, **args)
        else:
            result = json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)
    except Exception as exc:
        log.exception("tool %s failed", name)
        result = json.dumps({"error": str(exc)}, ensure_ascii=False)

    return _basic_internal_plan(result)
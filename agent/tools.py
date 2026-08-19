
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
    stages.StageType.COGNITIVE: "인지기능",
    stages.StageType.CLOSING: "마무리",
}


def _stage_label(stage: stages.StageType) -> str:
    return _STAGE_LABEL.get(stage, stage.value)


# 3.7단계(도움처 제안)에서 "근처에 도움받을 수 있는 곳"을 뭉뚱그려 말하면, 한 통화 안에 건강/기분
# 문제가 같이 나왔을 때 어르신이 어느 쪽 제안인지 헷갈려 거절할 수 있다(실제 사례) — 카테고리별로
# 구체적인 장소 유형을 명시하게 한다.
_RESOURCE_HINT: dict[stages.StageType, str] = {
    stages.StageType.HEALTH: "병원이나 보건소",
    stages.StageType.DEPRESSION: "상담센터나 정신건강복지센터",
    stages.StageType.ANXIETY: "상담센터나 정신건강복지센터",
    stages.StageType.COGNITIVE: "병원(신경과)이나 보건소",
}


# --------------------------------------------------------------------------
# "일상" 통합 화제 전환 — 매 통화 필수인 baseline 6개(근황/취미/복약/식사/우울/건강)는 원래도
# 전부 물어보게 되어있다(REACTIVE_ONLY_STAGES가 아니라서 건너뛸 수 없음). 문제는 이 6개 사이를
# 넘어갈 때마다 _stage_label로 "우울"/"건강" 같은 구체적 라벨을 모델에게 그대로 던져줘서,
# 모델이 그 임상적 카테고리명을 육성으로 말해버릴 위험이 있었다는 것 — 이 묶음 안에서는 화제
# 전환을 "일상" 한 번으로만 하고(맨 처음 진입할 때 한 번), 그 뒤로 6개 사이를 오갈 때는 새로
# 화제를 여는 것처럼 브릿지하지 않고 그냥 자연스럽게 이어서 묻는다 — 어차피 어르신 입장에선 계속
# 같은 "일상 얘기" 흐름이다. 불안/인지기능처럼 반응형으로만 뜨는 카테고리는 이 대상이 아니다
# (사용자가 명시적으로 이 6개만 지정함) — 그쪽은 기존처럼 구체적 라벨로 브릿지한다.
# --------------------------------------------------------------------------

DAILY_LIFE_STAGES: set[stages.StageType] = set(stages.INITIAL_PRIORITY.keys())


_ONE_QUESTION_GUARD = (
    "이번 턴에 실제로 묻는 질문은 이것 하나뿐이어야 합니다 — 다른 화제(예: 식사, 수면 등)를 "
    "곁들여 함께 묻거나, 뒤에 이어 붙일 다음 질문을 미리 얹지 마세요. 궁금한 게 여러 개여도 "
    "지금은 이 질문 하나만 하고 어르신 답을 기다리세요."
)


def _question_goal(q: stages.QuestionCandidate) -> str:
    # * improvise=True(entry — 검증된 척도가 아니라 그냥 자연스럽게 떠보려고 지은 지시문)면 text를
    # * 그대로 읽을 대본이 아니라 모델이 따라야 할 지시로 주고, 지금까지 나눈 대화 흐름에 자연스럽게
    # * 얹어 직접 질문을 지어내게 한다 — GDS-5/GAD-2/SIS 같은 본 문항(검증된 척도라 문구를 임의로
    # * 바꾸면 안 됨)은 여전히 text를 그대로 참고해서 묻는다.
    if q.improvise:
        return (
            f"다음 지시를 따르세요: \"{q.text}\" 지금까지 나눈 대화에 자연스럽게 이어 붙여, "
            f"이전 통화들과는 다른 표현으로 네가 직접 질문을 만들어 편하게 물어보세요. {_ONE_QUESTION_GUARD}"
        )
    return f"다음 질문: \"{q.text}\"를 참고해서 자연스럽게 이어서 질문하세요. {_ONE_QUESTION_GUARD}"


def _question_next_step(buffer: stages.CallStageBuffer, stage: stages.StageType, q: stages.QuestionCandidate) -> str:
    goal = _question_goal(q)
    # * 화제 전환 멘트는 그 자체로 질문이 되면 안 된다 — 질문 형태로 던지면 그 뒤에 이어지는 진짜
    # * entry 질문과 합쳐져 한 턴에 질문이 두 개가 나가버린다(실제 사례: "식사는 잘 챙겨 드셨고요?
    # * 혹시 오늘 특별한 계획이... 있으신지 궁금해요"). 짧은 감상/공감 한마디로만 넘어가게 한다.
    bridge_guard = "(질문 형태가 아닌, 짧은 감상이나 공감 한마디로만 화제를 넘기세요)"
    if stage in DAILY_LIFE_STAGES:
        if not buffer.entered_daily_life:
            buffer.entered_daily_life = True
            return f"'일상' 쪽으로 자연스러운 한마디로 화제를 전환한 뒤{bridge_guard}, {goal}"
        return goal
    return f"'{_stage_label(stage)}' 쪽으로 자연스러운 한마디로 화제를 전환한 뒤{bridge_guard}, {goal}"


def _known_issue_note(state: dict, stage_state: stages.StageState) -> str:
    # * discomfort_flags로 방금 스폰/재활성화된 스테이지(severity가 붙어있음)는, 다음에 실제로
    # * 뽑히는 문항(특히 entry처럼 넓은 질문)이 그 스폰을 유발한 구체적 문제(예: 다리 통증)를 직접
    # * 짚지 않을 수 있다 — 질문 후보 선택은 severity 필터보다 is_entry를 우선하기 때문이다
    # * (agent/stages.py의 next_question() 참고). 그 결과 "다른 불편한 데는 없냐"류의 넓은 질문에
    # * 어르신이 원래 문제를 그대로 재확인해줘도(예: "다리 아픈 거 말고는 괜찮아") 모델이 "다른
    # * 문제 없음=문제없음"으로 오판하고 그 문제를 조용히 묻어버리는 사례가 실제로 있었다 — 그
    # * 문제가 스폰 사유였다는 걸 next_step에 직접 박아 넣어서, 판정 시 반드시 그 기준으로 보게 한다.
    if stage_state.severity is None:
        return ""
    known_reason = (state["logic_data"].get(stage_state.stage.value) or {}).get("reason")
    if not known_reason:
        return ""
    return (
        f" 참고: 이 스테이지는 방금 언급하신 문제(\"{known_reason}\") 때문에 다시 확인하는 "
        "것입니다 — 질문을 \"그거 말고 다른 데는요?\"처럼 바꿔 물었더라도, 어르신이 그 답에서 "
        "원래 문제를 다시 언급하거나(예: \"발목 말고는 없어\") 여전히 있다고 확인해주면 그 자체로 "
        "문제있음입니다 — \"다른 새로운 문제는 없다\"는 뜻이지 \"이제 괜찮다\"는 뜻이 아닙니다. "
        "이 문제가 이제 괜찮아졌거나 없어졌다고 명확히 말했을 때만 문제없음으로 판정하세요."
    )


# --------------------------------------------------------------------------
# 툴 스키마
# --------------------------------------------------------------------------

def build_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "name": "start_call",
            "description": "통화 시작 후 인삿말 생성 툴입니다. 해당 툴의 반환값을 가지고 인삿말을 생성할 수 있습니다. ",
            "parameters": {"type": "object", "properties": {}, "required": []}
        },
        {
            "type": "function",
            "name": "check_in",
            "description": "어르신의 답변을 분석하는 함수입니다. 해당 툴의 반환값을 가지고 모델의 대답을 생성할 수 있습니다. 어르신이 통화 종료 의사를 "
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
                    "reason": {
                        "type": "string",
                        "description": "판단 근거를 15자 내외의 짧은 한 문장으로만. 지금 이 답변 "
                                    "내용에만 집중하고, 다른 스테이지나 이전에 확인된 문제(예: '~는 "
                                    "여전히 남아있어 검증 상태를 유지합니다' 같은 상태머신 설명)는 "
                                    "언급하지 마세요 — 그건 서버가 알아서 추적합니다.",
                    },
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
                                                "신호(예: 앞뒤가 안 맞는 이야기, 본인이나 주변 관계, 기존 근황/취미에서의 "
                                                "기억력 의심, 질문과 전혀 무관한 단어·문장으로 답함) 등 감지된 내용에 "
                                                "대한 짧은 설명.",
                                },
                                "category": {
                                    "type": "string", "enum": stages.DISCOMFORT_CATEGORY_VALUES,
                                    "description": "그 문제가 속한 스테이지 유형.",
                                },
                                "severity": {
                                    "type": "string", "enum": SEVERITIES,
                                    "description": "그 불편/요청의 심각도.",
                                },
                                "keyword": {
                                    "type": "string",
                                    "description": "signal이 어떤 구체적 증상/주제에 가장 가까운지 다음 "
                                                "어휘 중 하나로만 답하십시오 — pain(통증/두통 등), "
                                                "sleep(수면), fall(낙상/휘청거림), mobility(보행/기력 저하), "
                                                "chronic_illness(지병), weight_loss(체중감소), "
                                                "appetite(식욕), fatigue(무기력/피로), worry(불안/걱정), "
                                                "somatic(가슴 두근거림 등 신체화), mood(기분 저하), "
                                                "interest(흥미 저하), worthless(무가치감/죄책감), "
                                                "memory(기억력). 애매하거나 위 어휘 중 맞는 게 없으면 "
                                                "생략하십시오 — 그러면 서버가 심각도 기준으로만 다음 질문을 "
                                                "고릅니다.",
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
            "description": "근처 도움처(병원/보건소/상담센터 등)를 검색하는 툴입니다. check_in의 "
                        "next_step이 우려되는 내용과 관련해 도움처를 안내해도 될지 여쭤보라고 안내했고 "
                        "어르신이 동의했을 때만 호출하십시오.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stage_type": {"type": "string", "description": "우려 신호가 감지된 스테이지의 유형(e.g., '건강', '기분')"},
                    "answer_requirement": {"type": "string", "description": "답변이 요구하는 바(e.g., '병원 검색', '상담센터 검색')"},
                    "keyword": {
                        "type": "string",
                        "description": "검색어. 두통·어지러움은 '신경과', 소화·복통은 '내과', 관절·낙상·계단 "
                                    "오르기 어려움은 '정형외과', 수면 문제는 '수면클리닉' 또는 '내과', 우울·불안·"
                                    "무기력은 '정신건강복지센터', 인지저하·기억력은 '신경과', 그 외 일반적인 "
                                    "경우는 '보건소'. 어르신이 도움처 안내에 동의하지 "
                                    "않았으면 이 필드는 채우지 말고 생략하십시오.",
                    },
                },
                "required": ["stage_type", "answer_requirement"],
            }
        },
        {
            "type": "function",
            "name": "end_call",
            "description": "통화를 마쳐야 할 때 호출합니다. 종료 문구는 end_call 반환값이 알려줍니다.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        },
        {
            "type": "function",
            "name": "update_recipient_profile",
            "description": "어르신이 질문은행에 없던 새 취미/근황을 언급하면 action=\"add\"로 기록하십시오. "
                        "기존에 기록된 취미나 최근 특이사항(recent_events)을 어르신이 부인하면, 그 이유를 "
                        "구분해서 다르게 처리하십시오. (1) '이제 안 해요/끝났다/지난 일이다'처럼 한때는 "
                        "사실이었지만 지금은 아니라는 뉘앙스이거나, '그건 제가 아니라 다른 분 얘기 아니에요?' "
                        "처럼 정보 자체가 애초에 잘못 기록됐다고 확신에 차서 정정하는 경우엔 — 그 자리에서 "
                        "다시 묻지 말고 action=\"remove\"로 제거하십시오. (2) 반대로 수술·입원처럼 본인이 "
                        "겪었다면 보통 잊기 힘든 일인데도 자신 없이 얼버무리거나 애매하게 부인하는 경우엔 "
                        "— 정보가 틀렸다고 단정하지 말고 이 함수를 호출하지 마십시오(사실일 수 있으니 "
                        "프로필을 지우면 안 됩니다). 대신 check_in의 discomfort_flags에 "
                        "category=\"cognitive\"로 신고해 기억력 확인이 필요함을 알리십시오. "
                        "취미는 kind=\"hobby\"와 함께 hobby_name을, 최근 특이사항은 kind=\"event\"와 함께 "
                        "event_id를 넣으십시오. ",
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


# 통화 종료 시 보호자 요약 SMS에 실을 카테고리 순서 — baseline 6개(근황->취미->건강->기분->식사->복약)
# 먼저, 그다음 반응형(불안/인지기능)은 실제로 그 통화에서 떴을 때만(logic_data에 항목이 있을 때만)
# 이어붙인다. CLOSING은 대화 주제가 아니라 항상 제외.
_SUMMARY_STAGE_ORDER: list[stages.StageType] = [
    *stages.INITIAL_PRIORITY.keys(),
    stages.StageType.ANXIETY, stages.StageType.COGNITIVE,
]


# discomfort_flags의 keyword(build_tools의 스키마 설명 참고)를 보호자가 읽을 한국어 라벨로
# 옮긴다 — StageState.symptom_tag(agent/tools.py의 _spawn_discomfort_flags가 채움)에 실려서
# 통화 끝까지 남아있는 값을 그대로 요약 문구에 쓴다.
_KEYWORD_LABEL: dict[str, str] = {
    "pain": "통증", "sleep": "수면 문제", "fall": "낙상 위험", "mobility": "보행/기력 저하",
    "chronic_illness": "지병", "weight_loss": "체중감소", "appetite": "식욕 저하",
    "fatigue": "무기력/피로", "worry": "불안/걱정", "somatic": "신체화 증상",
    "mood": "기분 저하", "interest": "흥미 저하", "worthless": "무가치감", "memory": "기억력 저하",
}


# EVENT/INTEREST(근황/취미)는 HEALTH/DEPRESSION 등과 달리 "문제 있는지"를 스크리닝하는 카테고리가
# 아니라 그냥 근황 정보 수집용이다 — assessment가 거의 항상 "문제없음"으로 나오는데, 그렇다고
# "상태 양호"로 뭉개버리면 손주 결혼식 같은 실제 근황 내용이 통째로 사라진다(실제 사례). 이 둘은
# 통화 시작 시 넘어온 profile(기존에 이미 알고 있던 취미/근황)과 이번 통화에서 확인된 내용을
# 대조해서 보여준다(build_call_summary_text 참고).


def _existing_profile_text(stage: stages.StageType, profile: dict) -> str:
    if stage == stages.StageType.EVENT:
        items = [e.get("text", "") for e in (profile.get("recent_events") or []) if e.get("text")]
    elif stage == stages.StageType.INTEREST:
        items = [h for h in (profile.get("hobbies") or []) if h]
    else:
        items = []
    return ", ".join(items) if items else "없음"


def _match_existing_hobby(profile: dict, reason: str) -> str | None:
    # * INTEREST 요약 라벨을 "취미"라는 일반 명사 대신 실제 취미 이름(예: "등산")으로 달기 위해,
    # * profile의 기존 취미 목록 중 reason 문장에 실제로 언급된 것을 찾는다. 문항 자체가
    # * "{name} 요즘도 하고 계세요?" 식으로 이름을 넣어 물으므로(agent/stages.py의
    # * build_interest_questions), 모델이 판정 이유에 그 이름을 그대로 쓰는 경우가 많다.
    for hobby in profile.get("hobbies") or []:
        if hobby and hobby in reason:
            return hobby
    return None


def build_call_summary_text(logic_data: dict, buffer: stages.CallStageBuffer, profile: dict) -> str:
    # * agent/brain.py의 write_call_result_outbox가 backfill_implicit_screening 이후에 호출해,
    # * 통화 종료 시 보호자에게 보낼 SMS 본문(카테고리별 한 줄 요약)을 만든다. 통화가 도중에
    # * 끊겨 logic_data에 항목이 없는 카테고리는 건너뛴다(무리하게 "미확인"을 채워넣지 않음).
    # * EVENT는 "기존 - .../ 신규 - ..." 두 항목 대조. INTEREST는 기존 취미 이름을 찾을 수 있으면
    # * "{취미명}(기존 - {취미명}, 신규 - 지속/중단[, 사유 - ...])" 형태로 — 취미는 "문제 있는지"가
    # * 아니라 "계속 하고 계신지"가 핵심이라 상태(지속/중단)를 먼저 보여주고, 중단인 경우에만 사유를
    # * 덧붙인다. 매칭되는 기존 취미가 없으면(완전히 새 취미) EVENT와 같은 두 항목 대조로 대체한다.
    # * 나머지(문제 스크리닝 카테고리)는 "문제없음"이면 "상태 양호"로, 그 외("문제있음"/"부적절")는
    # * 그 카테고리에 남아있는 symptom_tag(discomfort_flags의 keyword)를 한국어로 붙여 "OO 호소"
    # * 형태로 만든다 — keyword가 없으면(discomfort_flags 없이 baseline 문항에서 직접 확인된 경우)
    # * reason 텍스트만 쓴다.
    lines = []
    for stage in _SUMMARY_STAGE_ORDER:
        entry = logic_data.get(stage.value)
        if not entry:
            continue
        reason = entry.get("reason") or "확인 필요"
        judgment = entry.get("judgment")

        if stage == stages.StageType.INTEREST:
            hobby = _match_existing_hobby(profile, reason)
            if hobby:
                if judgment == "문제없음":
                    lines.append(f"{hobby}(기존 - {hobby}, 신규 - 지속)")
                else:
                    lines.append(f"{hobby}(기존 - {hobby}, 신규 - 일시적 중단, 사유 - {reason})")
                continue
            # 매칭되는 기존 취미가 없음(완전히 새로 언급된 취미) — EVENT와 동일한 대조 포맷으로 대체
            lines.append(f"{_stage_label(stage)}: 기존 - {_existing_profile_text(stage, profile)} / 신규 - {reason}")
            continue

        if stage == stages.StageType.EVENT:
            existing = _existing_profile_text(stage, profile)
            lines.append(f"{_stage_label(stage)}: 기존 - {existing} / 신규 - {reason}")
            continue

        if judgment == "문제없음":
            lines.append(f"{_stage_label(stage)}: 상태 양호")
            continue
        keyword_label = _KEYWORD_LABEL.get(buffer.stages[stage].symptom_tag or "")
        if keyword_label:
            lines.append(f"{_stage_label(stage)}: {keyword_label} 호소 ({reason})")
        else:
            lines.append(f"{_stage_label(stage)}: {reason}")
    return "\n".join(lines)


def build_call_summary_highlights(logic_data: dict) -> str:
    # * ClawOps SMS 발송은 본문이 200바이트를 넘으면 그대로 거부한다(clawops.BadRequestError) —
    # * build_call_summary_text()의 풀 버전(기존/신규 대조, OO 호소 등)은 한글이 많아 카테고리
    # * 2~3개만 있어도 200바이트를 훌쩍 넘는다(실제 통화에서 발견된 실패 사례). 그래서 실제
    # * SMS 본문에는 이 압축판(문제있음류 카테고리 라벨만 나열)을 쓰고, 상세 내용은
    # * runtime/call_summaries/*.txt 디버그 파일에만 build_call_summary_text()로 남긴다.
    # * EVENT는 스크리닝 카테고리가 아니라(그냥 근황 수집용) judgment와 무관하게 항상 제외한다.
    labels = []
    for stage in _SUMMARY_STAGE_ORDER:
        if stage == stages.StageType.EVENT:
            continue
        entry = logic_data.get(stage.value)
        if not entry or entry.get("judgment") == "문제없음":
            continue
        labels.append(_stage_label(stage))
    if not labels:
        return "특이사항 없음"
    return ", ".join(labels) + " 관련 대화가 있었어요"


def build_profile_updates_text(profile_updates: list[dict]) -> str:
    # * update_recipient_profile로 이번 통화 중 반영된 취미/근황 변경 사항을 요약 SMS에 같이 실어
    # * 보호자가 통화 내용 전체(불편 사항 + 새로 알게 된 근황/취미)를 한 문자로 받게 한다.
    lines = []
    for u in profile_updates or []:
        kind_label = "취미" if u.get("kind") == "hobby" else "근황"
        if u.get("action") == "add":
            lines.append(f"{kind_label} 추가: {u.get('text') or ''}")
        else:
            name = u.get("hobby_name") or u.get("event_id") or ""
            reason = u.get("reason")
            suffix = f" ({reason})" if reason else ""
            lines.append(f"{kind_label} 정정: '{name}' 더 이상 해당 없음{suffix}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 1. 다음 스테이지/질문 선택 — check_in의 부트스트랩/일반 판정 경로가 공유한다.
# --------------------------------------------------------------------------

def _mark_asked(buffer: stages.CallStageBuffer, stage_state: stages.StageState, q: stages.QuestionCandidate) -> None:
    # * StageState.mark_asked()를 감싸면서, 방금 asked된 q에 의존하는 문항(depends_on==q.id)이
    # * 있으면 PendingGap을 예약한다 — StageState는 CallStageBuffer를 모르므로 이 조립은 여기서 한다.
    stage_state.mark_asked(q)
    dependent = next(
        (item for item in stage_state.questions if item.depends_on == q.id and item.gap > 0),
        None,
    )
    if dependent is not None:
        buffer.pending_gap = stages.PendingGap(
            stage=stage_state.stage, question_id=dependent.id, fillers_remaining=dependent.gap,
        )


def _consume_pending_gap(buffer: stages.CallStageBuffer) -> dict:
    # * buffer.pending_gap이 있는 동안 _advance()가 pick_next_stage() 대신 호출한다. fillers_remaining이
    # * 남아있으면 무관한 즉흥 질문을 하나 더 내고(다른 스테이지 포함 전부 멈춤), 다 채웠으면 예약해둔
    # * 의존 문항(sis_recall 등)을 강제로 낸다 — 무작위 후보 풀을 절대 거치지 않는다.
    gap = buffer.pending_gap
    if gap.fillers_remaining > 0:
        gap.fillers_remaining -= 1
        buffer.awaiting_gap_answer = True
        return {
            "ok": True, "action": "ask_question",
            "stage_type": gap.stage.value, "stage_status": "간격 채우기",
            "question": None, "note": None,
            "next_step": "지금까지 나눈 대화 흐름과는 무관하게, 어르신과 가볍게 나눌 수 있는 일상적인 "
                         "질문을 하나 자연스럽게 새로 만들어 물어보세요 — 평가 대상이 아니니 앞서 "
                         "말씀드린 단어들과는 무관한 화제로 물어보세요. 그 답을 들으면 같은 stage로 "
                         "check_in을 다시 호출하세요 — assessment는 아무 값이나 넣어도 됩니다.",
        }
    buffer.pending_gap = None
    stage_state = buffer.stages[gap.stage]
    buffer.set_active_stage(gap.stage)
    q = next(item for item in stage_state.questions if item.id == gap.question_id)
    _mark_asked(buffer, stage_state, q)
    return {
        "ok": True, "action": "ask_question",
        "stage_type": gap.stage.value, "stage_status": "검증",
        "question": q.text, "note": q.note,
        "next_step": _question_next_step(buffer, gap.stage, q),
    }


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
            "next_step": "end_call을 호출해 통화를 마치세요 — 종료 문구는 end_call 반환값이 알려주니 그때 생성하세요. "
                         "end_call 호출을 빼먹으면 통화가 안 끊깁니다.",
        }

    # pick_next_stage()가 후보를 못 찾으면 closing을 스스로 재활성화해서 돌려주므로 여기선 None을
    # 신경 쓸 필요가 없다 — closing 자신이 이미 completed인 경우(통화가 이미 끝났어야 하는 상태)만 예외.
    next_stage = stages.pick_next_stage(buffer)
    if next_stage is None:
        return {"ok": False, "error": "no stage left to advance to"}
    buffer.set_active_stage(next_stage)
    next_state = buffer.current()
    q = next_state.next_question()
    _mark_asked(buffer, next_state, q)
    return {
        "ok": True, "action": "ask_question",
        "stage_type": next_state.stage.value, "stage_status": "도전",
        "question": q.text, "note": q.note,
        "next_step": _question_next_step(buffer, next_state.stage, q),
    }


def _advance(state: dict, buffer: stages.CallStageBuffer) -> dict:
    # * 다음으로 물어볼 스테이지/질문을 고른다 — 기존 check_stage_status 본문과 동일한 분기.
    # * pending_gap이 있으면 depends_on 문항(sis_recall 등)을 강제로 채워야 하니 pick_next_stage()의
    # * 무작위 선택을 아예 건너뛴다 — 다른 스테이지도 이 간격 동안은 끼어들지 못한다.
    if buffer.pending_gap is not None:
        return _consume_pending_gap(buffer)
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
        if stage_state.pending_probe:
            # * discomfort_flags로 방금 이 스테이지가 뜬 이유(신고된 증상)를 척도 문항으로 바로
            # * 건너뛰지 않고, 자연어로 한 번 더 캐묻는다 — 임상 척도 문항(GAD-2/GDS-5/frailty 등)은
            # * 문구를 임의로 바꿀 수 없어서 특정 증상에 맞춰 고칠 수 없기 때문에, 척도 문항 앞에
            # * 자유 발화 확인 턴을 하나 끼워넣는 방식으로 대신한다. 이 턴은 실제 QuestionCandidate가
            # * 아니므로 mark_asked를 부르지 않는다 — 그 답변에 대한 채점(_apply_assessment,
            # * high_severity_asks 카운트)도 건너뛰어야 한다(agent/tools.py의 check_in
            # * awaiting_probe_answer 게이트 참고).
            probe_signal = stage_state.pending_probe
            stage_state.pending_probe = None
            stage_state.awaiting_probe_answer = True
            return {
                "ok": True, "action": "probe",
                "stage_type": stage_state.stage.value,
                "next_step": (
                    f"방금 말씀하신 \"{probe_signal}\"에 대해, 언제부터 그러셨는지·얼마나 자주 또는 "
                    "심하게 그러시는지 등을 자연스럽게 한 가지만 더 여쭤보세요. " + _ONE_QUESTION_GUARD +
                    " 어르신 답을 들으면 같은 stage로 check_in을 다시 호출하세요 — assessment는 아무 "
                    "값이나 넣어도 됩니다. 그러면 다음 질문을 안내해 드립니다."
                ),
            }
        q = stage_state.next_question()
        if q is None:  # 정상 흐름에선 도달하지 않지만, 방어적으로 다음 스테이지로 넘긴다
            return _complete_stage_and_advance(state, buffer, stage_state)
        _mark_asked(buffer, stage_state, q)
        return {
            "ok": True, "action": "ask_question",
            "stage_type": stage_state.stage.value, "stage_status": "도전",
            "question": q.text, "note": q.note,
            "next_step": _question_next_step(buffer, stage_state.stage, q) + _known_issue_note(state, stage_state),
        }

    if stage_state.level == stages.StageLevel.VALIDATE:
        q = stage_state.next_question()
        if q is None:
            return _complete_stage_and_advance(state, buffer, stage_state)
        _mark_asked(buffer, stage_state, q)
        return {
            "ok": True, "action": "ask_question",
            "stage_type": stage_state.stage.value, "stage_status": "검증",
            "question": q.text, "note": q.note,
            "next_step": (
                f"\"{q.text}\"를 참고해서 자연스럽게 이어서 질문하세요. {_ONE_QUESTION_GUARD} 그 답변에 "
                "대한 다음 check_in 호출의 assessment는 오직 이번 질문에 대한 답변 '내용'만 보고 "
                "판정하세요. 이전 답변은 평가 제외 대상입니다. 현재 답변에서 이전 문제가 이제 괜찮아졌거나 없어졌다고 "
                "명확히 말했을 때만, 또는 애초에 문제라 할 점이 정말 없을 때만 문제없음으로 판정하세요."
                + _known_issue_note(state, stage_state)
            ),
        }

    if stage_state.level == stages.StageLevel.FAILED:
        q = stage_state.next_question()
        if q is None:
            return _complete_stage_and_advance(state, buffer, stage_state, gave_up=True)
        _mark_asked(buffer, stage_state, q)
        return {
            "ok": True, "action": "ask_question",
            "stage_type": stage_state.stage.value, "stage_status": "실패",
            "question": q.text, "note": q.note,
            "next_step": f"아까 정확한 답변을 얻지 못했다고 짧게 고지한 뒤 \"{q.text}\"를 다시 여쭤보세요. {_ONE_QUESTION_GUARD}",
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
        # * 카테고리가 severity="medium"/"high"로 확인 중이면, 관련 질문은 최소/최대 2개로 캡을
        # * 건다 — 2개까지 물어봤는데도 계속 "문제있음"이면 그게 확인의 최대 한도이므로 더 캐묻지
        # * 않고 그대로 완료 처리해서 다음 스테이지로 넘어간다("문제없음"이 오면 그 전에 이미 else
        # * 분기에서 즉시 완료되므로 여기까지 안 온다). FIXED_SEQUENCE_STAGES(COGNITIVE)는
        # * severity="high"(진짜 SIS 배터리)일 때만 캡 없이 전체 소진까지 진행한다 —
        # * discomfort_flags로 severity="low"/"medium"만 받고 반응형으로 스폰된 경우(예: 애매한
        # * 발언 하나로 인지기능이 의심된 경우)까지 SIS 배터리 전체를 강제로 다 물어보면 안 된다
        # * (캡 없이 5~6개까지 캐물은 실제 사례로 발견된 버그).
        capped_tier = (
            stage_state.severity in ("medium", "high")
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
        # 실제로 소진되고(depends_on 문항 제외) 강제 대기 중인 PendingGap도 없을 때만 완료 처리하고,
        # 그 전엔 문제있음과 똑같이 validate로 계속 진행시킨다. severity="low"/"medium"인 반응형
        # 스폰(가벼운 의심만으로 뜬 경우)까지 이 "전체 소진" 규칙을 적용하면 안 된다 — 위 capped_tier와
        # 대칭. pending_gap 체크가 없으면, sis_encode가 다른 고티어 문항들보다 늦게 뽑혀 마지막으로
        # 채점되는 경우 next_question()이 이미 None(sis_recall은 후보 풀에 안 들어가므로)이라 아직
        # sis_recall을 강제 주입하기도 전에 COGNITIVE가 COMPLETED로 잘못 넘어가 버린다.
        still_has_items = (
            stage_state.stage in stages.FIXED_SEQUENCE_STAGES
            and stage_state.severity == "high"
            and (
                stage_state.next_question() is not None
                or (buffer.pending_gap is not None and buffer.pending_gap.stage == stage_state.stage)
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


def _category_had_confirmed_problem(state: dict, stage_state: stages.StageState) -> bool:
    # * 이 카테고리에서 지금까지 실제로 물어본 문항 중 "문제있음"으로 판정된 게 하나라도 있는지
    # * item_judgments(문항 단위 기록, 카테고리 단위 logic_data와 달리 안 덮어씀)로 확인한다.
    # * check_in의 logic_data 기록부가 이걸로 "이미 확인된 문제를 마지막 턴 하나가 문제없음으로
    # * 되돌려서 지워버리는" 걸 막는 데 쓴다(실제 통화 사례: 우울 카테고리가 2개 문항에서 "문제있음"
    # * 이었는데 극성이 헷갈리는 마지막 문항(GDS "집에 있는 게 좋은가요?")을 모델이 잘못 판정해서
    # * 카테고리 전체가 "문제없음"으로 기록됨 — 사람이 보는 요약/실제 채점 데이터에서 그 우울 신호가
    # * 통째로 사라지는 문제가 있었음).
    item_judgments = state.get("item_judgments") or {}
    asked_ids = {q.id for q in stage_state.questions if q.asked}
    return any(
        (item_judgments.get(qid) or {}).get("judgment") == "문제있음"
        for qid in asked_ids
    )


def _spawn_discomfort_flags(
    state: dict, buffer: stages.CallStageBuffer, stage_state: stages.StageState,
    discomfort_flags: list[dict] | None, assessed_stage: stages.StageType | None,
) -> None:
    # * discomfort_flags 배열을 순회하며 각 신호를 해당 스테이지에 스폰한다 — assessment 값과 무관하게
    # * 항상 평가. signal/severity가 진짜인지는 모델의 판단을 그대로 신뢰한다 — 서버는 별도로
    # * 재검증하지 않는다.
    # * logic_data는 여기서 바로 채운다 — 예전엔 스폰된 스테이지를 재평가(reassess)하는 왕복 호출로
    # * 채웠지만, 지금은 그 왕복이 없으므로(TRY로 흡수, agent/stages.py의 StageLevel 참고) 신호 자체를
    # * 그 스테이지의 판정으로 기록한다.
    # * assessed_stage: 이번 check_in 호출의 assessment가 실제로 적용될 스테이지(모델이 자칭한 stage가
    # * buffer.active_stage와 실제로 일치할 때만 채워짐, 그 외엔 None) — 아래 self-reference 판단에
    # * stage_state.stage(서버가 실제로 추적 중인 현재 스테이지)가 아니라 반드시 이걸 써야 한다.
    for flag in discomfort_flags or []:
        signal = flag.get("signal")
        category = flag.get("category")
        severity = flag.get("severity")
        keyword = flag.get("keyword")
        if not (signal and category):
            continue

        if assessed_stage is not None and category == assessed_stage.value:
            # * 자기 카테고리를 가리키는 self-reference(예: 지금 health를 판정 중인데
            # * discomfort_flags에도 category="health"가 실려옴) — 이 턴의 판정은 이미
            # * _apply_assessment가 정식으로 처리하므로 스폰(level/severity/probe 등)은 필요
            # * 없다. 다만 모델이 같이 보낸 keyword까지 통째로 버리면 안 된다 — 실제 통화에서
            # * category가 자기 자신인 discomfort_flags가 흔하게 나오는데(예: depression
            # * 판정 중 keyword="fatigue"), 이걸 그냥 버리면 서버가 알아낸 구체적 증상 태그가
            # * 사라져서 다음 태그매칭 문항 선택에 못 쓴다.
            # * assessed_stage가 None이면(모델이 자칭한 stage가 무효/불일치라 이 턴의 assessment가
            # * 결국 아무 데도 반영 안 될 예정) 아래 일반 스폰 경로로 흘려보낸다 — category가 서버의
            # * 실제 현재 스테이지와 우연히 같아도 "이 턴이 알아서 처리해줄 것"이라고 오판하면 안
            # * 된다(실제 사례: stage="interest"로 잘못 부르면서 category="health"인 flag를 같이
            # * 보냈는데, 서버는 이미 health로 넘어간 뒤라 category가 우연히 일치 — 예전 코드는 이걸
            # * self-reference로 오인해 스폰을 건너뛰었고, 정작 stage 불일치로 이 턴 전체가
            # * invalid_stage로 튕겨나가 "다리 아프다" 신호가 흔적도 없이 사라졌다).
            if keyword:
                stage_state.symptom_tag = keyword
            continue

        target_state = buffer.stages[stages.StageType(category)]
        target_state.level = stages.StageLevel.TRY
        target_state.priority = 1
        target_state.severity = severity
        # 재스폰(이전에 한 번 완료됐다가 discomfort_flags로 다시 뜨는 경우) 시 이전 라운드의
        # high_severity_asks가 남아있으면 새 라운드 첫 질문부터 곧바로 2개 캡에 걸려버린다 — 새
        # 라운드는 항상 0부터 다시 센다.
        target_state.high_severity_asks = 0
        target_state.symptom_tag = keyword
        target_state.pending_probe = signal
        # probe가 이 스테이지의 "진입"을 대신한다 — 이후 next_question()이 넓은 is_entry 문항을
        # entry-priority로 또 먼저 뽑아버리면(agent/stages.py의 next_question() 참고), 방금 probe로
        # 구체적으로 캐물은 증상 얘기 직후에 다시 "전반적으로 컨디션이 어떠세요" 같은 포괄적 질문이
        # 이어져 대화가 겉돈다. 그래서 이 스테이지의 is_entry 문항(들)은 여기서 미리 소진 처리해
        # next_question()이 severity/태그로 좁힌 관련 척도 문항으로 바로 넘어가게 한다 — mark_asked()가
        # entry 형제 변형을 전부 같이 소진시키는 것과 동일한 패턴.
        for q in target_state.questions:
            if q.is_entry:
                q.asked = True
        state["logic_data"][category] = {"judgment": "문제있음", "reason": signal}
        log.info(
            "[stage] %s spawned as try(priority=1, severity=%s) via discomfort from %s (signal=%r)",
            category, severity, stage_state.stage.value, signal,
        )


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
    #
    # discomfort_flags 대신 그냥 stage/assessment로 곧장 증상을 보고하는 경우도 있다(실제 사례:
    # 첫 답변에서 discomfort_flags 없이 stage="health"+assessment="문제있음"으로 허리/머리 통증을
    # 보고했는데 이 분기가 stage/assessment를 아예 안 보고 버려서 신호가 통째로 사라짐 — "첫
    # 호출엔 stage 생략" 지시를 모델이 안 지켜도 신호 자체를 놓치면 안 된다). 유효한 stage +
    # "문제있음"이면 discomfort_flags 하나를 합성해서 같은 스폰 로직을 그대로 태운다.
    if buffer.turn_count == 0:
        synthesized_flags = list(discomfort_flags or [])
        if assessment == "문제있음":
            try:
                reported_stage = stages.StageType(stage)
            except (ValueError, TypeError):
                reported_stage = None
            if reported_stage is not None:
                synthesized_flags.append({
                    "signal": reason or f"{reported_stage.value} 관련 문제 보고",
                    "category": reported_stage.value,
                    "severity": "low",
                })
        # * 부트스트랩은 어떤 카테고리의 assessment도 실제로 적용하지 않으므로(그냥 다음 질문으로
        # * 바로 넘어감) assessed_stage=None — self-reference 오판 여지 자체를 원천 차단한다.
        _spawn_discomfort_flags(state, buffer, buffer.current(), synthesized_flags, None)
        result = _advance(state, buffer)
        if "next_step" in result:
            # * 모델이 "첫 호출엔 stage 생략" 지시를 안 지키고 자기 나름대로 stage/assessment를
            # * 채워 보내는 경우가 있다(실제 사례: 인사 답변만 듣고 stage="event"+assessment="문제없음"을
            # * 자기가 판정했다고 여김) — 그 판정 자체는 위에서 이미 무시했지만(discomfort 합성 용도
            # * 외엔 안 씀), next_step에 아무 안내가 없으면 모델이 "내가 방금 확인한 그 얘기"를
            # * 자연스럽게 이어받아 자체 질문을 하나 더 끼워 넣고(실제 사례: "혹시 최근에 기분 좋은
            # * 일이나 특별한 근황 있으셨을까요?") 그 뒤에야 서버가 지정한 진짜 질문을 붙여서, 결과적으로
            # * 한 턴에 질문이 두 개 나가버렸다. 여기서 그 판정은 무효라는 걸 못박고 아래 지정된 질문
            # * 하나만 하라고 명시한다.
            result["next_step"] = (
                "이건 통화의 첫 실제 질문입니다 — 지금까지는 인사만 나눴을 뿐, 어떤 스테이지 질문도 "
                "아직 시작되지 않았습니다. 방금 당신이 함께 보낸 판정(stage/assessment)이 있었더라도 "
                "그건 무효이니 신경 쓰지 말고, 그에 대한 반응이나 스스로 지어낸 다른 질문을 추가로 "
                "얹지 마세요. 아래 지시된 질문 하나만 자연스럽게 물어보세요: " + result["next_step"]
            )
        return json.dumps(result, ensure_ascii=False)

    # 1) LLM이 자칭한 stage가 서버가 실제로 추적 중인 현재 스테이지와 같은지 먼저 계산해둔다(에러
    # 반환은 뒤에서 한다) — discomfort_flags의 self-reference 판단(2번)에 이 결과가 필요하기 때문.
    try:
        stage_type = stages.StageType(stage)
    except ValueError:
        stage_type = None
    assessed_stage = stage_type if stage_type == buffer.active_stage else None

    # 2) discomfort_flags는 지금 판정 중인 스테이지가 뭐든 상관없이(다른 카테고리를 가리키는
    # 신호라) 먼저 처리한다 — stage 검증보다 앞에 둔 이유: 모델이 stage를 잘못 보고해서 아래에서
    # invalid_stage로 튕겨나가더라도, 같이 실려온 진짜 우려/위급 신호까지 함께 버려지면 안 된다.
    # 실제로 "지금 판정 중인 스테이지(예: meal)와 무관한 화제(예: 다리 통증)가 나오면 discomfort_flags로
    # 신고하라"는 지시를 모델이 stage 자체를 그 화제로 착각해서 잘못 부르는 경우가 있었다 — 그럴 때도
    # discomfort_flags 안의 신호는 유효하므로 여기서 먼저 살린다.
    # * self-reference 판단에 buffer.current()(서버가 실제로 추적 중인 현재 스테이지)가 아니라 위
    # * assessed_stage(모델이 자칭한 stage가 실제로 유효할 때만 채워짐)를 쓴다 — 실제 통화에서
    # * 모델이 stage="interest"(직전 스테이지, 이미 완료됨)로 잘못 부르면서 discomfort_flags에
    # * category="health"(서버가 실제로는 이미 health로 넘어간 뒤)를 같이 보낸 사례가 있었다.
    # * buffer.current()로 비교하면 우연히 category==실제 현재 스테이지라서 "이 턴의 assessment가
    # * 알아서 처리해줄 것"이라 판단해 스폰을 건너뛰는데, 실제로는 이 호출 전체가 아래에서
    # * invalid_stage로 튕겨나가 assessment가 끝내 아무 데도 반영 안 되고 신호가 통째로 사라졌다
    # * (실제 통화 사례: "다리 아프다"는 발언이 흔적도 없이 스킵됨). assessed_stage 기준이면 이런
    # * 경우 self-reference로 오인하지 않고 정상적으로 스폰한다.
    _spawn_discomfort_flags(state, buffer, buffer.current(), discomfort_flags, assessed_stage)

    # 3) 위 1번에서 계산해둔 stage_type/assessed_stage로 실제 검증 — 여기서부터 에러를 반환한다.
    if stage_type is None:
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
    if assessed_stage is None:
        return json.dumps({
            "ok": False, "error": "invalid_stage",
            "next_step": f"지금 실제로 진행 중인 스테이지는 '{buffer.active_stage.value}'입니다.",
        }, ensure_ascii=False)
    stage_state = buffer.current()

    if stage_state.awaiting_probe_answer:
        # * 방금 그 턴은 pending_probe를 캐묻는 자유 발화 확인 질문이었다(_advance의 probe 분기
        # * 참고) — 지금 들어온 assessment/reason은 실제 척도 문항 판정이 아니므로 resource_offer_pending과
        # * 동일하게 logic_data/_apply_assessment(및 high_severity_asks 카운트)를 건너뛰고 곧장 다음
        # * 질문(이제 symptom_tag로 좁혀진 관련 척도 문항)으로 넘긴다.
        stage_state.awaiting_probe_answer = False
        result = _advance(state, buffer)
        if "next_step" in result:
            result["next_step"] = (
                "공감성 발언을 짧게 건넵니다. 그리고 해당 instruction을 실행하세요: " + result["next_step"]
            )
        return json.dumps(result, ensure_ascii=False)

    if stage_state.resource_offer_pending:
        # * 방금 그 턴은 도움처 검색 제안만 하고 끝났어야 한다(3.8 참고) — 지금 들어온 assessment/reason은
        # * 그 제안에 대한 어르신 답(수락/거절)일 뿐 실제 문항 판정이 아니므로 logic_data에 쓰면 안 되고,
        # * _apply_assessment도 타면 안 된다(이미 앞선 턴에서 문항 판정은 끝났음). 플래그만 내리고 곧장
        # * 진짜 다음 질문을 내준다.
        stage_state.resource_offer_pending = False
        result = _advance(state, buffer)
        if "next_step" in result:
            result["next_step"] = (
                "공감성 발언을 짧게 건넵니다. 그리고 해당 instruction을 실행하세요: " + result["next_step"]
            )
        return json.dumps(result, ensure_ascii=False)

    if buffer.awaiting_gap_answer:
        # * 방금 그 턴은 PendingGap이 낸 "무관한 즉흥 질문"이었다(_consume_pending_gap 참고) — 지금
        # * 들어온 assessment/reason은 그 잡담에 대한 답일 뿐 실제 문항 판정이 아니므로 resource_offer_pending과
        # * 동일하게 logic_data/_apply_assessment를 건너뛰고 곧장 다음 간격(또는 강제 문항)으로 넘긴다.
        buffer.awaiting_gap_answer = False
        result = _advance(state, buffer)
        if "next_step" in result:
            result["next_step"] = (
                "공감성 발언을 짧게 건넵니다. 그리고 해당 instruction을 실행하세요: " + result["next_step"]
            )
        return json.dumps(result, ensure_ascii=False)

    if buffer.closing_concern_pending:
        # * 방금 그 턴은 CLOSING에서 나온 곁가지 걱정/질문에 공감만 하고 끝났어야 한다(아래 3번
        # * CLOSING 분기 참고) — 지금 들어온 assessment/reason은 그 곁가지에 대한 답일 뿐 마무리
        # * 문항 판정이 아니다. 여기서 곧장 CLOSING의 next_question()만 다시 뽑으면 안 된다 —
        # * 그 곁가지 답변 자체가 discomfort_flags로 다른 카테고리(예: anxiety)를 스폰했거나 이미
        # * 완료된 카테고리(예: health)를 재활성화했을 수 있는데, CLOSING에만 갇혀서 그 재활성화된
        # * 스테이지를 영영 못 물어보고 통화가 끝나버린 실제 사례가 있었다. 반드시 _advance()를
        # * 거쳐서 pick_next_stage()가 그런 긴급 후보를 CLOSING보다 먼저 고르게 한다 — CLOSING은
        # * candidates에서 항상 제외되므로, 다른 후보가 하나라도 남아있으면 자동으로 그쪽이 먼저
        # * 픽업된다(agent/stages.py의 pick_next_stage 참고).
        buffer.closing_concern_pending = False
        result = _advance(state, buffer)
        if result.get("action") == "ask_question" and result.get("stage_type") == stages.StageType.CLOSING.value:
            # * pick_next_stage()가 다른 후보를 못 찾아 CLOSING으로 되돌아온 경우에만, 방금 나눈
            # * 곁가지 얘기를 다시 "마무리로 화제 전환"하는 것처럼 어색하게 브릿지하지 않고 곧장
            # * 재질문하도록 문구를 다듬는다.
            result["next_step"] = (
                f"화제 전환 멘트 없이 곧바로, \"{result['question']}\"의 취지대로 방금 그 이야기 말고 "
                f"다른 궁금한 점이나 도움이 필요한 부분이 있는지 자연스럽게 다시 여쭤보세요. "
                f"{_ONE_QUESTION_GUARD}"
            )
        elif "next_step" in result:
            result["next_step"] = (
                "공감성 발언을 짧게 건넵니다. 그리고 해당 instruction을 실행하세요: " + result["next_step"]
            )
        return json.dumps(result, ensure_ascii=False)

    # * "문제없음"이 방금 이 카테고리에서 이미 "문제있음"으로 확인된 문항을 덮어써서 지워버리면
    # * 안 된다 — 예: 우울 카테고리가 앞선 2개 문항에서 문제있음이었는데, 극성이 헷갈리는 마지막
    # * 문항 하나를 모델이 잘못 판정해서 카테고리 전체가 문제없음으로 기록되는 실제 사례가 있었다.
    # * 이 경우 logic_data는 건드리지 않고 이전 값(이미 확인된 문제있음)을 그대로 유지한다 — 실제
    # * 대화 흐름/상태 전이(_apply_assessment)는 이 판정을 그대로 반영하므로, 여기서 막는 건 오직
    # * "보고되는 요약/채점 데이터"뿐이다.
    if not (assessment == "문제없음" and _category_had_confirmed_problem(state, stage_state)):
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
        # * 여기서 곧장 마무리 질문을 다시 묻지 않는다 — closing_concern_pending을 세우고 이번
        # * 턴은 곁가지 응답만으로 끝낸다. 마무리 재질문은 그 답을 들은 다음 check_in 호출에서
        # * 위 closing_concern_pending 게이트가 처리한다(resource_offer_pending과 동일한 이유 —
        # * 한 next_step에 "곁가지 응답"과 "마무리 재질문" 지시를 같이 실으면 모델이 한 턴에 질문을
        # * 두 개 해버렸다, 실제 통화 사례).
        stage_state.fail_current_question()
        buffer.closing_concern_pending = True
        return json.dumps({
            "ok": True, "action": "acknowledge",
            "stage_type": stage_type.value,
            "next_step": "방금 하신 말씀에 공감성 발언으로 짧게 반응하고, 필요하면 확인 질문은 "
                         "하나만 하세요 — 마무리 질문(더 궁금한 점 있는지)은 이번 턴에 같이 묻지 "
                         "마세요. 어르신 답을 들으면 같은 stage로 check_in을 다시 호출하세요 — "
                         "assessment는 아무 값이나 넣어도 됩니다. 그러면 마무리 질문을 다시 "
                         "안내해 드립니다.",
        }, ensure_ascii=False)
    else:
        # * 이 라운드에서 이미 실제로 확인된 문제(item_judgments에 "문제있음"이 기록됨)가 있는데,
        # * 지금 판정이 "문제없음"이면 — 모델이 "그거 말고 다른 데는요?"류로 바꿔 물었을 때 어르신이
        # * "발목 말고는 없어" 식으로 원래 문제를 재확인해줘도, 모델이 이걸 "전체적으로 문제없음"으로
        # * 오판하는 사례가 프롬프트를 강화했음에도(_known_issue_note) 반복됐다(실제 통화 2건). 이러면
        # * 확인 라운드가 조기 종료돼서 high_severity_asks가 2를 못 채우고 도움처 제안까지 아예
        # * 못 간다 — 그래서 여기서는 모델의 오판을 코드로 방어한다: severity가 medium/high인 확인
        # * 라운드 안에서 이미 문제있음이 한 번이라도 확인됐다면, 이번 "문제없음"은 상태 전이
        # * (_apply_assessment/severity 승격/도움처 제안 카운트)에서는 문제있음으로 취급해 라운드를
        # * 계속 이어간다 — 기존 캡(high_severity_asks>=2)이 여전히 무한 반복을 막아준다. logic_data/
        # * item_judgments 기록은 위에서 이미 실제 판정(assessment) 그대로 남겼으니 왜곡되지 않는다.
        effective_assessment = assessment
        if (
            assessment == "문제없음"
            and stage_state.severity in ("medium", "high")
            and _category_had_confirmed_problem(state, stage_state)
        ):
            effective_assessment = "문제있음"
        stage_completed = _apply_assessment(buffer, stage_state, effective_assessment)

    # 3.5~3.6은 서로 배타적이다(elif로 묶는다) — 한 턴에 severity가 한 단계만 전이하게 하기 위해서다.
    # 안 그러면 예를 들어 3.6이 방금 severity를 None->low로 올렸는데 바로 이어서 3.5가 같은 턴의
    # 같은 "문제있음" 판정을 또 보고 low->high까지 한 턴에 훌쩍 건너뛰어 버린다 — low 티어로 한 번
    # 더 확인하자는 원래 의도(성급하게 확정하지 않기)가 무너진다. 위 effective_assessment 보정과
    # 같은 이유로, 아래도 원판정이 아니라 effective_assessment를 기준으로 삼는다(CLOSING 분기는
    # else 밖이라 여기 안 옴 — effective_assessment는 else 블록에서만 정의되므로 CLOSING이었으면
    # 이 지점에 도달하지 않는다).
    #
    # 3.5) severity="low"로 스폰된 스테이지(의심은 되지만 아직 불확실한 상태 — questions.json의 low
    # 티어 문항으로 가볍게 물어보던 중)였는데, 그 low 문항에 대한 답까지 "문제있음"으로 판정되면 —
    # 즉 의심(스폰 시점 또는 3.6)에 이어 또 한 번("또") 문제가 확인된 것 — severity를 high로 즉시
    # 올린다. 이러면 next_question()이 다음부터 high 티어 문항을 물어본다.
    if stage_state.severity == "low" and effective_assessment == "문제있음":
        stage_state.severity = "high"

    # 3.6) baseline으로 승격된 스테이지(DEPRESSION/HEALTH)는 severity 없이(None) 시작해서, 인지/불안
    # 신호까지 겸해서 받는 넓은 is_entry 문항 하나만 우선 묻는다. 그 entry 답변에서 "문제있음"이
    # 나오면, 아직 확신할 단계는 아니니 곧장 high로 올리지 않고 먼저 low로 내려서 그 카테고리의
    # 부드러운 후속 문항들로 한 번 더 확인한다 — 거기서 또 문제가 확인되면 다음 턴엔 3.5가 이어받는다.
    elif (
        stage_state.current_question is not None
        and stage_state.current_question.is_entry
        and stage_state.severity is None
        and effective_assessment == "문제있음"
    ):
        stage_state.severity = "low"

    # 3.7) severity="medium"/"high"(capped_tier와 동일 범위 — 3.5번은 low->high만 처리해서 medium은
    # 계속 medium인 채로 캡에 걸릴 수 있는데, 그 경우에도 권유는 나가야 한다)로 확인 중일 때, 그
    # 티어의 문항을 캡(2개)까지 다 물어봤는데도 마지막 답까지 "문제있음"으로 나오면
    # (high_severity_asks>=2 — 이 시점엔 _apply_assessment가 이미 스테이지를 COMPLETED 처리한
    # 뒤다) 확실히 도움이 필요하다고 판단된 것이므로 관련 도움처를 물어보라고 안내한다. 첫 번째
    # 문항만 "문제있음"인 단계에서는 아직 확신할 수 없으니 권유하지 않고 남은 문항을 마저 물어본다
    # (_advance가 정상적으로 이어감).
    #
    # 예전엔 이 권유 문구를 "다음 질문" 지시와 한 next_step에 이어붙이고 "원치 않으면 그때 다음
    # 질문으로"라고 텍스트로만 순서를 설명했는데, 모델이 그 순서를 안 지키고 한 턴에 권유+다음
    # 질문을 같이 말해버리는 경우가 있었다(실제 통화 사례 — 프롬프트 문구를 아무리 강하게 써도
    # 완전히 막긴 어려움). 그래서 여기서는 텍스트가 아니라 상태(resource_offer_pending)로
    # 순서를 강제한다 — 이 턴은 권유 next_step만 돌려주고 _advance()를 아예 안 부른다. 진짜
    # 다음 질문(다른 스테이지)은 위 resource_offer_pending 게이트(이 함수 상단)가 다음 호출에서
    # 내준다 — 이 스테이지는 이미 COMPLETED라 priority를 따로 올려둘 필요가 없다.
    if (
        stage_state.severity in ("medium", "high")
        and stage_state.high_severity_asks >= 2
        and effective_assessment == "문제있음"
    ):
        stage_state.resource_offer_pending = True
        resource_hint = _RESOURCE_HINT.get(stage_type, "병원이나 상담센터 등")
        return json.dumps({
            "ok": True, "action": "offer_resource",
            "stage_type": stage_type.value,
            "next_step": f"공감성 발언을 짧게 건넨 뒤, 방금 확인된 내용과 관련해 근처 {resource_hint} "
                         "정보를 찾아봐드릴지 여쭤보고 이번 턴을 끝내세요 — 어떤 종류의 도움처인지 말하지 않고 "
                         "그냥 '도움받을 수 있는 곳'이라고만 뭉뚱그리면, 통화 중 다른 카테고리(예: 우울/불안) "
                         "얘기도 같이 나왔을 때 어르신이 어느 문제에 대한 제안인지 헷갈려 거절할 수 있습니다 "
                         f"— 반드시 위에 알려드린 종류({resource_hint})로 구체적으로 말씀드리세요. 어르신이 "
                         "원하면 check_external_api_necessity를 호출해서 안내하세요. 어르신 답을 들으면"
                         "(원하든 원치 않든) 같은 stage로 check_in을 다시 호출하세요 — assessment는 아무 "
                         "값이나 넣어도 됩니다. 그러면 다음 질문을 안내해 드립니다.",
        }, ensure_ascii=False)

    # 4) closing이 방금 완료됐으면 바로 작별 인사 안내
    if stage_completed and stage_type == stages.StageType.CLOSING:
        return json.dumps({
            "ok": True, "action": "end_call",
            "next_step": "end_call 호출하세요.",
        }, ensure_ascii=False)

    # 5) 다음 스테이지/질문 고르기 — trigger_next_turn 왕복 없이 이번 응답에 바로 실린다.
    result = _advance(state, buffer)
    if "next_step" in result:
        result["next_step"] = (
            "공감성 발언을 짧게 건넵니다. 그리고 해당 instruction을 실행하세요: " + result["next_step"]
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

    if nearby_resource:
        state["_last_offered_resources"] = nearby_resource
        names = ", ".join(r["name"] for r in nearby_resource)
        return json.dumps({
            "ok": True,
            "next_step": f"이런 곳들의 이름만 자연스럽게 안내하세요(전화번호나 주소는 말하지 마세요): "
                         f"{names}. 어르신이 더 궁금해하시거나 연락처를 원하시면 문자로 보내드리겠다고 "
                         "제안하세요.",
        }, ensure_ascii=False)

    if keyword:
        # 검색을 시도했지만(keyword가 있었음) integrations/dispatch.py가 결과를 못 채워준 경우 —
        # KAKAO_API_KEY 미설정이나 지오코딩 실패. "외부 정보 불필요"와는 다른 상황이니 구분해서 안내.
        return json.dumps({
            "ok": True,
            "next_step": "죄송하지만 지금은 근처 정보를 확인하기 어렵다고 자연스럽게 안내하세요.",
        }, ensure_ascii=False)

    return json.dumps({
        "ok": True,
        "next_step": "어르신이 병원 등 도움처 안내를 원하지 않는 것으로 보입니다. 괜찮다고 자연스럽게 "
                     "답하고 대화를 이어가세요.",
    }, ensure_ascii=False)


# --------------------------------------------------------------------------
# 4. 통화 종료
# --------------------------------------------------------------------------

def end_call(state: dict) -> str:
    # * 통화 종료 — 여기서 하는 일은 오직 "지금 무슨 작별 문구로 마무리할지" 결정뿐이다.
    next_step = "지금 바로 다음 답변으로 부드러운 작별 인사를 건네고 통화를 마치세요."
    return json.dumps({"call_ended": True, "next_step": next_step}, ensure_ascii=False)


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
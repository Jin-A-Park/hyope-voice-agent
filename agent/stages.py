# * 스테이지(StageType) 상태머신 — 통화 하나의 진행 상황을 스테이지별 우선순위/상태로 추적하고,
# * 매 턴 pick_next_stage()가 가장 시급한 스테이지를 고르는 데 쓰는 자료구조/선택 로직.
# * agent/tools.py의 check_in이 이 모듈의 CallStageBuffer를 조작한다.
# ! 네트워킹 x — agent/tools.py와 같은 원칙.

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

log = logging.getLogger("tools")  # agent/tools.py의 [stage] picked 로그와 같은 스트림에 묶이도록 동일 이름 사용

ROOT = Path(__file__).resolve().parent.parent


class StageType(Enum):
    EVENT = "event"
    INTEREST = "interest"
    ANXIETY = "anxiety"
    DEPRESSION = "depression"
    HEALTH = "health"
    MEDICATION = "medication"
    MEAL = "meal"
    COGNITIVE = "cognitive"  # 인지기능(SIS) — ANXIETY와 같은 방식으로 통화 시작 시 비활성, 어르신 답변에서
                              # 인지 저하 신호(기억 못함/앞뒤 안 맞음 등)가 감지될 때만 반응적으로 활성화됨.
    CLOSING = "closing"


class StageLevel(Enum):
    COMPLETED = "completed"
    TRY = "try"  # 아직 안 물어본 상태 — baseline이 순서를 기다리는 중이든, discomfort로 막 반응형
                 # 스폰된 직후든 동일하게 이 레벨. 급한 정도는 severity 필드가 별도로 표현한다.
    FAILED = "failed"
    VALIDATE = "validate"


LEVEL_ORDER: dict[StageLevel, int] = {
    StageLevel.COMPLETED: 0,
    StageLevel.TRY: 1,
    StageLevel.FAILED: 2,
    StageLevel.VALIDATE: 3,
}

# 스테이지 초기 우선순위값 — pick_next_stage()가 같은 레벨(TRY) 내에서 값이 큰 쪽을 먼저 고른다.
# 6개 baseline 전부 서로 다른 값을 줘서 매 통화 항상 근황 -> 취미 -> 건강 -> 기분 -> 식사 -> 복약
# 순으로 고정되게 한다(동률이 없으니 pick_next_stage()의 random.choice(top) tie-break까지 안 감).
# -1은 별도로 "이미 완료" 표시로만 쓰인다(여기 값과 안 겹치게 0부터 시작).
# COGNITIVE/ANXIETY/CLOSING은 여기 안 실려있고 별도로 비활성(completed/-1) 초기화된다
# (CallStageBuffer 참고) — 이 셋은 discomfort_flags로 반응형 스폰될 때만 활성화되고, CLOSING은
# pick_next_stage()가 다른 후보를 하나도 못 찾았을 때만 마지막 수단으로 재활성화된다.
#
# HEALTH는 baseline으로 유지하되(건강 확인 자체는 매 통화 필수 질문), 심각도가 escalate됐을 때
# 뽑히던 프레일티 지수/낙상 위험 척도 배터리(계단 오르기, 지난 4주간 피로도, 11개 질환 여부,
# 체중감소, 낙상 이력 등 — static/questions.json의 frail_*/fall_* 문항)는 대화 맥락과 무관하게
# 기계적으로 나온다는 피드백으로 questions.json에서 enabled:false 처리했다 — health_entry/sleep/
# pain/health_self_rating과 low tier는 그대로 남아있어 자연스러운 확인 질문은 계속된다.
#
# DEPRESSION/HEALTH는 baseline이지만, 매 통화 4개(우울/불안/건강/인지) 다 물어보면 피로도가 크다고
# 판단해서 이 둘의 entry 질문(questions.json의 is_entry:true)을 넓게 설계해 인지/불안 신호까지
# 겸해서 받는다 — "어제 하루 어떻게 보내셨어요"는 우울+인지, "몸/마음/잠은 어떠세요"는 건강+불안을
# 같이 겨냥한다. 거기서 뭔가 걸리면 discomfort_flags로 인지/불안이 반응형 스폰되거나, 같은 카테고리
# 안에서 severity가 low로 내려가 더 깊이 물어본다(check_in의 3.6단계 참고).
INITIAL_PRIORITY: dict[StageType, int] = {
    StageType.EVENT: 5,       # 1. 근황
    StageType.INTEREST: 4,    # 2. 취미
    StageType.HEALTH: 3,      # 3. 건강
    StageType.DEPRESSION: 2,  # 4. 기분
    StageType.MEAL: 1,        # 5. 식사
    StageType.MEDICATION: 0,  # 6. 복약
}

# 통화 시작 시 비활성(completed/-1)으로 두고, check_in의 discomfort_flags로만 반응적으로 활성화되는
# 스테이지 — 매 통화 필수인 baseline 6개(EVENT/INTEREST/MEDICATION/MEAL/DEPRESSION/HEALTH) 외 전부.
REACTIVE_ONLY_STAGES: set[StageType] = {
    StageType.COGNITIVE, StageType.ANXIETY,
}

# 통화 시작 시 비활성(completed/-1)으로 두는 전체 집합 — 위 REACTIVE_ONLY_STAGES에 더해 CLOSING도 포함한다.
# CLOSING은 discomfort_flags가 아니라 pick_next_stage()가 다른 후보를 하나도 못 찾았을 때만
# 마지막 수단으로 재활성화한다(다른 스테이지와 우선순위 경쟁을 하지 않음).
INACTIVE_AT_START: set[StageType] = REACTIVE_ONLY_STAGES | {StageType.CLOSING}

# check_in의 discomfort_flags 항목 중 category가 받을 수 있는 값 — closing 제외.
DISCOMFORT_CATEGORY_VALUES: list[str] = [
    s.value for s in StageType if s != StageType.CLOSING
]

# 문항을 하나씩 판정해서 끝내는 게 아니라, 정해진 배터리를 끝까지 다 돌아야 하는 스테이지(SIS 인지검사 등).
# check_in은 이 스테이지들에 한해 "문제없음"이 나와도 은행이 다 소진되기 전까지는 스테이지를
# 끝내지 않고 validate로 계속 진행시킨다(agent/tools.py의 _apply_assessment 참고).
FIXED_SEQUENCE_STAGES: set[StageType] = {StageType.COGNITIVE}

QUESTION_FAIL_LIMIT = 2  # 이 횟수 넘게 실패/불일치가 반복되면 더 이상 재시도하지 않고 포기 처리(무한루프 방지)


@dataclass
class QuestionCandidate:
    id: str
    text: str
    asked: bool = False
    fail_count: int = 0  # check_in 안에서 질문 불일치 또는 '부적절' 판정으로 되돌려질 때마다 +1
    depends_on: str | None = None  # 이 id의 질문은 next_question()의 무작위 후보 풀에 절대 안 들어간다 —
                                    # 선행 질문(예: sis_encode)이 asked된 뒤 정확히 gap개의 "메꿈용" 턴이
                                    # 지나면 PendingGap이 강제로 주입한다(agent/tools.py의 _mark_asked/
                                    # _consume_pending_gap 참고). 즉 "언제 물어볼지"가 무작위 선택이 아니라
                                    # 전적으로 그 강제 주입 메커니즘에 달려있다.
    gap: int = 0  # depends_on 선행 질문이 asked된 직후 강제로 끼워 넣을 "무관한 즉흥 질문" 턴 수 — 최소치가
                  # 아니라 정확히 이 개수만큼만 채우고 그 다음 턴에 곧바로 이 질문을 강제로 묻는다.
    note: str | None = None  # 모델에게 이 질문을 물을 때 지켜야 할 특수 지시(예: 정답 노출 금지)
    severity: str | None = None  # low/medium/high — static/questions.json에서 태그. 이 스테이지가 그
                                  # 심각도로 스폰됐을 때 우선 뽑힌다(StageState.next_question 참고).
                                  # 안 달려있으면(None) 기존처럼 severity 구분 없이 전체에서 무작위로 뽑힌다.
    is_entry: bool = False  # baseline으로 승격된 스테이지(DEPRESSION/HEALTH)의 "무조건 처음에 확정
                             # 출제"용 질문 표시. next_question()이 severity 티어링/무작위 선택보다
                             # 먼저 이걸 확인한다 — 안 그러면 무작위 선택 때문에 첫 질문부터 gds5_3
                             # 같은 고티어 임상 문항이 튀어나올 수 있다(entry/main 경계를 없앤 뒤로
                             # 순서 보장이 사라졌기 때문). discomfort_flags로 스폰될 때는(HEALTH가
                             # 아직 baseline 차례를 못 만난 상태에서 신호가 먼저 뜬 경우) probe가 이
                             # 자리를 대신하고 entry는 미리 소진 처리된다(agent/tools.py의
                             # _spawn_discomfort_flags 참고).
    improvise: bool = False  # GDS-5/GAD-2/SIS 같은 본 문항과 달리 entry는 검증된 척도가 아니라 그냥
                              # 자연스럽게 떠보려고 직접 지은 지시문이다 — text를 그대로 읽게 고정하는 대신,
                              # text/note를 모델이 따라야 할 지시로 주고 대화 흐름에 맞춰 직접 질문을
                              # 지어내게 한다(agent/tools.py의 _question_goal 참고).
    tags: list[str] = field(default_factory=list)  # discomfort_flags의 keyword(예: "pain", "fall")와
                              # 매칭할 통제 어휘 — StageState.next_question()이 severity tier로 좁힌
                              # 후보를 이 태그로 한 번 더 좁혀, 방금 신고된 구체적 증상과 무관한 문항
                              # (예: 두통 신고인데 frail_stairs)이 먼저 뽑히지 않게 한다. 안 달려있으면
                              # 기존처럼 태그 구분 없이 취급된다.


@dataclass
class PendingGap:
    """depends_on 질문(예: sis_recall)을 강제로 끼워 넣기 위해 CallStageBuffer가 들고 있는 대기 상태.
    선행 질문(sis_encode)이 asked된 순간 agent/tools.py의 _mark_asked()가 생성한다 — 이게 있는 동안
    _advance()는 pick_next_stage()의 무작위 선택을 완전히 건너뛰고 이 gap부터 처리한다(다른 스테이지도
    포함해서 전부 멈춘다 — 기억 검사 중간에 엉뚱한 실제 문항이 리허설 단서로 끼어들면 안 되므로)."""
    stage: StageType
    question_id: str
    fillers_remaining: int  # 이 개수만큼 무관한 즉흥 질문을 먼저 채운 뒤 question_id를 강제로 묻는다


def _load_question_bank() -> dict[StageType, list[QuestionCandidate]]:
    # * static/questions.json(스테이지당 문항 배열 하나)을 읽어 StageType별 QuestionCandidate 목록으로
    # * 변환한다. 예전엔 "무조건 먼저 묻는 entry 하나 + priority로 고르는 main 목록"으로 나뉘어 있었는데,
    # * 그 구분을 없앴다 — 옛 entry였던 문항도 그냥 priority=0(기본값, JSON에 안 적으면 자동으로 0)으로
    # * 두면 다른 문항(priority>=1)보다 항상 먼저 뽑히므로, 특별 취급 코드 없이 우선순위 하나로 통일된다.

    # * "enabled": false는 QuestionCandidate 필드가 아니라 로더 전용 스위치다 — JSON엔 진짜 주석이
    # * 없어서, 테스트 중 특정 문항을 무작위 풀에서 빼고 싶을 때(예: sis_day/month/year를 잠시 끄고
    # * sis_encode->sis_recall 흐름만 확인) 항목을 지우지 않고 이 키만 false로 둔다 — true로 되돌리면
    # * 바로 복원된다.
    raw: dict = json.loads((ROOT / "static" / "questions.json").read_text())
    bank: dict[StageType, list[QuestionCandidate]] = {}
    for stage_value, items in raw.items():
        stage = StageType(stage_value)
        bank[stage] = [
            QuestionCandidate(**{k: v for k, v in item.items() if k != "enabled"})
            for item in items
            if item.get("enabled", True)
        ]
    return bank


QUESTION_BANK: dict[StageType, list[QuestionCandidate]] = _load_question_bank()


@dataclass
class StageState:
    """스테이지 하나의 상태 + 그 스테이지 안에서의 질문 진행 로직"""
    stage: StageType
    level: StageLevel = StageLevel.TRY
    priority: int = 0
    severity: str | None = None  # low/medium/high — new 스테이지 생성 시점에만 채워짐, 나머지는 None
    # severity="high"로 확인 중인 동안 실제로 몇 번 물어봤는지 — agent/tools.py의
    # _apply_assessment가 "최소/최대 2개까지만 확인하고 그 이상은 리미트로 보고 넘어간다"는 캡을
    # 걸 때 쓴다. 스테이지가 재스폰될 때(discomfort_flags로 다시 TRY가 될 때) 0으로 리셋해야
    # 이전 라운드에서 이미 2를 채운 카운트가 새 라운드에 그대로 넘어와 첫 질문부터 즉시 캡에
    # 걸려버리는 걸 막는다(agent/tools.py의 _spawn_discomfort_flags 참고).
    high_severity_asks: int = 0
    # * agent/tools.py의 check_in 3.8번(도움처 검색 제안)이 True로 세운다 — 제안과 "다음 질문"을
    # * 한 next_step에 같이 실어보내면 모델이 한 턴에 둘 다 말해버리는 문제가 있어서(실제 통화
    # * 사례), 그 턴엔 제안만 돌려주고 다음 질문은 이 플래그가 다시 False로 돌아온 그다음 호출
    # * 에야 내준다 — 프롬프트 문구가 아니라 상태로 순서를 강제한다.
    resource_offer_pending: bool = False
    # discomfort_flags가 이 스테이지를 스폰/승격시킬 때(agent/tools.py의 _spawn_discomfort_flags)
    # 같이 채워지는, 아직 척도 문항으로 들어가기 전에 자연어로 한 번 더 캐물어야 할 원본 signal —
    # 있는 동안은 _advance()가 next_question() 대신 이 내용을 그대로 캐묻는 probe next_step을
    # 돌려주고 스스로 None으로 비운다(agent/tools.py 참고).
    pending_probe: str | None = None
    # 방금 그 턴이 pending_probe를 캐묻는 질문이었다는 표시 — resource_offer_pending과 동일한
    # 패턴으로, 그 답은 실제 척도 문항이 아니므로 _apply_assessment(및 high_severity_asks 카운트)를
    # 건너뛰고 곧장 다음 질문으로 넘긴다(agent/tools.py의 check_in 참고).
    awaiting_probe_answer: bool = False
    # discomfort_flags의 선택적 keyword — next_question()이 severity tier로 좁힌 후보를 이 값으로
    # 한 번 더 좁혀 방금 신고된 증상과 가장 관련 있는 문항을 우선 선택하는 데 쓴다.
    symptom_tag: str | None = None
    current_question: QuestionCandidate | None = None
    questions: list[QuestionCandidate] = field(init=False)

    def __post_init__(self):
        # 전역 QUESTION_BANK를 직접 참조하면 asked 플래그가 통화 간에 공유돼버림 -> 콜별 독립 사본 필요
        self.questions = copy.deepcopy(QUESTION_BANK[self.stage])

    def next_question(self) -> QuestionCandidate | None:
        # depends_on이 있는 질문(예: sis_recall)은 여기 무작위 풀에 절대 안 들어간다 — PendingGap이
        # 정확한 타이밍에 강제로만 주입한다(agent/tools.py의 _mark_asked/_consume_pending_gap 참고).
        remaining = [q for q in self.questions if not q.asked and q.depends_on is None]
        if not remaining:
            return None
        # is_entry 문항이 안 물어본 채 남아있으면 severity/일반 무작위 선택보다 먼저 그것부터 출제
        # 한다 — baseline으로 승격된 스테이지가 뭐가 됐든 첫 질문은 반드시 entry 문항이어야 한다
        # (일반 무작위 선택에 맡기면 첫 턴부터 고티어 임상 문항이 나올 수 있다). 같은 카테고리에
        # entry 변형이 여러 개(예: depression_entry/_2/_3) 있으면 그중 하나만 무작위로 골라 묻고,
        # 나머지 변형은 mark_asked()가 이번 통화에서 아예 못 쓰게 소진시킨다 — 통화마다 같은 문구만
        # 반복되지 않게.
        entry_candidates = [q for q in remaining if q.is_entry]
        if entry_candidates:
            return random.choice(entry_candidates)
        # 이 스테이지가 특정 심각도로 스폰됐으면(discomfort_flags) 그 심각도로 태그된 문항만 쓴다 —
        # 예를 들어 severity="low"로 들어왔으면 questions.json에 severity:"low"로 태그된 문항만 물어보고,
        # 그게 다 소진되면(더 못 물어볼 게 없으면) None을 돌려줘서 스테이지를 끝낸다 — 다른 심각도
        # 문항으로 새지 않는다(가벼운 신호였는데 하다 보니 고심각도 문항까지 다 물어보는 걸 방지).
        if self.severity:
            tier = self.severity
            has_tier = any(q.severity == tier for q in self.questions)
            # severity="medium"용 문항은 questions.json 어디에도 없다 — 그렇다고 "필터 없음"으로
            # 풀어버리면 애매한(medium) 의심인데도 low 티어의 가벼운 잡담 문항이 섞여 뽑히고, 그걸
            # 그대로 high_severity_asks 캡(2회)까지 세버려서 진짜 진단성 high 문항은 한 번도 안
            # 물어본 채 도움처 제안이 나가는 문제가 생긴다(실제 통화에서 발견됨) — medium은 low보다
            # high 쪽에 가까운 의심이므로, 전용 문항이 없으면 high 티어로 승격해서 취급한다.
            if not has_tier and tier == "medium":
                tier = "high"
                has_tier = any(q.severity == tier for q in self.questions)
            # 그래도 태그된 문항이 하나도 없으면(카테고리 자체에 severity 태그를 아예 안 단 경우)
            # 기존처럼 전체에서 무작위로 고른다 — 하위호환.
            if has_tier:
                remaining = [q for q in remaining if q.severity == tier]
                if not remaining:
                    return None
        # discomfort_flags가 keyword를 같이 보고했으면(예: "두통" -> "pain"), severity tier로 좁힌
        # 후보 중에서도 그 증상과 태그가 일치하는 문항을 우선한다 — 방금 신고한 증상과 무관한
        # 문항(예: 두통 신고인데 frail_stairs)이 먼저 뽑히는 걸 막는다. 태그가 일치하는 문항이
        # 하나도 없으면(태그를 아예 안 단 카테고리 포함) 기존처럼 tier 전체에서 고른다 — 하위호환.
        if self.symptom_tag:
            tag_matched = [q for q in remaining if self.symptom_tag in q.tags]
            if tag_matched:
                remaining = tag_matched
        # 순서를 고정하지 않고 매번 무작위로 고른다 — 통화마다 같은 카테고리 안에서도 질문 순서가
        # 달라져서 기계적으로 안 느껴지게 한다. depends_on/severity/태그 필터링은 위에서 이미
        # 끝났으므로 여기 남은 후보는 전부 "지금 당장 물어봐도 되는" 것들뿐이다.
        return random.choice(remaining)

    def mark_asked(self, q: QuestionCandidate) -> None:
        q.asked = True
        self.current_question = q
        log.info("[question] selected: %s", json.dumps(dataclasses.asdict(q), ensure_ascii=False))
        if q.is_entry:
            # entry 변형 중 하나를 골라 물었으면 나머지 형제 변형은 이번 통화에서 다시 후보로 안
            # 나오게 asked=True로 같이 소진시킨다 — 안 그러면 다음 턴에 next_question()이 남은
            # 다른 entry 변형을 또 골라버린다(한 통화에 entry는 딱 하나만 나와야 한다).
            for sibling in self.questions:
                if sibling.is_entry and sibling is not q:
                    sibling.asked = True
        # depends_on 질문(예: sis_recall)의 강제 주입 예약은 여기서 하지 않는다 — StageState는
        # CallStageBuffer(PendingGap을 들고 있는 쪽)를 모르기 때문에, agent/tools.py의 _mark_asked()가
        # 이 메서드를 감싸면서 처리한다.

    def fail_current_question(self) -> bool:
        """check_in 안에서 질문 불일치 또는 '부적절' 판정 시 호출.
        fail_count를 올리고, 한계 이내면 asked를 되돌려 재시도 가능하게 하고 True를 반환한다.
        한계를 넘으면 asked=True를 그대로 유지(포기)하고 False를 반환한다 —
        이후 next_question()이 이 질문을 건너뛰고 다음 질문으로(또는 은행 소진과 동일하게) 넘어간다."""
        q = self.current_question
        q.fail_count += 1
        if q.fail_count > QUESTION_FAIL_LIMIT:
            return False
        q.asked = False
        return True


@dataclass
class CallStageBuffer:
    """통화 하나 전체의 스테이지 상태 모음 + 활성 스테이지 전환"""
    call_id: str
    stages: dict[StageType, StageState] = field(init=False)
    # 통화의 첫 check_in() 호출이 곧바로 덮어쓰므로 어떤 값이든 상관없다 — 임의로 EVENT.
    active_stage: StageType = StageType.EVENT
    turn_count: int = 0  # pick_next_stage()가 호출될 때마다 +1 — buffer.turn_count == 0으로 통화의
                          # 첫 check_in 호출(아직 스테이지가 하나도 안 뽑힌 부트스트랩 시점)을 구분한다.
    # agent/tools.py의 _question_next_step()이 "일상" 묶음(근황/취미/복약/식사/우울/건강)에
    # 처음 진입할 때 딱 한 번만 "일상 쪽으로 화제 전환" 브릿지를 쓰고, 그 뒤로 묶음 안을 오갈
    # 때는 재브릿지 없이 이어서 묻게 하기 위한 콜 전체 플래그.
    entered_daily_life: bool = False
    # depends_on 문항(예: sis_recall)을 강제로 끼워 넣기 위한 대기 상태 — 있는 동안 _advance()는
    # pick_next_stage()를 아예 안 부르고 이것부터 처리한다(agent/tools.py의 _consume_pending_gap).
    pending_gap: PendingGap | None = None
    # pending_gap이 무관한 즉흥 질문을 낸 직후에만 True — 그 답변에 대한 다음 check_in 호출은
    # 실제 판정(logic_data/문항 채점) 없이 곧장 _consume_pending_gap으로 흘려보내야 하므로,
    # check_in 상단에서 이 플래그로 게이트한다(resource_offer_pending과 동일한 패턴).
    awaiting_gap_answer: bool = False
    # CLOSING 단계에서 "문제없음"이 아닌 답(곁가지 질문/걱정거리 등)이 나온 직후에만 True — 그
    # 답을 받아준 턴과 마무리 질문("더 궁금한 거 있으세요?")을 다시 묻는 턴을 분리하기 위한
    # 게이트다(resource_offer_pending/awaiting_gap_answer와 동일한 패턴). 이게 없으면 한 next_step
    # 안에 "곁가지 응답"과 "마무리 재질문" 지시가 같이 실려서, 모델이 한 턴에 질문을 두 개
    # (예: "아직도 불편하세요?" + "그 밖에 궁금한 거 있으세요?") 말해버리는 문제가 있었다
    # (agent/tools.py의 check_in CLOSING 분기 참고).
    closing_concern_pending: bool = False

    def __post_init__(self):
        self.stages = {
            stage_type: StageState(
                stage=stage_type,
                level=StageLevel.COMPLETED if stage_type in INACTIVE_AT_START else StageLevel.TRY,
                priority=-1 if stage_type in INACTIVE_AT_START else INITIAL_PRIORITY[stage_type],
            )
            for stage_type in StageType
        }

    def current(self) -> StageState:
        return self.stages[self.active_stage]

    def set_active_stage(self, next_stage: StageType) -> None:
        # 어떤 스테이지를 다음으로 고를지(level/priority 비교)는 pick_next_stage()가 결정 -> 여기선 전환만
        self.active_stage = next_stage


def pick_next_stage(buffer: CallStageBuffer) -> StageType | None:
    # * agent/tools.py의 _advance()(check_in이 부트스트랩/일반 호출 모두에서 공유)가 호출하는 선택 로직.
    # * _advance()는 buffer.pending_gap이 있으면 이 함수를 아예 안 부르고 _consume_pending_gap()으로
    # * 먼저 처리한다 — depends_on 문항(sis_recall 등) 강제 주입은 여기서 다루지 않는다.
    # * 0) 호출될 때마다 턴 시계를 하나 진행시킨다(buffer.turn_count==0으로 부트스트랩 판별에 씀).
    # * 1) severity가 채워진 TRY 스테이지(아직 안 물어본 상태로 discomfort_flags에 의해 막 스폰됐거나,
    #      또는 baseline 대기 중 severity가 붙은 경우)가 하나라도 있으면, 다른 정렬 다 무시하고
    #      그중 최우선 선택(severity 높은 순 -> 동률이면 랜덤) — 안전 예외.
    # * 2) 그 외엔 CLOSING을 뺀 나머지 중에서 LEVEL_ORDER가 1차 기준(값이 클수록 먼저), priority가
    #      2차 기준(값이 클수록 먼저), 둘 다 동률이면 랜덤. CLOSING은 다른 스테이지와 우선순위를
    #      다투지 않는다 — 그래서 아래 후보 목록에서 항상 제외하고, 정말 아무 후보도 안 남았을 때만
    #      마지막 수단으로 재활성화한다.

    buffer.turn_count += 1
    candidates = [
        s for s in buffer.stages.values()
        if s.level != StageLevel.COMPLETED and s.stage != StageType.CLOSING
        and s.next_question() is not None
    ]
    if not candidates:
        closing = buffer.stages[StageType.CLOSING]
        closing.level = StageLevel.TRY
        closing.priority = 0
        return StageType.CLOSING

    urgent_new = [s for s in candidates if s.level == StageLevel.TRY and s.severity is not None]
    if urgent_new:
        severity_rank = {"high": 2, "medium": 1, "low": 0}
        max_rank = max(severity_rank[s.severity] for s in urgent_new)
        top = [s for s in urgent_new if severity_rank[s.severity] == max_rank]
        return random.choice(top).stage

    max_level = max(LEVEL_ORDER[s.level] for s in candidates)
    same_level = [s for s in candidates if LEVEL_ORDER[s.level] == max_level]
    max_priority = max(s.priority for s in same_level)
    top = [s for s in same_level if s.priority == max_priority]
    return random.choice(top).stage


# --------------------------------------------------------------------------
# 콜별 개인화 — INTEREST/EVENT는 recipient profile(취미/최근 특이사항)로 질문은행을 새로 구성한다.
# 나머지 스테이지는 위 전역 QUESTION_BANK를 그대로(deepcopy로) 쓴다.
# --------------------------------------------------------------------------

def _build_personalized_questions(
    entry_id: str, fallback_entry_text: str, item_id_prefix: str, item_question_template: str,
    names: list[str],
) -> list[QuestionCandidate]:
    if not names:
        return [QuestionCandidate(id=entry_id, text=fallback_entry_text, severity="high", improvise=True)]
    shuffled = random.sample(names, len(names))
    return [
        QuestionCandidate(
            id=f"{item_id_prefix}_{i}",
            text=item_question_template.format(name=name),
            severity="high",
        )
        for i, name in enumerate(shuffled, start=1)
    ]


def build_interest_questions(profile: dict) -> list[QuestionCandidate]:
    # * 기존 agent/brain.py의 _profile_to_block()과 같은 개인화 의도 — 취미는 INTEREST 전용으로 분리.
    hobbies = list(profile.get("hobbies") or [])
    return _build_personalized_questions(
        entry_id="interest_entry",
        fallback_entry_text=QUESTION_BANK[StageType.INTEREST][0].text,
        item_id_prefix="hobby",
        item_question_template="평소 즐겨하시던 '{name}', 요즘도 하고 계세요?",
        names=hobbies,
    )


def build_event_questions(profile: dict) -> list[QuestionCandidate]:
    # * 최근 특이사항(recent_events)은 EVENT 전용으로 분리 — hobbies와 달리 event는 원래 고유 id를 갖고
    # * 있었지만(update_recipient_profile의 kind="event" 삭제 대상), 여기선 텍스트만 문항화하면 충분하다.
    events = [e.get("text", "") for e in (profile.get("recent_events") or []) if e.get("text")]
    return _build_personalized_questions(
        entry_id="event_entry",
        fallback_entry_text=QUESTION_BANK[StageType.EVENT][0].text,
        item_id_prefix="event",
        item_question_template="최근에 '{name}' 하셨다고 들었는데, 어떠셨어요?",
        names=events,
    )


active_calls: dict[str, CallStageBuffer] = {}


def get_or_create_buffer(call_id: str, profile: dict | None = None) -> CallStageBuffer:
    if call_id not in active_calls:
        buffer = CallStageBuffer(call_id=call_id)
        profile = profile or {}
        buffer.stages[StageType.INTEREST].questions = build_interest_questions(profile)
        buffer.stages[StageType.EVENT].questions = build_event_questions(profile)
        active_calls[call_id] = buffer
    return active_calls[call_id]

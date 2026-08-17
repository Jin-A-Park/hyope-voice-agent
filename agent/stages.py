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
    EMERGENCY = "emergency"
    COGNITIVE = "cognitive"  # 인지기능(SIS) — EMERGENCY와 같은 방식으로 통화 시작 시 비활성, 어르신 답변에서
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

# 스테이지 초기 우선순위값(같은 레벨 내에서 최우선순위가 같으면 랜덤으로 선택).
# -1 - 이미 완료해서 더이상 체크할 필요 없음 / 0 - 아직 안 물어봄 / 1,2 - 같은 시급도 중 먼저 물어봄.
# EMERGENCY/COGNITIVE/ANXIETY/CLOSING은 여기 안 실려있고 별도로 비활성(completed/-1) 초기화된다
# (CallStageBuffer 참고) — 이 넷은 discomfort_flags로 반응형 스폰될 때만 활성화되고, CLOSING은
# pick_next_stage()가 다른 후보를 하나도 못 찾았을 때만 마지막 수단으로 재활성화된다.
#
# DEPRESSION/HEALTH는 baseline이지만, 매 통화 4개(우울/불안/건강/인지) 다 물어보면 피로도가 크다고
# 판단해서 이 둘의 entry 질문(questions.json의 is_entry:true)을 넓게 설계해 인지/불안 신호까지
# 겸해서 받는다 — "어제 하루 어떻게 보내셨어요"는 우울+인지, "몸/마음/잠은 어떠세요"는 건강+불안을
# 같이 겨냥한다. 거기서 뭔가 걸리면 discomfort_flags로 인지/불안이 반응형 스폰되거나, 같은 카테고리
# 안에서 severity가 low로 내려가 더 깊이 물어본다(check_in의 3.7단계 참고).
INITIAL_PRIORITY: dict[StageType, int] = {
    StageType.EVENT: 1,
    StageType.INTEREST: 1,
    StageType.DEPRESSION: 1,
    StageType.HEALTH: 1,
    StageType.MEDICATION: 0,
    StageType.MEAL: 0,
}

# 통화 시작 시 비활성(completed/-1)으로 두고, check_in의 discomfort_flags로만 반응적으로 활성화되는
# 스테이지 — 매 통화 필수인 baseline 6개(EVENT/INTEREST/MEDICATION/MEAL/DEPRESSION/HEALTH) 외 전부.
REACTIVE_ONLY_STAGES: set[StageType] = {
    StageType.EMERGENCY, StageType.COGNITIVE, StageType.ANXIETY,
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
    depends_on: str | None = None  # 이 id의 질문이 먼저 asked된 뒤에야 이 질문이 후보가 됨(예: sis_recall -> sis_encode)
    min_gap: int = 0  # depends_on 질문이 asked된 시점(turn)로부터 최소 이만큼의 "다른 스테이지" 턴이 지나야 후보가 됨
    unlocked_at_turn: int | None = None  # depends_on이 asked될 때 turn+min_gap으로 채워짐(StageState.mark_asked)
    note: str | None = None  # 모델에게 이 질문을 물을 때 지켜야 할 특수 지시(예: 정답 노출 금지)
    severity: str | None = None  # low/medium/high — static/questions.json에서 태그. 이 스테이지가 그
                                  # 심각도로 스폰됐을 때 우선 뽑힌다(StageState.next_question 참고).
                                  # 안 달려있으면(None) 기존처럼 severity 구분 없이 전체에서 무작위로 뽑힌다.
    is_entry: bool = False  # baseline으로 승격된 스테이지(DEPRESSION/HEALTH)의 "무조건 처음에 확정
                             # 출제"용 질문 표시. next_question()이 severity 티어링/무작위 선택보다
                             # 먼저 이걸 확인한다 — 안 그러면 무작위 선택 때문에 첫 질문부터 gds5_3
                             # 같은 고티어 임상 문항이 튀어나올 수 있다(entry/main 경계를 없앤 뒤로
                             # 순서 보장이 사라졌기 때문).
    improvise: bool = False  # GDS-5/GAD-2/SIS 같은 본 문항과 달리 entry는 검증된 척도가 아니라 그냥
                              # 자연스럽게 떠보려고 직접 지은 문구다 — text를 그대로 읽게 고정하는 대신,
                              # text/note를 "이런 걸 파악하라"는 목표로만 주고 모델이 대화 흐름에 맞춰
                              # 직접 질문을 지어내게 한다(agent/tools.py의 _question_next_step 참고).
    angles: list[str] | None = None  # improvise 질문에서, text가 늘 같은 표현으로 수렴하는 걸 막기
                                      # 위한 관점 후보들 — 통화마다 하나만 랜덤으로 골라 그걸 목표로
                                      # 준다(agent/tools.py의 _question_goal 참고). 없으면 text를 그대로 씀.


def _load_question_bank() -> dict[StageType, list[QuestionCandidate]]:
    # * static/questions.json(스테이지당 문항 배열 하나)을 읽어 StageType별 QuestionCandidate 목록으로
    # * 변환한다. 예전엔 "무조건 먼저 묻는 entry 하나 + priority로 고르는 main 목록"으로 나뉘어 있었는데,
    # * 그 구분을 없앴다 — 옛 entry였던 문항도 그냥 priority=0(기본값, JSON에 안 적으면 자동으로 0)으로
    # * 두면 다른 문항(priority>=1)보다 항상 먼저 뽑히므로, 특별 취급 코드 없이 우선순위 하나로 통일된다.

    raw: dict = json.loads((ROOT / "static" / "questions.json").read_text())
    bank: dict[StageType, list[QuestionCandidate]] = {}
    for stage_value, items in raw.items():
        stage = StageType(stage_value)
        bank[stage] = [QuestionCandidate(**item) for item in items]
    return bank


QUESTION_BANK: dict[StageType, list[QuestionCandidate]] = _load_question_bank()


@dataclass
class StageState:
    """스테이지 하나의 상태 + 그 스테이지 안에서의 질문 진행 로직"""
    stage: StageType
    level: StageLevel = StageLevel.TRY
    priority: int = 0
    severity: str | None = None  # low/medium/high — new 스테이지 생성 시점에만 채워짐, 나머지는 None
    # severity="high"인데 모델이 확신하지 못한 채(discomfort_flags의 certain=false) 스폰된 경우 False.
    # 이 상태면 _write_dashboard_alert를 아직 안 보낸 것 — agent/tools.py의 _advance/check_in이 확인
    # 질문 라운드를 한 번 강제하고, 그 판정이 나온 뒤에야 True로 바뀌며 확정된다(모델의 재신고 없이도
    # 서버가 직접 관리하는 상태라, 모델이 확신한다고 우겨도 이 라운드를 건너뛸 수 없다).
    confirmed: bool = True
    # severity="high"(EMERGENCY 제외)로 확인 중인 동안 실제로 몇 번 물어봤는지 — agent/tools.py의
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
    current_question: QuestionCandidate | None = None
    questions: list[QuestionCandidate] = field(init=False)

    def __post_init__(self):
        # 전역 QUESTION_BANK를 직접 참조하면 asked 플래그가 통화 간에 공유돼버림 -> 콜별 독립 사본 필요
        self.questions = copy.deepcopy(QUESTION_BANK[self.stage])

    def next_question(self, current_turn: int = 0) -> QuestionCandidate | None:
        remaining = [q for q in self.questions if not q.asked and self._is_unlocked(q, current_turn)]
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
        # 이 카테고리에 그 심각도로 태그된 문항이 애초에 하나도 없으면(태그 자체를 안 단 경우) 기존처럼
        # 전체에서 무작위로 고른다 — 하위호환.
        if self.severity:
            has_tier = any(q.severity == self.severity for q in self.questions)
            if has_tier:
                remaining = [q for q in remaining if q.severity == self.severity]
                if not remaining:
                    return None
        # 순서를 고정하지 않고 매번 무작위로 고른다 — 통화마다 같은 카테고리 안에서도 질문 순서가
        # 달라져서 기계적으로 안 느껴지게 한다. depends_on/severity 필터링은 위에서 이미 끝났으므로
        # 여기 남은 후보는 전부 "지금 당장 물어봐도 되는" 것들뿐이다.
        return random.choice(remaining)

    @staticmethod
    def _is_unlocked(q: QuestionCandidate, current_turn: int) -> bool:
        # depends_on이 없으면 처음부터 후보. 있으면 그 선행 질문이 실제로 asked돼서
        # unlocked_at_turn이 찍히고, 그 턴까지 지나야 후보가 된다 — unlocked_at_turn이 아직
        # None이면(선행 질문을 안 물어봤으면) 무조건 잠긴 상태다.
        #
        # 예전엔 이 None 체크가 반대로 "제한 없음"으로 취급돼서, sis_recall(depends_on=sis_encode)이
        # sis_encode를 묻기도 전에 이미 후보에 들어가 있었다 — priority 정렬(sis_encode가 낮은 값)이
        # 우연히 먼저 뽑히게 가려주고 있었을 뿐, 실제로 막고 있는 게 아니었다. 순서를 무작위로 바꾸면
        # 이 우연한 보호막이 사라져서 sis_recall이 먼저 뽑힐 수 있었으므로 여기서 제대로 고쳤다.
        if q.depends_on is None:
            return True
        return q.unlocked_at_turn is not None and current_turn >= q.unlocked_at_turn

    def has_pending_gap_items(self, current_turn: int) -> bool:
        # * next_question()이 지금 당장은 None이어도, min_gap 때문에 나중 턴에 나올 문항이 남아있는지.
        # * 있으면 "은행 소진"이 아니라 "대기 중"이므로 스테이지를 completed 처리하면 안 된다.
        return any(
            not q.asked and q.unlocked_at_turn is not None and q.unlocked_at_turn > current_turn
            for q in self.questions
        )

    def mark_asked(self, q: QuestionCandidate, current_turn: int = 0) -> None:
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
        for other in self.questions:
            # min_gap이 0(기본값)이어도 선행 질문이 asked됐다는 사실 자체는 기록해야 _is_unlocked가
            # 풀어준다 — 예전엔 "and other.min_gap"으로 min_gap=0인 depends_on 문항은 이 turn 자체가
            # 안 찍혀서 영원히 잠긴 채로 남았다(당장은 sis_recall만 depends_on을 쓰고 min_gap=2라
            # 드러나지 않았을 뿐).
            if other.depends_on == q.id:
                other.unlocked_at_turn = current_turn + other.min_gap

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
    turn_count: int = 0  # pick_next_stage()가 호출될 때마다 +1 — min_gap 계산의 기준 시계
    # agent/tools.py의 _question_next_step()이 "일상" 묶음(근황/취미/복약/식사/우울/건강)에
    # 처음 진입할 때 딱 한 번만 "일상 쪽으로 화제 전환" 브릿지를 쓰고, 그 뒤로 묶음 안을 오갈
    # 때는 재브릿지 없이 이어서 묻게 하기 위한 콜 전체 플래그.
    entered_daily_life: bool = False

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
    # * 0) 호출될 때마다 턴 시계를 하나 진행시킨다(min_gap 계산 기준).
    # * 1) severity가 채워진 TRY 스테이지(아직 안 물어본 상태로 discomfort_flags에 의해 막 스폰됐거나,
    #      또는 baseline 대기 중 severity가 붙은 경우)가 하나라도 있으면, 다른 정렬 다 무시하고
    #      그중 최우선 선택(severity 높은 순 -> 동률이면 EMERGENCY 우선, 그래도 동률이면 랜덤) —
    #      안전 예외. EMERGENCY를 우선하는 이유: 위급 신호를 확인하던 도중 어르신이 같은 증상을
    #      다른 카테고리(예: health)로도 겸해 언급하면 그 카테고리가 severity="high"로 같이
    #      뜨는데, 여기서 랜덤에 맡기면 확인 라운드 도중에 응급 대응이 아니라 엉뚱한 baseline
    #      질문(지병 여부 등)으로 새버릴 수 있다 — 실제 통화에서 발견된 문제.
    # * 2) 그 외엔 CLOSING을 뺀 나머지 중에서 LEVEL_ORDER가 1차 기준(값이 클수록 먼저), priority가
    #      2차 기준(값이 클수록 먼저), 둘 다 동률이면 랜덤. CLOSING은 다른 스테이지와 우선순위를
    #      다투지 않는다 — 그래서 아래 후보 목록에서 항상 제외하고, 정말 아무 후보도 안 남았을 때만
    #      마지막 수단으로 재활성화한다.
    # * 3) 지금 당장 물어볼 문항이 없는데 min_gap 대기 중인 문항만 남은 스테이지(예: sis_recall 대기
    #      중인 COGNITIVE)는 후보에서 아예 뺀다 — 그래야 그 사이에 다른 스테이지가 실제로 끼어든다.

    buffer.turn_count += 1
    candidates = [
        s for s in buffer.stages.values()
        if s.level != StageLevel.COMPLETED and s.stage != StageType.CLOSING
        and not (
            s.next_question(buffer.turn_count) is None
            and s.has_pending_gap_items(buffer.turn_count)
        )
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
        emergency_top = [s for s in top if s.stage == StageType.EMERGENCY]
        if emergency_top:
            return emergency_top[0].stage
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
        return [QuestionCandidate(id=entry_id, text=fallback_entry_text, severity="high")]
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

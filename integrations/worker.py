from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from integrations import sms

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("worker")

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = ROOT / "runtime" # brain.py가 쓰는 append 로그/원장류 임시 파일 전용 폴더 — agent/brain.py와 공유

GUARDIAN_ALERTS_FILE = RUNTIME_DIR / "alerts" / "inbox.jsonl"
PROCESSED_FILE = RUNTIME_DIR / "alerts" / "processed.txt"
POLL_SECONDS = 2

# server.py가 통화 종료 시 여기 한 줄씩 남긴다(GUARDIAN_ALERTS_FILE과 같은 내구성 패턴) —
# 실제 Spring 전송과 채점 모델 호출은 이 파일을 폴링하며 여기서 처리한다.
CALL_RESULT_OUTBOX = RUNTIME_DIR / "call_results" / "outbox.jsonl"
CALL_RESULT_PROCESSED_FILE = RUNTIME_DIR / "call_results" / "processed.txt"
SPRING_BASE_URL = os.environ.get("SPRING_BASE_URL", "http://localhost:8080")


def push_guardian_alert(alert: dict) -> None:
    # high-severity 알림을 Spring에 구조화된 데이터로 넘긴다 — 대시보드 알림 생성과 보호자
    # 연락처 조회는 그 정보를 이미 갖고 있는 Spring이 처리한다(push_call_result와 동일한 패턴).
    # 다만 Spring엔 SMS 발신 수단이 없으므로, 실제 문자 발송은 여기서 기존 ClawOps 클라이언트로
    # 한다 — Spring이 응답으로 돌려준 보호자 번호를 그대로 쓴다.
    payload = {
        "recipient_id": (alert.get("metadata") or {}).get("recipient_id"),
        "signal": alert["alert"],
        "severity": "high",
        "logic_data": alert["logic_data"],
        "filed_at": alert["filed_at"],
    }
    resp = requests.post(
        f"{SPRING_BASE_URL}/internal/emergency-alerts",
        json=payload,
        headers={"X-API-KEY": os.environ.get("INTERNAL_API_KEY", "")},
        timeout=10.0,
    )
    resp.raise_for_status()
    data = resp.json()

    phone = data.get("guardian_phone_number")
    if phone:
        phone = phone.replace("-", "").replace(" ", "")
    if not phone:
        # 보호자가 위급상황 알림을 꺼둔 경우 — 정상적인 "문자 안 보냄"이라 에러가 아니다.
        log.info("guardian opted out of emergency alerts — skipping SMS: %s", alert["id"])
        return

    # SMS 실패는 raise하지 않는다 — 대시보드 알림은 이미 생성됐으므로, SMS만 재시도하겠다고
    # 이 함수 전체를 재시도하면 Spring 쪽 알림이 매번 중복 생성된다.
    if not sms.send_emergency_sms(phone, data["recipient_name"], alert["alert"]):
        log.warning("guardian emergency SMS failed (dashboard notification already filed): %s", alert["id"])


def load_processed() -> set[str]:
    if PROCESSED_FILE.exists():
        return set(PROCESSED_FILE.read_text().split())
    return set()


def mark_processed(alert_id: str) -> None:
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PROCESSED_FILE.open("a") as f:
        f.write(alert_id + "\n")


def pending_alerts() -> list[dict]:
    if not GUARDIAN_ALERTS_FILE.exists():
        return []
    processed = load_processed()
    alerts = []
    for line in GUARDIAN_ALERTS_FILE.read_text().splitlines():
        if not line.strip():
            continue
        alert = json.loads(line)
        if alert["id"] not in processed:
            alerts.append(alert)
    return alerts


def process_once() -> int:
    handled = 0
    for alert in pending_alerts():
        log.info("new guardian alert: %s (%s)", alert["id"], alert["alert"])
        try:
            push_guardian_alert(alert)
            mark_processed(alert["id"])   # ledger advances ONLY on success
            handled += 1
        except Exception:
            log.exception("failed on %s — will retry next poll", alert["id"])
    handled += process_call_results_once()
    return handled


# --------------------------------------------------------------------------
# call_results_outbox -> Spring POST /internal/call-results (X-API-KEY 인증)
# --------------------------------------------------------------------------


ASSESS_URL = os.environ.get("ASSESS_URL", "").strip()
ASSESS_TIMEOUT = float(os.environ.get("ASSESS_TIMEOUT", "20"))


def _placeholder_assessment(reason: str, measured_at: str) -> tuple[dict, list]:
    """채점 서비스를 못 쓸 때. 진짜 값이 아님을 risk_level/summary에 박아둔다.

    measured_at은 None을 쓰지 않는다 — Spring Assessment.measuredAt이 non-null이라
    None을 보내면 DB엔 NULL로 저장되고, 프론트는 DateTime.parse(null)에서 그대로
    죽어버린다("다시 시도" 에러 화면). 대신 통화 종료 시각(ended_at)을 쓴다.
    """
    log.warning("compute_assessment: %s — placeholder 반환", reason)
    return {
        "measured_at": measured_at,
        "depression_score": 0.5, "emotional_stability": 0.5, "health_risk": 0.5,
        "cognitive_score": 0.5, "overall_risk": 0.5, "risk_level": "OBSERVE",
        "summary": f"채점 미연동({reason}) — placeholder 값입니다.",
        "recommendation": "채점 서비스 연동 후 재확인 필요.",
    }, []


def compute_assessment(call_log_entries: list, logic_data: dict, emergencies: list, measured_at: str, item_judgments: dict | None = None) -> tuple[dict, list, list | None]:
    """채점 서비스(hyope-ai serve/)에 위임한다.

    ASSESS_URL이 없으면 예전처럼 placeholder를 반환한다. 채점 서비스 없이도
    통화 → Spring 전송 파이프라인은 그대로 돌아야 하기 때문이다. 이 경로엔
    척도 판정이 없어 incomplete_categories를 낼 수 없으므로 None을 돌려주고,
    호출부가 outbox entry(brain.py가 통화 흐름 기준으로 낸 값)로 대체하게 한다.

    여기는 통화가 끝난 뒤 worker가 폴링하며 도는 경로라 지연에 여유가 있다.
    실패하면 예외를 그대로 올려 push_call_result가 재시도하게 둔다 —
    placeholder를 Spring에 보내고 처리완료로 찍어버리면 그 통화의 진짜 점수는
    영영 못 만든다.
    """
    if not ASSESS_URL:
        assessment, adherence_records = _placeholder_assessment("ASSESS_URL 미설정", measured_at)
        return assessment, adherence_records, None

    resp = requests.post(
        ASSESS_URL,
        json={"call_log_entries": call_log_entries,
              "logic_data": logic_data,
              "emergencies": emergencies,
              "item_judgments": item_judgments or {}},
        timeout=ASSESS_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    assessment = data["assessment"]
    # serve/assess.py는 통화 시각을 모르니 measured_at을 항상 None으로 두고
    # "worker.py가 채운다"고 위임한다 — 그 계약을 여기서 이행한다.
    assessment["measured_at"] = measured_at
    # incomplete_categories는 척도 판정(logic_data) 기준이라 통화 흐름 기준인
    # entry 쪽 값보다 정확하다 — 얹혀 있으면 그대로 쓴다(빈 리스트도 유효한 값).
    return assessment, data.get("adherence_records", []), data.get("incomplete_categories", [])


def load_call_result_processed() -> set[str]:
    if CALL_RESULT_PROCESSED_FILE.exists():
        return set(CALL_RESULT_PROCESSED_FILE.read_text().split())
    return set()


def mark_call_result_processed(entry_id: str) -> None:
    CALL_RESULT_PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CALL_RESULT_PROCESSED_FILE.open("a") as f:
        f.write(entry_id + "\n")


def pending_call_results() -> list[dict]:
    if not CALL_RESULT_OUTBOX.exists():
        return []
    processed = load_call_result_processed()
    entries = []
    for line in CALL_RESULT_OUTBOX.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["id"] not in processed:
            entries.append(entry)
    return entries


def push_call_result(entry: dict) -> None:
    assessment, adherence_records, incomplete_categories = compute_assessment(
        entry["call_log_entries"], entry["logic_data"], entry["emergencies"],
        entry["call_log"]["ended_at"], entry.get("item_judgments", {}),
    )
    if incomplete_categories is None:
        # 채점 서비스 미연동(placeholder) 경로 — entry 쪽 값으로 대체.
        # 이 필드 없는 예전 outbox 항목과도 하위호환.
        incomplete_categories = entry.get("incomplete_categories", [])
    payload = {
        "recipient_id": entry["recipient_id"],
        "call_log": entry["call_log"],
        "call_log_entries": entry["call_log_entries"],
        "assessment": assessment,
        "adherence_records": adherence_records,
        "profile_updates": entry["profile_updates"],  # Spring 원본 스펙엔 없는 확장 필드 — 조율 필요
        "incomplete_categories": incomplete_categories,
    }
    resp = requests.post(
        f"{SPRING_BASE_URL}/internal/call-results",
        json=payload,
        headers={"X-API-KEY": os.environ.get("INTERNAL_API_KEY", "")},
        timeout=10.0,
    )
    resp.raise_for_status()


def process_call_results_once() -> int:
    handled = 0
    for entry in pending_call_results():
        log.info("new call result: %s (recipient_id=%s)", entry["id"], entry["recipient_id"])
        try:
            push_call_result(entry)
            mark_call_result_processed(entry["id"])   # ledger advances ONLY on success
            handled += 1
        except Exception:
            log.exception("failed to push call result %s — will retry next poll", entry["id"])
    return handled


def run_forever() -> None:
    # server.py가 백그라운드 스레드로 이걸 직접 호출한다(같은 프로세스 = 같은 파일시스템이라
    # runtime/ 아래 파일을 server.py와 그대로 공유) — CLI로 별도 실행할 때는 main()이 이걸 부른다.
    log.info("watching %s and %s (Ctrl-C to stop)", GUARDIAN_ALERTS_FILE, CALL_RESULT_OUTBOX)
    while True:
        process_once()
        time.sleep(POLL_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardian alert worker")
    parser.add_argument("--once", action="store_true", help="drain the backlog and exit")
    args = parser.parse_args()

    if args.once:
        n = process_once()
        log.info("done: %d item(s) processed", n)
        return

    run_forever()


if __name__ == "__main__":
    main()

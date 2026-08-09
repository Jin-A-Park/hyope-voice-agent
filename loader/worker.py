from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("worker")

ROOT = Path(__file__).resolve().parent.parent

GUARDIAN_ALERTS_FILE = ROOT / "alerts" / "inbox.jsonl"
PROCESSED_FILE = ROOT / "alerts" / "processed.txt"
OUTBOX = ROOT / "guardian_outbox"
POLL_SECONDS = 2

# server.py가 통화 종료 시 여기 한 줄씩 남긴다(GUARDIAN_ALERTS_FILE과 같은 내구성 패턴) —
# 실제 Spring 전송과 채점 모델 호출은 이 파일을 폴링하며 여기서 처리한다.
CALL_RESULT_OUTBOX = ROOT / "call_results_outbox.jsonl"
CALL_RESULT_PROCESSED_FILE = ROOT / "call_results_processed.txt"
SPRING_BASE_URL = os.environ.get("SPRING_BASE_URL", "http://localhost:8080")

GUARDIAN_EMAIL = os.environ.get("GUARDIAN_EMAIL", "guardian@example.com")

DRAFTER_PROMPT = (
    "당신은 어르신 안부전화 서비스에서, 통화가 위급 상황으로 조기 종료됐을 때 "
    "보호자에게 보낼 알림 이메일을 작성합니다. 알림 JSON(간단한 요약과 카테고리별 "
    "판단·이유·권고사항이 담긴 logic_data)을 받아서, 무슨 일이 있었는지, 어떤 항목이 "
    "왜 우려되는지, 무엇을 권장하는지를 차분하지만 진지한 톤으로 명확하게 설명하는 "
    "이메일 본문을 작성하세요. 평문만 사용하고, 제목 줄이나 자리표시자는 넣지 마세요. "
    "서명은 '어르신 안부전화 서비스 (자동발송)'로 하세요."
)


def draft_email(alert: dict) -> str:
    """LLM-as-function — alert JSON in, email body out.

    The worker gets its own model and its own prompt (Slide 22, "two brains"):
    the voice path converses; this one produces an artifact, with zero latency
    pressure — which is why WORKER_MODEL may be a smaller, cheaper model.
    """
    client = OpenAI(
        base_url=os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1"),
        api_key=os.environ["XAI_API_KEY"],
    )
    chat = client.chat.completions.create(
        model=os.environ.get("WORKER_MODEL") or os.environ.get("CHAT_MODEL", "grok-4"),
        messages=[
            {"role": "system", "content": DRAFTER_PROMPT},
            {"role": "user", "content": json.dumps(alert, ensure_ascii=False)},
        ],
    )
    body = chat.choices[0].message.content
    if not body:
        raise RuntimeError("empty draft from model")
    return body


def send_email(alert: dict, body: str) -> Path:
    OUTBOX.mkdir(exist_ok=True)
    subject = f"[긴급] 어르신 안부전화 조기종료 안내 — {alert['id']}"
    path = OUTBOX / f"{alert['id']}.eml"
    path.write_text(
        f"To: {GUARDIAN_EMAIL}\n"
        f"From: dispatch@elder-checkin.example\n"
        f"Subject: {subject}\n\n"
        f"{body}\n"
    )
    log.info("EMAIL SENT -> %s (%s)", path, subject)
    return path


def load_processed() -> set[str]:
    if PROCESSED_FILE.exists():
        return set(PROCESSED_FILE.read_text().split())
    return set()


def mark_processed(alert_id: str) -> None:
    PROCESSED_FILE.parent.mkdir(exist_ok=True)
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
            body = draft_email(alert)
            send_email(alert, body)
            mark_processed(alert["id"])   # ledger advances ONLY on success
            handled += 1
        except Exception:
            log.exception("failed on %s — will retry next poll", alert["id"])
    handled += process_call_results_once()
    return handled


# --------------------------------------------------------------------------
# call_results_outbox -> Spring POST /internal/call-results (X-API-KEY 인증)
# --------------------------------------------------------------------------


def compute_assessment(call_log_entries: list, logic_data: dict, emergencies: list) -> tuple[dict, list]:
    """실제 채점 모델 연동 전 스텁.

    별도 채점 모델이 call_log_entries/logic_data를 보고 depression_score 등 정규화된
    점수와 adherence_records를 내는 게 최종 그림인데, 그 모델은 이번 작업 범위 밖이라
    지금은 파이프라인(아웃박스 -> Spring 전송)을 끝까지 테스트할 수 있게 placeholder만
    반환한다. 진짜 값이 아니므로 risk_level/summary에 명확히 표시해서 실수로 진짜 값처럼
    쓰이지 않게 한다.
    """
    log.warning("compute_assessment: 채점 모델 미연동 — placeholder 값을 반환합니다")
    assessment = {
        "measured_at": None,
        "depression_score": 0.5,
        "emotional_stability": 0.5,
        "health_risk": 0.5,
        "cognitive_score": 0.5,
        "overall_risk": 0.5,
        "risk_level": "LOW",
        "summary": "채점 모델 미연동 — placeholder 값입니다.",
        "recommendation": "채점 모델 연동 후 재확인 필요.",
    }
    adherence_records: list = []
    return assessment, adherence_records


def load_call_result_processed() -> set[str]:
    if CALL_RESULT_PROCESSED_FILE.exists():
        return set(CALL_RESULT_PROCESSED_FILE.read_text().split())
    return set()


def mark_call_result_processed(entry_id: str) -> None:
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
    assessment, adherence_records = compute_assessment(
        entry["call_log_entries"], entry["logic_data"], entry["emergencies"],
    )
    payload = {
        "recipient_id": entry["recipient_id"],
        "call_log": entry["call_log"],
        "call_log_entries": entry["call_log_entries"],
        "assessment": assessment,
        "adherence_records": adherence_records,
        "profile_updates": entry["profile_updates"],  # Spring 원본 스펙엔 없는 확장 필드 — 조율 필요
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardian alert worker")
    parser.add_argument("--once", action="store_true", help="drain the backlog and exit")
    args = parser.parse_args()

    if args.once:
        n = process_once()
        log.info("done: %d item(s) processed", n)
        return

    log.info("watching %s and %s (Ctrl-C to stop)", GUARDIAN_ALERTS_FILE, CALL_RESULT_OUTBOX)
    while True:
        process_once()
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

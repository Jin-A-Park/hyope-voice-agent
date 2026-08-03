from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("worker")

GUARDIAN_ALERTS_FILE = Path("alerts/inbox.jsonl")
PROCESSED_FILE = Path("alerts/processed.txt")
OUTBOX = Path("guardian_outbox")
POLL_SECONDS = 2

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
    return handled


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardian alert worker")
    parser.add_argument("--once", action="store_true", help="drain the backlog and exit")
    args = parser.parse_args()

    if args.once:
        n = process_once()
        log.info("done: %d alert(s) processed", n)
        return

    log.info("watching %s (Ctrl-C to stop)", GUARDIAN_ALERTS_FILE)
    while True:
        process_once()
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

# 어르신께 도움처 연락처를 문자(SMS)로 전송 — ClawOps Messages API 사용.
# server.py의 통화 발신용 클라이언트(call_client())와는 별개로 이 모듈이 자체 싱글턴을
# 갖는다(integrations/dispatch.py -> server.py 순환 참조를 피하기 위해, geo.py가 카카오
# 클라이언트를 자체 소유하는 것과 같은 방식).
from __future__ import annotations

import logging
import os
from datetime import datetime

from clawops import AsyncClawOps, ClawOps

log = logging.getLogger("dispatcher")

_client: AsyncClawOps | None = None
_sync_client: ClawOps | None = None


def _sms_client() -> AsyncClawOps:
    global _client
    if _client is None:
        _client = AsyncClawOps(
            api_key=os.environ["CLAWOPS_API_KEY"],
            account_id=os.environ["CLAWOPS_ACCOUNT_ID"],
        )
    return _client


def _sms_client_sync() -> ClawOps:
    # integrations/worker.py는 asyncio 없이 도는 폴링 스크립트라 동기 클라이언트가 필요하다.
    global _sync_client
    if _sync_client is None:
        _sync_client = ClawOps(
            api_key=os.environ["CLAWOPS_API_KEY"],
            account_id=os.environ["CLAWOPS_ACCOUNT_ID"],
        )
    return _sync_client


def _resource_sms_body(resources: list[dict]) -> str:
    entries = "\n\n".join(f"{r['name']} {r['phone']}\n{r['address']}" for r in resources)
    return f"안녕하세요 어르신, 요청하신 정보 드릴게요.\n\n{entries}\n\n감사합니다."


async def send_resource_sms(to: str, resources: list[dict]) -> bool:
    """resources({"name","phone","address"} 목록)를 to로 문자 발송. 성공하면 True.

    발신 실패는 전부 여기서 흡수해 False만 돌려준다 — 호출부(integrations/dispatch.py)가 이것 때문에
    죽지 않고, agent/tools.py가 "직접 불러드리세요" 폴백 안내로 자연스럽게 넘어가게 설계돼 있다.
    """
    body = _resource_sms_body(resources)
    try:
        await _sms_client().messages.create(
            to=to, from_=os.environ["CLAWOPS_FROM_NUMBER"], body=body, type="sms",
        )
    except Exception:
        log.exception("send_resource_sms failed — caller will fall back to voice guidance")
        return False
    return True


def _call_summary_sms_body(recipient_name: str, summary_text: str) -> str:
    called_at = datetime.now().strftime("%m월 %d일 %H:%M")
    return (
        "[하이오피] 통화 요약\n"
        f"대상자: {recipient_name}님\n"
        f"통화 시각: {called_at}\n\n"
        f"{summary_text}"
    )


def send_call_summary_sms(to: str, recipient_name: str, summary_text: str) -> bool:
    """통화 종료 후(integrations/worker.py의 동기 폴링 루프에서) 보호자에게 통화 요약 문자 발송.

    성공하면 True. 발신 실패는 send_resource_sms와 동일하게 여기서 흡수해 False만 돌려준다 —
    통화 결과 자체는 이미 Spring에 성공적으로 전달됐으니, 요약 SMS 하나 실패했다고 재시도 루프를
    돌 이유는 없다(호출부가 로그만 남기고 넘어간다).
    """
    try:
        _sms_client_sync().messages.create(
            to=to, from_=os.environ["CLAWOPS_FROM_NUMBER"],
            body=_call_summary_sms_body(recipient_name, summary_text), type="sms",
        )
    except Exception:
        log.exception("send_call_summary_sms failed")
        return False
    return True

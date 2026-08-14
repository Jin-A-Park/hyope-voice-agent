# 어르신께 도움처 연락처를 문자(SMS)로 전송 — ClawOps Messages API 사용.
# server.py의 통화 발신용 클라이언트(call_client())와는 별개로 이 모듈이 자체 싱글턴을
# 갖는다(integrations/dispatch.py -> server.py 순환 참조를 피하기 위해, geo.py가 카카오
# 클라이언트를 자체 소유하는 것과 같은 방식).
from __future__ import annotations

import logging
import os

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


async def send_resource_sms(to: str, resources: list[dict]) -> bool:
    """resources({"name","phone","address"} 목록)를 to로 문자 발송. 성공하면 True.

    발신 실패는 전부 여기서 흡수해 False만 돌려준다 — 호출부(integrations/dispatch.py)가 이것 때문에
    죽지 않고, agent/tools.py가 "직접 불러드리세요" 폴백 안내로 자연스럽게 넘어가게 설계돼 있다.
    """
    body = "\n".join(f"{r['name']} {r['phone']}" for r in resources)
    try:
        await _sms_client().messages.create(
            to=to, from_=os.environ["CLAWOPS_FROM_NUMBER"], body=body, type="sms",
        )
    except Exception:
        log.exception("send_resource_sms failed — caller will fall back to voice guidance")
        return False
    return True


def send_emergency_sms(to: str, recipient_name: str, signal: str) -> bool:
    """위급 상황 감지 시 보호자에게 즉시 문자 발송. 성공하면 True.

    integrations/worker.py(동기 폴링 스크립트)에서 호출되므로 동기 클라이언트를 쓴다.
    발신 실패는 send_resource_sms와 동일하게 여기서 흡수해 False만 돌려준다 — 대시보드
    알림(Spring 쪽)은 이미 남았으니, SMS 하나가 일시적으로 실패했다고 worker.py가
    전체 알림 처리를 재시도하며 대시보드 알림을 중복 생성하게 만들 필요는 없다.
    """
    body = f"[하이오피] {recipient_name}님에게 위급 상황이 감지되었습니다: {signal}. 앱에서 확인해주세요."
    try:
        _sms_client_sync().messages.create(
            to=to, from_=os.environ["CLAWOPS_FROM_NUMBER"], body=body, type="sms",
        )
    except Exception:
        log.exception("send_emergency_sms failed — dashboard notification already filed, continuing")
        return False
    return True

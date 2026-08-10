# 어르신께 도움처 연락처를 문자(SMS)로 전송 — ClawOps Messages API 사용.
# server.py의 통화 발신용 클라이언트(call_client())와는 별개로 이 모듈이 자체 싱글턴을
# 갖는다(integrations/dispatch.py -> server.py 순환 참조를 피하기 위해, geo.py가 카카오
# 클라이언트를 자체 소유하는 것과 같은 방식).
from __future__ import annotations

import logging
import os

from clawops import AsyncClawOps

log = logging.getLogger("dispatcher")

_client: AsyncClawOps | None = None


def _sms_client() -> AsyncClawOps:
    global _client
    if _client is None:
        _client = AsyncClawOps(
            api_key=os.environ["CLAWOPS_API_KEY"],
            account_id=os.environ["CLAWOPS_ACCOUNT_ID"],
        )
    return _client


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

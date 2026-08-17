# 네트워킹이 필요한 처리를 agent.tools.run_tool() 앞뒤로 가로채는 공용 허브. server.py/
# gemini_loader.py는 둘 다 agent.tools.run_tool을 직접 부르지 않고 이 dispatch_tool을 거쳐야 한다 —
# agent/ 쪽은 네트워킹 금지 원칙이라, 실제 API 호출(지오코딩·SMS·Spring)은 여기(네트워킹 계층)에서
# 해결한다.
#
# run_tool() 호출 *전*: check_external_api_necessity(keyword가 있으면 카카오 지오코딩)와
# send_resource_info(ClawOps SMS 발송)를 가로채 결과값을 args에 실어 넘긴다.
# run_tool() 호출 *후*: agent.tools.check_in이 EMERGENCY를 확정하면서 쌓아둔
# state["_pending_guardian_alerts"]를 비운다 — Spring 대시보드 알림 + 보호자 SMS를 그 자리에서
# 즉시 시도하고(hyope-call-agent의 flag_emergency 즉시-통보 패턴을 그대로 따름), 실패했을 때만
# 파일에 폴백 기록해 integrations/worker.py가 나중에 재시도하게 한다.
from __future__ import annotations

import json
import logging
import os

import httpx

from agent import brain, tools as agent_tools
from integrations import geo, sms

log = logging.getLogger("dispatcher")

SPRING_BASE_URL = os.environ.get("SPRING_BASE_URL", "http://localhost:8080")


async def _send_emergency_alert_now(state: dict, alert: dict) -> bool:
    """위급 확정을 감지 즉시 Spring(대시보드 알림)과 보호자 SMS로 통보한다.

    성공하면 True. Spring 응답이 없거나 오류면 False를 돌려주고, 호출부(dispatch_tool)가
    GUARDIAN_ALERTS_FILE에 폴백 기록을 남겨 integrations/worker.py가 나중에 재시도하게 한다
    (파일 큐는 이제 실패 시 안전망 용도로만 쓰인다).

    SMS 발송 실패는 여기서 흡수한다 — Spring POST(대시보드 알림)는 이미 성공했으므로, 그것까지
    실패로 치면 호출부가 파일에 다시 기록해 대시보드 알림이 중복 생성된다.
    """
    recipient_id = (state.get("_metadata") or {}).get("recipient_id")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{SPRING_BASE_URL}/internal/emergency-alerts",
                json={
                    "recipient_id": recipient_id,
                    "signal": alert["alert"],
                    "severity": "high",
                    "logic_data": alert["logic_data"],
                    "filed_at": alert["filed_at"],
                },
                headers={"X-API-KEY": os.environ.get("INTERNAL_API_KEY", "")},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        log.exception("emergency alert POST to Spring failed — falling back to file queue")
        return False

    phone = data.get("guardian_phone_number")
    if phone:
        phone = phone.replace("-", "").replace(" ", "")
    if not phone:
        # 보호자가 위급상황 알림을 꺼뒀으면 Spring이 이 필드를 비워서 응답한다 — 정상적인
        # "문자 안 보냄"이지 실패가 아니다.
        log.info("guardian opted out of emergency alerts — skipping SMS")
        return True

    recipient_name = data.get("recipient_name") or "대상자"
    if not await sms.send_emergency_sms_async(phone, recipient_name, alert["alert"]):
        log.warning("guardian emergency SMS failed (dashboard notification already filed)")
    return True


def _file_fallback_guardian_alert(state: dict, alert: dict) -> None:
    brain.GUARDIAN_ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(brain.GUARDIAN_ALERTS_FILE, "a") as f:
        f.write(json.dumps({
            "id": brain.new_alert_id(),
            "alert": alert["alert"],
            "logic_data": alert["logic_data"],
            "metadata": state.get("_metadata") or {},
            "filed_at": alert["filed_at"],
        }, ensure_ascii=False) + "\n")


async def dispatch_tool(state: dict, name: str, args: dict) -> str:
    if name == "check_external_api_necessity":
        keyword = args.get("keyword")
        if keyword:
            address = ((state.get("_metadata") or {}).get("profile") or {}).get("address")
            if address:
                nearby = await geo.resolve_nearby_resource(address, keyword=keyword)
                if nearby:
                    args = {**args, "nearby_resource": nearby}

    elif name == "send_resource_info":
        resource = agent_tools.find_offered_resource(state, args.get("resource_name", ""))
        if resource is not None:
            phone = (state.get("_metadata") or {}).get("phone_number")
            sent = await sms.send_resource_sms(phone, [resource]) if phone else False
            args = {**args, "sms_sent": sent}

    result = agent_tools.run_tool(state, name, args)

    for alert in state.pop("_pending_guardian_alerts", []):
        ok = await _send_emergency_alert_now(state, alert)
        if not ok:
            _file_fallback_guardian_alert(state, alert)

    return result

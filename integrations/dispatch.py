# 네트워킹이 필요한 툴 호출을 agent.tools.run_tool() 앞에서 가로채는 공용 허브. server.py/
# gemini_loader.py는 둘 다 agent.tools.run_tool을 직접 부르지 않고 이 dispatch_tool을 거쳐야 한다 —
# agent/ 쪽은 네트워킹 금지 원칙이라, 실제 API 호출(지오코딩·SMS·Spring)은 여기(네트워킹 계층)에서
# 미리 해결해서 결과값만 agent.tools에 넘긴다.
from __future__ import annotations

import logging
import os
import time

import httpx

from agent import tools as agent_tools
from integrations import geo, sms

log = logging.getLogger("dispatcher")

SPRING_BASE_URL = os.environ.get("SPRING_BASE_URL", "http://localhost:8080")


async def _send_emergency_alert_now(state: dict, signal: str) -> bool:
    """위급 상황을 감지 즉시 Spring(대시보드 알림)과 보호자 SMS로 통보한다.

    성공하면 True. Spring 응답이 없거나 오류면 False를 돌려주고, 호출부인
    agent.tools.flag_emergency가 GUARDIAN_ALERTS_FILE에 폴백 기록을 남겨
    integrations/worker.py가 나중에 재시도하게 한다(파일 큐는 이제 실패 시 안전망 용도로만 남음).

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
                    "signal": signal,
                    "severity": "high",
                    "logic_data": state["logic_data"],
                    "filed_at": time.time(),
                },
                headers={"X-API-KEY": os.environ.get("INTERNAL_API_KEY", "")},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        log.exception("emergency alert POST to Spring failed — falling back to file queue")
        return False

    phone = data.get("guardian_phone_number")
    if not phone:
        # 보호자가 위급상황 알림을 꺼뒀으면 Spring이 이 필드를 비워서 응답한다 — 정상적인
        # "문자 안 보냄"이지 실패가 아니다. None으로 그대로 보내면 ClawOps가 거부해서
        # 진짜 실패처럼 로그가 남는다.
        log.info("guardian opted out of emergency alerts — skipping SMS")
        return True

    if not await sms.send_emergency_sms_async(phone, data["recipient_name"], signal):
        log.warning("guardian emergency SMS failed (dashboard notification already filed)")
    return True


async def dispatch_tool(state: dict, name: str, args: dict) -> str:
    """flag_emergency(severity=="high")는 Spring 대시보드 알림 + 보호자 SMS를 즉시 시도해서
    결과(alert_dispatched)를 args에 실어 넘긴다. flag_emergency(severity!=high, "default" 카테고리),
    log_answer_analysis(큐레이션된 질문에서 "우려" 판정), search_nearby_resource(어르신이 직접 물어봄)는
    카카오 지오코딩으로 nearby_resource를, send_resource_info는 ClawOps SMS 발송 결과(sms_sent)를
    args에 실어 agent.tools.run_tool에 넘긴다. 나머지 툴 호출은 그대로 통과."""

    if name == "flag_emergency" and args.get("severity") == "high":
        ok = await _send_emergency_alert_now(state, args.get("signal", ""))
        args = {**args, "alert_dispatched": ok}

    elif name == "flag_emergency" and args.get("severity") != "high":
        signal = args.get("signal", "")
        if agent_tools.emergency_category(signal) == "default":
            address = ((state.get("_metadata") or {}).get("profile") or {}).get("address")
            if address:
                keyword = agent_tools.default_category_keyword(signal)
                nearby = await geo.resolve_nearby_resource(address, keyword=keyword)
                if nearby:
                    args = {**args, "nearby_resource": nearby}

    elif name == "log_answer_analysis":
        keyword = agent_tools.should_offer_resource(
            state, args.get("category", ""), args.get("question_id", ""), args.get("assessment", ""),
        )
        if keyword:
            address = ((state.get("_metadata") or {}).get("profile") or {}).get("address")
            if address:
                nearby = await geo.resolve_nearby_resource(address, keyword=keyword)
                if nearby:
                    args = {**args, "nearby_resource": nearby}

    elif name == "search_nearby_resource":
        address = ((state.get("_metadata") or {}).get("profile") or {}).get("address")
        if address:
            nearby = await geo.resolve_nearby_resource(address, keyword=args.get("search_keyword", ""))
            if nearby:
                args = {**args, "nearby_resource": nearby}

    elif name == "send_resource_info":
        resources = agent_tools.find_offered_resources(state, args.get("resource_names") or [])
        if resources:
            phone = (state.get("_metadata") or {}).get("phone_number")
            sent = await sms.send_resource_sms(phone, resources) if phone else False
            args = {**args, "sms_sent": sent}

    return agent_tools.run_tool(state, name, args)

# 네트워킹이 필요한 처리를 agent.tools.run_tool() 앞뒤로 가로채는 공용 허브. server.py/
# gemini_loader.py는 둘 다 agent.tools.run_tool을 직접 부르지 않고 이 dispatch_tool을 거쳐야 한다 —
# agent/ 쪽은 네트워킹 금지 원칙이라, 실제 API 호출(지오코딩·SMS·Spring)은 여기(네트워킹 계층)에서
# 해결한다.
#
# run_tool() 호출 전: check_external_api_necessity(keyword가 있으면 카카오 지오코딩)와
# send_resource_info(ClawOps SMS 발송)를 가로채 결과값을 args에 실어 넘긴다.
from __future__ import annotations

import logging

from agent import tools as agent_tools
from integrations import geo, sms

log = logging.getLogger("dispatcher")


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

    return agent_tools.run_tool(state, name, args)

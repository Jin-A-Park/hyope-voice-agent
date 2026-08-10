# 네트워킹이 필요한 툴 호출을 agent.tools.run_tool() 앞에서 가로채는 공용 허브. server.py/
# gemini_loader.py는 둘 다 agent.tools.run_tool을 직접 부르지 않고 이 dispatch_tool을 거쳐야 한다 —
# agent/ 쪽은 네트워킹 금지 원칙이라, 실제 API 호출(지오코딩·SMS)은 여기(네트워킹 계층)에서
# 미리 해결해서 결과값만 agent.tools에 넘긴다.
from __future__ import annotations

from agent import tools as agent_tools
from integrations import geo, sms


async def dispatch_tool(state: dict, name: str, args: dict) -> str:
    """flag_emergency(severity!=high, "default" 카테고리), log_answer_analysis(큐레이션된 질문에서
    "우려" 판정), search_nearby_resource(어르신이 직접 물어봄)는 카카오 지오코딩으로 nearby_resource를,
    send_resource_info는 ClawOps SMS 발송 결과(sms_sent)를 args에 실어 agent.tools.run_tool에 넘긴다.
    나머지 툴 호출은 그대로 통과."""

    if name == "flag_emergency" and args.get("severity") != "high":
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
        resource = agent_tools.find_offered_resource(state, args.get("resource_name", ""))
        if resource is not None:
            phone = (state.get("_metadata") or {}).get("phone_number")
            sent = await sms.send_resource_sms(phone, [resource]) if phone else False
            args = {**args, "sms_sent": sent}

    return agent_tools.run_tool(state, name, args)

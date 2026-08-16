# 주소 -> 좌표 -> 주변 시설 검색(카카오 로컬 API). integrations/dispatch.py의 dispatch_tool()이
# check_external_api_necessity 호출을 가로채서, keyword가 있을 때만 이 모듈을 불러 조회한다
# — agent/ 쪽은 네트워킹 금지 원칙이라 여기(네트워킹 계층)에서 미리 해결해서 값만 넘긴다.
from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger("dispatcher")

_ADDRESS_SEARCH_URL = "https://dapi.kakao.com/v2/local/search/address.json"
_KEYWORD_SEARCH_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


async def resolve_nearby_resource(
    address: str, keyword: str = "노인복지관", radius: int = 5000,
) -> list[dict] | None:
    """주소 근처 keyword 장소 최대 3곳을 거리순으로 반환. KAKAO_API_KEY가 없거나

    지오코딩·검색 중 뭐가 됐든 실패하면 None — 호출부(integrations/dispatch.py의 dispatch_tool)가
    nearby_resource를 안 실어서 agent/tools.py의 _help_resource_for가 static/help_resources.json
    정적 폴백으로 자연스럽게 넘어가게 설계돼 있다.
    """
    if not address:
        return None
    api_key = os.environ.get("KAKAO_API_KEY")
    if not api_key:
        log.warning("KAKAO_API_KEY not set — nearby_resource lookup skipped, static fallback will be used")
        return None

    headers = {"Authorization": f"KakaoAK {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            geo_resp = await client.get(_ADDRESS_SEARCH_URL, params={"query": address}, headers=headers)
            geo_resp.raise_for_status()
            docs = geo_resp.json().get("documents") or []
            if not docs:
                log.warning("no geocoding match for address (nearby_resource skipped)")
                return None
            x, y = docs[0]["x"], docs[0]["y"]

            search_resp = await client.get(_KEYWORD_SEARCH_URL, params={
                "query": keyword, "x": x, "y": y, "radius": radius, "sort": "distance",
            }, headers=headers)
            search_resp.raise_for_status()
            places = (search_resp.json().get("documents") or [])[:3]
    except Exception:
        log.exception("nearby_resource lookup failed — static fallback will be used")
        return None

    if not places:
        return None
    return [
        {
            "name": p["place_name"],
            "phone": p.get("phone") or "번호 미등록",
            "address": p.get("road_address_name") or p.get("address_name") or "",
        }
        for p in places
    ]

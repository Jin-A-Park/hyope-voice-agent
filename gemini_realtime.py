"""claw-ops 전화/브라우저 오디오 <-> Gemini Live API(BidiGenerateContent) 브릿지.

xAI Realtime과 프로토콜이 통째로 다르다(세션 설정 메시지, 오디오 전송/수신 이벤트,
tool 호출·응답 이벤트 모양이 전부 다름) — server.py의 xAI 경로에 분기를 섞으면
지저분해지니 여기 따로 둔다. model이 MODEL_PREFIX로 시작하면 claw_stream_ws는
run_session(), browser_ws는 run_browser_session()으로 위임한다.

테스트 전용 브릿지라 문서 기준으로만 짰고, 아래 몇 군데는 실제 xAI 쪽처럼 라이브로
검증 안 됐다 — 처음 써볼 때 제일 먼저 의심할 지점:
  - _build_setup_message의 voiceName / automaticActivityDetection 필드명·값
  - 통화 시작 시 인사를 먼저 하게 만드는 초기 클라이언트 턴(bootstrap greeting) 방식
"""
from __future__ import annotations

import asyncio
import audioop
import base64
import contextlib
import json
import logging
import os

import websockets
from fastapi import WebSocket, WebSocketDisconnect

import brain

log = logging.getLogger("dispatcher")

MODEL_PREFIX = "gemini-"


def is_gemini_model(model: str | None) -> bool:
    return bool(model) and model.startswith(MODEL_PREFIX)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. GEMINI_API_KEY를 .env에 채워라.")
    return value


def _realtime_url(model: str) -> str:
    base = os.environ.get(
        "GEMINI_REALTIME_BASE_URL",
        "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
    )
    return f"{base}?key={_require_env('GEMINI_API_KEY')}"


def _to_function_declarations(xai_tools: list[dict]) -> list[dict]:
    """brain.build_tools()의 xAI(OpenAI 스타일) 함수 스키마에서 "type": "function" 래퍼만
    벗겨 Gemini의 functionDeclarations 모양으로. parameters(JSON Schema) 내부는 그대로 —
    둘 다 소문자 "object"/"string" 타입을 쓴다."""
    return [
        {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
        for t in xai_tools
    ]


def _build_setup_message(question_bank: dict, model: str) -> dict:
    return {
        "setup": {
            "model": f"models/{model}",
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": os.environ.get("GEMINI_TTS_VOICE", "Kore"),
                        },
                    },
                },
            },
            "systemInstruction": {
                "parts": [{"text": brain.build_full_instructions(question_bank)}],
            },
            "tools": [
                {"functionDeclarations": _to_function_declarations(brain.build_tools(question_bank))},
            ],
            "realtimeInputConfig": {
                "automaticActivityDetection": {
                    "prefixPaddingMs": int(os.environ.get("PREFIX_MS", "300")),
                    "silenceDurationMs": int(os.environ.get("SILENCE_MS", "900")),
                },
            },
            "outputAudioTranscription": {},
            "inputAudioTranscription": {},
        },
    }


# --- 오디오 트랜스코딩 ---
# claw-ops(G.711 μ-law, 8kHz) <-> Gemini Live(PCM16, 입력 16kHz/출력 24kHz).
# server.py에 이미 24kHz용 버전이 있지만, gemini_realtime.py가 server.py를 참조하면
# 순환 import가 생겨서(server.py도 이 파일을 import) 여기 따로 둔다 — 몇 줄 안 되는 stdlib
# 호출이라 중복 비용보다 파일 분리 이득이 크다고 판단.

def _claw_media_to_pcm16k(b64_ulaw: str, state) -> tuple[str, object]:
    ulaw = base64.b64decode(b64_ulaw)
    pcm8k = audioop.ulaw2lin(ulaw, 2)
    pcm16k, state = audioop.ratecv(pcm8k, 2, 1, 8000, 16000, state)
    return base64.b64encode(pcm16k).decode("ascii"), state


def _pcm24k_to_claw_media(b64_pcm16_24k: str, state) -> tuple[str, object]:
    leftover, resample_state = state if state else (b"", None)
    pcm24k = leftover + base64.b64decode(b64_pcm16_24k)
    if len(pcm24k) % 2:
        pcm24k, leftover = pcm24k[:-1], pcm24k[-1:]
    else:
        leftover = b""
    pcm8k, resample_state = audioop.ratecv(pcm24k, 2, 1, 24000, 8000, resample_state)
    ulaw = audioop.lin2ulaw(pcm8k, 2)
    return base64.b64encode(ulaw).decode("ascii"), (leftover, resample_state)


# 브라우저(/ws/browser): claw-ops처럼 μ-law가 아니라 이미 PCM16이라 리샘플만 하면 된다.
# 입력(브라우저->Gemini) 24kHz->16kHz만 필요 — 출력은 Gemini도 24kHz라 그대로 통과시킨다.

def _pcm24k_to_pcm16k(b64_pcm16_24k: str, state) -> tuple[str, object]:
    leftover, resample_state = state if state else (b"", None)
    pcm24k = leftover + base64.b64decode(b64_pcm16_24k)
    if len(pcm24k) % 2:
        pcm24k, leftover = pcm24k[:-1], pcm24k[-1:]
    else:
        leftover = b""
    pcm16k, resample_state = audioop.ratecv(pcm24k, 2, 1, 24000, 16000, resample_state)
    return base64.b64encode(pcm16k).decode("ascii"), (leftover, resample_state)


async def _pump_upstream(upstream, client_ws: WebSocket, state: dict, session_id: str) -> bool:
    """Gemini -> claw-ops. 오디오는 μ-law 8kHz media 이벤트로 변환해 내보내고, toolCall이
    오면 brain.run_tool을 실행해 결과를 돌려준다. xAI와 달리 tool 결과를 보낸 뒤 별도
    재요청(response.create 격) 없이 같은 수신 루프에서 이어지는 응답을 그대로 받는다."""
    resample_state = None
    ending = False
    try:
        async for raw in upstream:
            event = json.loads(raw)

            server_content = event.get("serverContent")
            if server_content:
                model_turn = server_content.get("modelTurn") or {}
                for part in model_turn.get("parts", []):
                    inline = part.get("inlineData")
                    if inline and inline.get("data"):
                        b64_ulaw, resample_state = _pcm24k_to_claw_media(inline["data"], resample_state)
                        await client_ws.send_json({"event": "media", "media": {"payload": b64_ulaw}})
                if server_content.get("outputTranscription", {}).get("text"):
                    log.info("[%s] gemini: %s", session_id, server_content["outputTranscription"]["text"])
                if server_content.get("inputTranscription", {}).get("text"):
                    log.info("[%s] 어르신: %s", session_id, server_content["inputTranscription"]["text"])
                if ending and server_content.get("turnComplete"):
                    return True

            tool_call = event.get("toolCall")
            if tool_call:
                function_responses = []
                for fc in tool_call["functionCalls"]:
                    name = fc["name"]
                    result = brain.run_tool(state, name, fc.get("args") or {})
                    function_responses.append({
                        "id": fc["id"], "name": name, "response": json.loads(result),
                    })
                    log.info("[%s] gemini tool call: %s(%s)", session_id, name, fc.get("args"))
                    if name == "end_call":
                        ending = True
                await upstream.send(json.dumps({
                    "toolResponse": {"functionResponses": function_responses},
                }))
    except (WebSocketDisconnect, asyncio.CancelledError):
        raise
    except Exception:
        log.exception("[%s] gemini upstream pump failed", session_id)
        return False
    return True


async def _pump_downstream_claw(client_ws: WebSocket, upstream) -> None:
    """claw-ops -> Gemini. media는 8kHz μ-law -> 16kHz PCM16으로 변환해 realtimeInput으로."""
    resample_state = None
    while True:
        event = await client_ws.receive_json()
        etype = event.get("event")

        if etype == "media":
            b64_pcm16, resample_state = _claw_media_to_pcm16k(event["media"]["payload"], resample_state)
            await upstream.send(json.dumps({
                "realtimeInput": {"audio": {"data": b64_pcm16, "mimeType": "audio/pcm;rate=16000"}},
            }))
        elif etype == "dtmf":
            log.info("claw dtmf: %s", event.get("dtmf", {}).get("digit"))
        elif etype == "stop":
            return


@contextlib.asynccontextmanager
async def _open_session(model: str, question_bank: dict):
    """연결 + setup + 초기 인사 턴까지 끝낸 upstream을 내준다. claw/browser 양쪽 경로가 공유."""
    async with websockets.connect(_realtime_url(model)) as upstream:
        await upstream.send(json.dumps(_build_setup_message(question_bank, model)))
        await upstream.recv()  # BidiGenerateContentSetupComplete — 이후부터 오디오 전송 안전

        # xAI 쪽 GREETING_PROMPT과 같은 역할: 어르신이 먼저 말하기 전에 인사부터 하게
        # 만드는 초기 턴. Gemini는 시스템 role의 1회성 메시지 개념이 없어서 user 턴으로
        # 흉내내는 것 — 실제로 이렇게 트리거되는지는 라이브로 확인 필요(모듈 docstring 참고).
        await upstream.send(json.dumps({
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": brain.GREETING_PROMPT}]}],
                "turnComplete": True,
            },
        }))
        yield upstream


async def run_session(client_ws: WebSocket, state: dict, question_bank: dict, model: str, session_id: str) -> bool:
    """claw_stream_ws가 xAI 대신 이걸 호출한다(model이 MODEL_PREFIX로 시작할 때만).

    반환값은 claw_stream_ws가 call_result 상태(COMPLETED/FAILED)에 그대로 반영한다.
    """
    async with _open_session(model, question_bank) as upstream:
        upstream_task = asyncio.create_task(_pump_upstream(upstream, client_ws, state, session_id))
        downstream_task = asyncio.create_task(_pump_downstream_claw(client_ws, upstream))

        done, pending = await asyncio.wait(
            {upstream_task, downstream_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        ok = True
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                log.exception("[%s] gemini session pump failed", session_id, exc_info=exc)
                ok = False
            elif task is upstream_task and task.result() is False:
                ok = False
        return ok


async def _pump_upstream_browser(upstream, client_ws: WebSocket, state: dict, session_id: str) -> None:
    """Gemini -> 브라우저. index.html이 기대하는 xAI 경로와 같은 이벤트 모양(type: audio/
    transcript/tool_used/turn_done/call_ended)으로 맞춰 보낸다 — 프론트는 손 안 대도 되게."""
    ending = False
    agent_text_parts: list[str] = []
    async for raw in upstream:
        event = json.loads(raw)

        server_content = event.get("serverContent")
        if server_content:
            model_turn = server_content.get("modelTurn") or {}
            for part in model_turn.get("parts", []):
                inline = part.get("inlineData")
                if inline and inline.get("data"):  # 이미 24kHz PCM16 — claw 경로와 달리 변환 불필요
                    await client_ws.send_json({"type": "audio", "audio": inline["data"]})
            out_text = (server_content.get("outputTranscription") or {}).get("text")
            if out_text:
                agent_text_parts.append(out_text)
                await client_ws.send_json({"type": "transcript", "role": "agent", "delta": out_text})
            in_text = (server_content.get("inputTranscription") or {}).get("text")
            if in_text:
                await client_ws.send_json({"type": "transcript", "role": "elder", "text": in_text})
            if server_content.get("turnComplete"):
                if agent_text_parts:
                    log.info("[%s] gemini: %s", session_id, "".join(agent_text_parts))
                    agent_text_parts.clear()
                await client_ws.send_json({"type": "turn_done"})
                if ending:
                    await client_ws.send_json({"type": "call_ended"})
                    return

        tool_call = event.get("toolCall")
        if tool_call:
            function_responses = []
            for fc in tool_call["functionCalls"]:
                name = fc["name"]
                result = brain.run_tool(state, name, fc.get("args") or {})
                function_responses.append({"id": fc["id"], "name": name, "response": json.loads(result)})
                log.info("[%s] gemini tool call: %s(%s)", session_id, name, fc.get("args"))
                await client_ws.send_json({"type": "tool_used", "name": name})
                if name == "end_call":
                    ending = True
            await upstream.send(json.dumps({
                "toolResponse": {"functionResponses": function_responses},
            }))


async def _pump_downstream_browser(client_ws: WebSocket, upstream) -> None:
    """브라우저 -> Gemini. PCM16 24kHz(이미 raw PCM, μ-law 아님) -> 16kHz로 리샘플만 해서 전달."""
    resample_state = None
    while True:
        msg = await client_ws.receive_json()
        kind = msg.get("type")

        if kind == "audio":
            b64_pcm16_16k, resample_state = _pcm24k_to_pcm16k(msg["audio"], resample_state)
            await upstream.send(json.dumps({
                "realtimeInput": {"audio": {"data": b64_pcm16_16k, "mimeType": "audio/pcm;rate=16000"}},
            }))
        elif kind == "hangup":
            return


async def run_browser_session(client_ws: WebSocket, state: dict, question_bank: dict, model: str, session_id: str) -> None:
    """browser_ws가 xAI 대신 이걸 호출한다(model이 MODEL_PREFIX로 시작할 때만)."""
    async with _open_session(model, question_bank) as upstream:
        await client_ws.send_json({"type": "ready"})

        upstream_task = asyncio.create_task(_pump_upstream_browser(upstream, client_ws, state, session_id))
        downstream_task = asyncio.create_task(_pump_downstream_browser(client_ws, upstream))

        done, pending = await asyncio.wait(
            {upstream_task, downstream_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                log.exception("[%s] gemini browser session pump failed", session_id, exc_info=exc)

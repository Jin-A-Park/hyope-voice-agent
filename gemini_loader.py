
# * 전화/브라우저 오디오 <-> Gemini Live API(BidiGenerateContent) 브릿지.

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os

import websockets
from fastapi import WebSocket, WebSocketDisconnect

from agent import brain, tools as agent_tools
from integrations import dispatch
from sinks import OutputSink, BrowserSink, CallSink, call_media_to_pcm16k, pcm24k_to_pcm16k





log = logging.getLogger("dispatcher")

MODEL_PREFIX = "gemini-"

def is_gemini_model(model: str | None) -> bool:
    return bool(model) and model.startswith(MODEL_PREFIX)

def _require_env(name: str) -> str:
    # * 환경변수 받아오기

    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. GEMINI_API_KEY를 .env에 채워라.")
    return value

def _realtime_url(model: str) -> str:
    # * 환경변수로 고정 x, 모델 선택

    base = os.environ.get(
        "GEMINI_REALTIME_BASE_URL",
        "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
    )
    return f"{base}?key={_require_env('GEMINI_API_KEY')}"

def _to_function_declarations(xai_tools: list[dict]) -> list[dict]:
    # * xAI 함수 스키마에서 "type": "function" 래퍼만 벗겨 Gemini의 functionDeclarations 모양으로.
    
    return [
        {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
        for t in xai_tools
    ]

def _build_setup_message(question_bank: dict, model: str) -> dict:
    # * xAI의 session.update에 대응하는 Gemini Live의 setup 메시지 — 모델/음성/시스템 프롬프트/툴/VAD를 한 번에 설정.
    
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
                {"functionDeclarations": _to_function_declarations(agent_tools.build_tools(question_bank))},
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

async def _pump_downstream_call(client_ws: WebSocket, upstream) -> None:
    # * phone call -> Gemini. media는 8kHz μ-law -> 16kHz PCM16으로 변환해 realtimeInput으로.
    
    resample_state = None
    while True:
        event = await client_ws.receive_json()
        etype = event.get("event")

        if etype == "media":
            b64_pcm16, resample_state = call_media_to_pcm16k(event["media"]["payload"], resample_state)
            await upstream.send(json.dumps({
                "realtimeInput": {"audio": {"data": b64_pcm16, "mimeType": "audio/pcm;rate=16000"}},
            }))
        elif etype == "dtmf":
            log.info("phone call dtmf: %s", event.get("dtmf", {}).get("digit"))
        elif etype == "stop":
            return

async def _pump_downstream_browser(client_ws: WebSocket, upstream) -> None:
    # * 브라우저 -> Gemini. PCM16 24kHz(이미 raw PCM, μ-law 아님) -> 16kHz로 리샘플만 해서 전달.
    
    resample_state = None
    while True:
        msg = await client_ws.receive_json()
        kind = msg.get("type")

        if kind == "audio":
            b64_pcm16_16k, resample_state = pcm24k_to_pcm16k(msg["audio"], resample_state)
            await upstream.send(json.dumps({
                "realtimeInput": {"audio": {"data": b64_pcm16_16k, "mimeType": "audio/pcm;rate=16000"}},
            }))
        elif kind == "hangup":
            return

async def _pump_upstream(upstream, sink: OutputSink, state: dict, session_id: str) -> bool:
    # * Gemini -> sink(브라우저 or 전화망). xAI와 달리 tool 결과 후 별도 재요청 없이 같은 수신 루프에서 이어지는 응답을 받는다.

    ending = False
    agent_text_parts: list[str] = []
    try:
        async for raw in upstream:
            event = json.loads(raw)

            server_content = event.get("serverContent")
            if server_content:
                model_turn = server_content.get("modelTurn") or {}
                for part in model_turn.get("parts", []):
                    inline = part.get("inlineData")
                    if inline and inline.get("data"):
                        await sink.send_audio(inline["data"])
                out_text = (server_content.get("outputTranscription") or {}).get("text")
                if out_text:
                    agent_text_parts.append(out_text)
                    await sink.send_event({"type": "transcript", "role": "agent", "delta": out_text})
                in_text = (server_content.get("inputTranscription") or {}).get("text")
                if in_text:
                    log.info("[%s] 어르신: %s", session_id, in_text)
                    await sink.send_event({"type": "transcript", "role": "elder", "text": in_text})
                if server_content.get("turnComplete"):
                    if agent_text_parts:
                        log.info("[%s] gemini: %s", session_id, "".join(agent_text_parts))
                        agent_text_parts.clear()
                    await sink.send_event({"type": "turn_done"})
                    if ending:
                        await sink.send_event({"type": "call_ended"})
                        return True

            tool_call = event.get("toolCall")
            if tool_call:
                function_responses = []
                for fc in tool_call["functionCalls"]:
                    name = fc["name"]
                    result = await dispatch.dispatch_tool(state, name, fc.get("args") or {})
                    function_responses.append({
                        "id": fc["id"], "name": name, "response": json.loads(result),
                    })
                    log.info("[%s] gemini tool call: %s(%s)", session_id, name, fc.get("args"))
                    await sink.send_event({"type": "tool_used", "name": name})
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

@contextlib.asynccontextmanager
async def _open_session(model: str, question_bank: dict):
    # * 연결 + setup + 초기 인사 턴까지 끝낸 upstream을 내준다. phone call/browser 양쪽 경로가 공유.

    async with websockets.connect(_realtime_url(model)) as upstream:
        await upstream.send(json.dumps(_build_setup_message(question_bank, model)))
        await upstream.recv()  # BidiGenerateContentSetupComplete — 이후부터 오디오 전송 안전

        # xAI GREETING_PROMPT과 같은 역할 — Gemini엔 1회성 시스템 메시지가 없어 user 턴으로 흉내(라이브 미검증).
        await upstream.send(json.dumps({
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": brain.GREETING_PROMPT}]}],
                "turnComplete": True,
            },
        }))
        yield upstream


async def run_session(client_ws: WebSocket, state: dict, question_bank: dict, model: str, session_id: str) -> bool:
    # * call_stream_ws가 xAI 대신 이걸 호출한다(model이 MODEL_PREFIX로 시작할 때만).
    # * 반환값은 call_stream_ws가 call_result 상태(COMPLETED/FAILED)에 그대로 반영한다.
    
    async with _open_session(model, question_bank) as upstream:
        upstream_task = asyncio.create_task(_pump_upstream(upstream, CallSink(client_ws), state, session_id))
        downstream_task = asyncio.create_task(_pump_downstream_call(client_ws, upstream))

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


async def run_browser_session(client_ws: WebSocket, state: dict, question_bank: dict, model: str, session_id: str) -> None:
    # * browser_ws가 xAI 대신 이걸 호출한다(model이 MODEL_PREFIX로 시작할 때만)."""
    
    async with _open_session(model, question_bank) as upstream:
        await client_ws.send_json({"type": "ready"})

        upstream_task = asyncio.create_task(_pump_upstream(upstream, BrowserSink(client_ws), state, session_id))
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

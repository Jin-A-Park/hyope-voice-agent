
# * 전화/브라우저 오디오 <-> Gemini Live API(BidiGenerateContent) 브릿지.

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import time

import websockets
from fastapi import WebSocket, WebSocketDisconnect

from agent import brain, prompts, tools as agent_tools
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

def _build_setup_message(model: str) -> dict:
    # * xAI의 session.update에 대응하는 Gemini Live의 setup 메시지 — 모델/음성/툴/VAD를 한 번에 설정.
    # * systemInstruction은 prompts.IDENTITY_PROMPT 한 줄만 싣는다 — 이게 없으면 모델이 툴 호출도
    # * 전에 자기 기본 정체성으로 반사적으로 인사부터 뱉는 문제가 있었다(server.py 참고). 나머지
    # * 규칙(질문/화제전환 등)은 여전히 next_step/GREETING_PROMPT로만 전달한다.

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
                "parts": [{"text": prompts.IDENTITY_PROMPT}],
            },
            "tools": [
                {"functionDeclarations": _to_function_declarations(agent_tools.build_tools())},
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
    elder_text_parts: list[str] = []
    # 응답을 다 생성했다고(turnComplete) 상대가 그걸 다 "재생"했다는 보장은 없다 — end_call로
    # 끊기 직전 마지막 발화가 재생될 시간을 벌어주기 위해 이번 턴에서 보낸 오디오 바이트 수를
    # 세어둔다(Gemini 출력도 PCM16 24kHz mono = 초당 48000바이트).
    turn_audio_bytes = 0

    # 지연시간 계측(server.py의 xAI 경로와 동일한 목적) — Gemini는 "어르신 발화 완료" 이벤트가
    # 따로 없어(inputTranscription은 델타로만 옴) 가장 최근 델타 수신 시각을 근사치로 쓴다.
    timing: dict[str, float | None] = {"voice_in": None, "tool_called": None}
    answer_started = False

    # * 진단용 카운터 — 대시보드에 대화내역(call_log_entries)이 안 올라가는 문제(실제 사례: 지표는
    # * 업데이트됐는데 대화내역만 비어 있음)의 원인이 "오디오는 오는데 inputTranscription만 안 오는
    # * 건지" 아니면 "애초에 오디오 자체가 이상한 건지" 로그만 보고 바로 구분하기 위함 — 통화가
    # * 끝날 때(finally) 한 줄로 요약해서 남긴다.
    event_counts = {"audio_chunk": 0, "output_transcription": 0, "input_transcription": 0, "turn_complete": 0}

    def _mark_answer_started() -> None:
        """이번 턴 들어 처음으로 실제 응답(오디오/텍스트)이 왔다 = LLM 답변 생성 시작."""
        nonlocal answer_started
        if answer_started:
            return
        answer_started = True
        if timing["tool_called"] is None:
            return
        elapsed_ms = (time.perf_counter() - timing["tool_called"]) * 1000
        log.info("[%s] 툴 호출 -> LLM 답변 생성: %.0f ms", session_id, elapsed_ms)
        timing["tool_called"] = None

    try:
        async for raw in upstream:
            event = json.loads(raw)

            server_content = event.get("serverContent")
            if server_content:
                model_turn = server_content.get("modelTurn") or {}
                for part in model_turn.get("parts", []):
                    inline = part.get("inlineData")
                    if inline and inline.get("data"):
                        event_counts["audio_chunk"] += 1
                        _mark_answer_started()
                        turn_audio_bytes += len(base64.b64decode(inline["data"]))
                        await sink.send_audio(inline["data"])
                out_text = (server_content.get("outputTranscription") or {}).get("text")
                if out_text:
                    event_counts["output_transcription"] += 1
                    _mark_answer_started()
                    if not agent_text_parts:  # 이 턴의 첫 델타 — call_log_entries.asked_at으로 쓸 시각을 못박는다
                        state["_last_agent_utterance_at"] = brain.spring_timestamp()
                    agent_text_parts.append(out_text)
                    await sink.send_event({"type": "transcript", "role": "agent", "delta": out_text})
                # * inputTranscription도 outputTranscription처럼 문장이 여러 조각(delta)으로 쪼개져 온다.
                # * 조각마다 완결 텍스트로 취급해 보내면(과거엔 "text"로 보냈음) 브라우저에서 말풍선이
                # * 조각 수만큼 새로 열려버린다 — 여기서도 agent와 동일하게 누적한다.
                in_text = (server_content.get("inputTranscription") or {}).get("text")
                if in_text:
                    event_counts["input_transcription"] += 1
                    timing["voice_in"] = time.perf_counter()  # 가장 최근 델타 수신 시각으로 근사
                    elder_text_parts.append(in_text)
                    await sink.send_event({"type": "transcript", "role": "elder", "delta": in_text})
                # * turnComplete는 "모델 턴이 끝났다"는 뜻일 뿐 "어르신 턴이 끝났다"는 보장이 아니다.
                # * 툴 호출 후 응답을 이어가는 동안 모델이 아무 말도 안 하는 무음 턴에서도 turnComplete가
                # * 여러 번 온다 — 그때마다 말풍선/call_log_entries를 확정하면 어르신이 그 사이 이어 말한
                # * 내용이 조각조각 분리되어 버린다. agent_text_parts가 실제로 채워진 턴(=모델이 진짜로
                # * 응답한 턴)에서만 확정한다.
                if server_content.get("turnComplete"):
                    event_counts["turn_complete"] += 1
                if server_content.get("turnComplete") and agent_text_parts:
                    if elder_text_parts:
                        elder_answer = "".join(elder_text_parts)
                        log.info("[%s] 어르신: %s", session_id, elder_answer)
                        if state.get("_last_agent_utterance"):
                            state["call_log_entries"].append({
                                "sequence": len(state["call_log_entries"]) + 1,
                                "question": state["_last_agent_utterance"],
                                "answer": elder_answer,
                                "asked_at": state.get("_last_agent_utterance_at") or brain.spring_timestamp(),
                            })
                        elder_text_parts.clear()
                    joined = "".join(agent_text_parts)
                    log.info("[%s] gemini: %s", session_id, joined)
                    state["_last_agent_utterance"] = joined
                    agent_text_parts.clear()
                    await sink.send_event({"type": "turn_done"})
                    if ending:
                        playback_seconds = turn_audio_bytes / (24000 * 2)  # PCM16 24kHz mono
                        await asyncio.sleep(playback_seconds + 0.5)
                        await sink.send_event({"type": "call_ended"})
                        return True
                    turn_audio_bytes = 0

            tool_call = event.get("toolCall")
            if tool_call:
                if timing["voice_in"] is not None:
                    names = ", ".join(fc["name"] for fc in tool_call["functionCalls"])
                    log.info(
                        "[%s] 음성 들어감 -> 툴 선정(%s): %.0f ms",
                        session_id, names, (time.perf_counter() - timing["voice_in"]) * 1000,
                    )
                    timing["voice_in"] = None
                # * 모델이 end_call과 다른 툴(주로 check_in)을 같은 턴에 같이 부르는 경우가 있다 —
                # * end_call이 끼어있으면 그게 이번 턴의 최종 결정이다. 다른 툴의 next_step("다음
                # * 질문 물어봐라")을 그대로 돌려주면 작별 인사와 다음 질문 안내가 한 응답 안에서
                # * 뒤섞여 뭉개진 문장이 나온다(실제 사례). 상태 반영(logic_data 등)은 그대로 하되,
                # * 그 결과의 next_step만 "무시하라"는 안내로 바꿔치기한다.
                calls = tool_call["functionCalls"]
                ending_this_batch = any(fc["name"] == "end_call" for fc in calls)
                function_responses = []
                for fc in calls:
                    name = fc["name"]
                    result = await dispatch.dispatch_tool(state, name, fc.get("args") or {})
                    parsed = json.loads(result)
                    if ending_this_batch and name != "end_call" and "next_step" in parsed:
                        parsed["next_step"] = (
                            "통화 종료 절차가 이미 진행 중입니다 — 이 안내는 무시하고, 방금 건넨 "
                            "작별 인사만 자연스럽게 마무리하세요."
                        )
                    function_responses.append({
                        "id": fc["id"], "name": name, "response": parsed,
                    })
                    log.info("[%s] gemini tool call: %s(%s)", session_id, name, fc.get("args"))
                    await sink.send_event({"type": "tool_used", "name": name})
                    if name == "end_call":
                        ending = True
                await upstream.send(json.dumps({
                    "toolResponse": {"functionResponses": function_responses},
                }))
                # 이 응답 이후 다음 오디오/텍스트 델타가 오면 그게 "답변 생성 시작"이다.
                timing["tool_called"] = time.perf_counter()
                answer_started = False
    except (WebSocketDisconnect, asyncio.CancelledError):
        raise
    except Exception:
        log.exception("[%s] gemini upstream pump failed", session_id)
        return False
    finally:
        log.info(
            "[%s] gemini 이벤트 요약 — 오디오청크=%d 출력전사=%d 입력전사=%d 턴종료=%d "
            "call_log_entries=%d",
            session_id, event_counts["audio_chunk"], event_counts["output_transcription"],
            event_counts["input_transcription"], event_counts["turn_complete"],
            len(state["call_log_entries"]),
        )
    return True

@contextlib.asynccontextmanager
async def _open_session(model: str):
    # * 연결 + setup + 초기 활성화 턴까지 끝낸 upstream을 내준다. phone call/browser 양쪽 경로가 공유.

    async with websockets.connect(_realtime_url(model)) as upstream:
        await upstream.send(json.dumps(_build_setup_message(model)))
        await upstream.recv()  # BidiGenerateContentSetupComplete — 이후부터 오디오 전송 안전

        # Gemini엔 xAI의 response.create 같은 1회성 트리거가 없어 user 턴으로 흉내낸다(라이브
        # 미검증) — 인사말 내용 자체는 여기서 안 실어보낸다. 모델이 이 턴을 받고 스스로 start_call을
        # 호출해 인사말 규칙(GREETING_PROMPT)을 받아 인사하고, 어르신이 답하면 check_in을 인자 없이
        # 호출해 첫 질문을 얻어와 그대로 말한다. 언어/정체성은 이제 systemInstruction(IDENTITY_PROMPT)
        # 이 세션 레벨에서 커버하므로 이 턴에 따로 안 실어도 된다.
        await upstream.send(json.dumps({
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": "(통화가 방금 연결되었습니다.)"}]}],
                "turnComplete": True,
            },
        }))
        yield upstream


async def run_session(client_ws: WebSocket, state: dict, model: str, session_id: str) -> bool:
    # * call_stream_ws가 xAI 대신 이걸 호출한다(model이 MODEL_PREFIX로 시작할 때만).
    # * 반환값은 call_stream_ws가 call_result 상태(COMPLETED/FAILED)에 그대로 반영한다.

    async with _open_session(model) as upstream:
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


async def run_browser_session(client_ws: WebSocket, state: dict, model: str, session_id: str) -> None:
    # * browser_ws가 xAI 대신 이걸 호출한다(model이 MODEL_PREFIX로 시작할 때만).

    async with _open_session(model) as upstream:
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

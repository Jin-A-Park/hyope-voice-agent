from __future__ import annotations

import asyncio
import audioop  # G.711 μ-law <-> PCM16 변환 + 리샘플링(claw-ops 브릿지 전용).
                 # 3.13에서 제거 예정 — 그때는 audioop-lts로 교체.
import base64
import json
import logging
import os
import time
import uuid

import requests
import websockets
from clawops import AsyncClawOps, WebhookVerificationError
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks, FastAPI, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect,
)
from fastapi.staticfiles import StaticFiles

import brain
import gemini_realtime

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dispatcher")

app = FastAPI(title="HYOPE AI voice agent")


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in; "
            f"model names and voices live at https://docs.x.ai."
        )
    return value


def realtime_url(model: str | None = None) -> str:
    base = os.environ.get("XAI_REALTIME_BASE_URL", "wss://api.x.ai/v1/realtime")
    model = model or os.environ.get("REALTIME_MODEL", "grok-voice-latest")
    return f"{base}?model={model}"

"""Spring 문서 기준 서버 간 인증 헤더는 X-API-KEY."""
def check_internal_key(x_api_key: str | None) -> None:
    expected = require_env("INTERNAL_API_KEY")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="unauthorized")

def turn_detection_config() -> dict:

    return {
        "type": "server_vad",
        "threshold": float(os.environ.get("VAD_THRESHOLD", "0.5")),
        "prefix_padding_ms": int(os.environ.get("PREFIX_MS", "300")),
        "silence_duration_ms": int(os.environ.get("SILENCE_MS", "900")),
        "idle_timeout_ms": int(os.environ.get("IDLE_TIMEOUT_MS", "30000")),
    }


# --------------------------------------------------------------------------
# claw-ops (실제 전화망) 브릿지. 에이전트 SDK 대신 직접 voiceML구현
# --------------------------------------------------------------------------

_claw_client: AsyncClawOps | None = None


def claw_client() -> AsyncClawOps:
    global _claw_client
    if _claw_client is None:
        _claw_client = AsyncClawOps(
            api_key=require_env("CLAWOPS_API_KEY"),
            account_id=require_env("CLAWOPS_ACCOUNT_ID"),
        )
    return _claw_client

def verify_claw_signature(url: str, params: dict, signature: str | None) -> None:
    if not signature:
        raise HTTPException(status_code=401, detail="missing X-Signature")
    try:
        claw_client().webhooks.verify(
            url=url, params=params, signature=signature,
            signing_key=require_env("CLAWOPS_SIGNING_KEY"),
        )
    except WebhookVerificationError:
        raise HTTPException(status_code=401, detail="invalid signature")

def public_base_url() -> str:
    return require_env("PUBLIC_BASE_URL").rstrip("/")


# POST /call에서 발신 시 채워두고, /ws/claw-stream의 start 이벤트(callId)로 꺼내
# state["_metadata"]에 심는다. HTTP 요청(/call)과 WS 접속(claw-ops가 나중에 별도로
# 걸어옴)이 서로 다른 요청이라 callId로 이어줄 방법이 이거 말고 없다 — 단일 프로세스
# 배포를 전제로 한 인메모리 저장(다른 상태 저장과 동일한 방식).
PENDING_CALL_METADATA: dict[str, dict] = {}

"""_pump_upstream(XAI -> 서버) 출력 어댑터 인터페이스. BrowserSink와 CallSink가 상속받아 사용"""
class OutputSink:

    async def send_audio(self, b64_pcm16_24k: str) -> None:
        raise NotImplementedError

    async def send_event(self, payload: dict) -> None:
        raise NotImplementedError

"""/ws/browser — PCM16 24kHz(브라우저 데모) 전달"""
class BrowserSink(OutputSink):

    def __init__(self, client_ws: WebSocket):
        self._ws = client_ws

    async def send_audio(self, b64_pcm16_24k: str) -> None:
        await self._ws.send_json({"type": "audio", "audio": b64_pcm16_24k})

    async def send_event(self, payload: dict) -> None:
        await self._ws.send_json(payload)

"""/ws/claw-stream — PCM16 24kHz -> μ-law 8kHz(통화 스트림) 변환"""
class CallSink(OutputSink):

    def __init__(self, client_ws: WebSocket):
        self._ws = client_ws
        self._resample_state = None

    async def send_audio(self, b64_pcm16_24k: str) -> None:
        b64_ulaw, self._resample_state = pcm24k_to_claw_media(b64_pcm16_24k, self._resample_state)
        await self._ws.send_json({"event": "media", "media": {"payload": b64_ulaw}})

    async def send_event(self, payload: dict) -> None:
        pass

"""브라우저(/ws/browser)에서 받아온 데이터 -> 모델"""
async def _pump_downstream(client_ws: WebSocket, upstream) -> None:
    while True:
        msg = await client_ws.receive_json()
        kind = msg.get("type")

        if kind == "audio":
            await upstream.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": msg["audio"],
            }))
        elif kind == "hangup":
            return

"""통화 스트림(/ws/claw-stream)에서 받아온 데이터 -> 오디오 변환 -> 모델"""
async def _pump_downstream_claw(client_ws: WebSocket, upstream, state: dict) -> None:

    resample_state = None
    while True:
        event = await client_ws.receive_json()
        etype = event.get("event")

        if etype == "media":
            b64_pcm16, resample_state = claw_media_to_pcm24k(event["media"]["payload"], resample_state)
            await upstream.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": b64_pcm16,
            }))
        elif etype == "dtmf":
            log.info("claw dtmf: %s", event.get("dtmf", {}).get("digit"))
        elif etype == "stop":
            return
        # connected/mark는 특별히 할 일이 없어 무시한다.

""" 오디오 트랜스코딩: claw-ops(G.711 μ-law, 8kHz) -> xAI Realtime(PCM16, 24kHz) """
def claw_media_to_pcm24k(b64_ulaw: str, state) -> tuple[str, object]:
    """claw-ops가 보낸 μ-law 8kHz 청크를 xAI Realtime이 기대하는 PCM16 24kHz로."""
    ulaw = base64.b64decode(b64_ulaw)
    pcm8k = audioop.ulaw2lin(ulaw, 2)
    pcm24k, state = audioop.ratecv(pcm8k, 2, 1, 8000, 24000, state)
    return base64.b64encode(pcm24k).decode("ascii"), state

""" 오디오 트랜스코딩: claw-ops(G.711 μ-law, 8kHz) <- xAI Realtime(PCM16, 24kHz) """
def pcm24k_to_claw_media(b64_pcm16_24k: str, state) -> tuple[str, object]:
    """xAI Realtime의 PCM16 24kHz 델타를 claw-ops Stream이 기대하는 μ-law 8kHz로."""
    leftover, resample_state = state if state else (b"", None)
    pcm24k = leftover + base64.b64decode(b64_pcm16_24k)
    if len(pcm24k) % 2:
        pcm24k, leftover = pcm24k[:-1], pcm24k[-1:]
    else:
        leftover = b""
    pcm8k, resample_state = audioop.ratecv(pcm24k, 2, 1, 24000, 8000, resample_state)
    ulaw = audioop.lin2ulaw(pcm8k, 2)
    return base64.b64encode(ulaw).decode("ascii"), (leftover, resample_state)


"""
1. xAI realtime ws 발화 -> 서버 -> sink(브라우저 or 전화망)
2. xAI realtime ws 이벤트 -> 서버 실행 -> 결과 다시 xAI -> 응답 재요청
"""
async def _pump_upstream(upstream, sink: OutputSink, state: dict, session_id: str) -> None:
    
    ending = False
    # response.create로 대답을 시킨다 — 그 사이에 클라이언트가 
    # 새 입력을 보내면 response.create가 겹치는 문제 발생. 
    # 그래서 "이 응답이 이어질 예정이다"를 추적해, 
    # 진짜로 할 말이 끝난 response.done에서만 turn_done을 보낸다.
    awaiting_continuation = False

    # 지연시간 계측: 구간 시작 시각만 들고 있다가, 그 구간을 끝내는 이벤트가 오면
    # 그 시각과의 차이를 ms로 로그에 남긴다(perf_counter라 단조 증가만 보장하면 됨).
    timing = {
        "voice_in": None,   # 어르신 발화 STT 완료 시각 — ~툴 선정까지 구간의 시작점
        "tool_called": None,  # 가장 최근 툴 호출 시각 — ~다음 답변 생성 시작까지 구간의 시작점
    }

    # 실제 대화 내용 로깅: 에이전트 발화는 delta로 쪼개져 오므로 한 턴이 끝날 때까지
    # 모았다가 한 줄로 찍는다. 어르신 발화는 STT 결과가 이미 완성된 문장으로 온다.
    agent_text_parts: list[str] = []

    def _mark_answer_started() -> None:
        """response.output_audio(.transcript).delta가 오는 시점 = LLM 답변 생성 시작."""
        if timing["tool_called"] is None:
            return
        elapsed_ms = (time.perf_counter() - timing["tool_called"]) * 1000
        log.info("[%s] 툴 호출 -> LLM 답변 생성: %.0f ms", session_id, elapsed_ms)
        timing["tool_called"] = None

    async for raw in upstream:
        event = json.loads(raw)
        etype = event.get("type", "")

        if etype == "response.created":
            awaiting_continuation = False

        elif etype == "response.output_audio.delta":
            _mark_answer_started()
            await sink.send_audio(event["delta"])

        elif etype == "response.output_audio_transcript.delta":
            _mark_answer_started()
            delta = event.get("delta", "")
            if not agent_text_parts:  # 이 턴의 첫 델타 — call_log_entries.asked_at으로 쓸 시각을 못박는다
                state["_last_agent_utterance_at"] = brain.spring_timestamp()
            agent_text_parts.append(delta)
            await sink.send_event({
                "type": "transcript", "role": "agent", "delta": delta,
            })

        elif etype == "conversation.item.input_audio_transcription.completed":
            timing["voice_in"] = time.perf_counter()
            transcript = event.get("transcript", "")
            log.info("\n[%s] 어르신: %s", session_id, transcript)  # 앞 줄바꿈으로 이전 턴과 구분
            # 직전에 에이전트가 실제로 뭔가 물어봤을 때만 Q&A 쌍으로 기록한다 —
            # 통화 시작 직후처럼 아직 아무 질문도 안 나간 상태의 발화는 짝지을 게 없다.
            if state.get("_last_agent_utterance"):
                state["call_log_entries"].append({
                    "sequence": len(state["call_log_entries"]) + 1,
                    "question": state["_last_agent_utterance"],
                    "answer": transcript,
                    "asked_at": state.get("_last_agent_utterance_at") or brain.spring_timestamp(),
                })
            await sink.send_event({
                "type": "transcript", "role": "elder", "text": transcript,
            })
        #모델이 tool 호출
        elif etype == "response.function_call_arguments.done":
            t_selected = time.perf_counter()
            name = event["name"]
            call_id = event["call_id"]
            args = json.loads(event.get("arguments") or "{}")

            if timing["voice_in"] is not None:
                log.info(
                    "[%s] 음성 들어감 -> 툴 선정(%s): %.0f ms",
                    session_id, name, (t_selected - timing["voice_in"]) * 1000,
                )

            t_called = time.perf_counter()
            log.info(
                "[%s] 툴 선정 -> 툴 호출(%s): %.1f ms",
                session_id, name, (t_called - t_selected) * 1000,
            )
            # 서버가 tool 실행
            result = brain.run_tool(state, name, args)
            log.info(
                "[%s] 툴 실행 소요(%s): %.1f ms",
                session_id, name, (time.perf_counter() - t_called) * 1000,
            )

            #결과 다시 서버 -> 모델
            await upstream.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                },
            }))

            # checkin_start만 예외: 여기서 response.create를 또 걸면 모델이 인사와 첫
            # 본 질문을 한 응답 안에 이어붙여 말해버린다(실제로 겪은 문제). 이 턴은 그냥
            # 끝내고 turn_done을 보내, 어르신이 실제로 뭐라고 답한 다음에야 첫 질문이
            # 나가게 만든다. response.create를 안 거니 뒤이은 답변 생성도 없어 여기서는
            # "툴 호출 -> 답변 생성" 구간을 재지 않는다.
            if name != "checkin_start":
                timing["tool_called"] = t_called
                await upstream.send(json.dumps({"type": "response.create"}))
                awaiting_continuation = True  # 방금 response.create를 새로 걸었다

            await sink.send_event({"type": "tool_used", "name": name})
            if name == "end_call":
                ending = True  # 작별 인사(다음 response)까지는 듣고 나서 끊는다

        elif etype == "response.done":
            if awaiting_continuation:
                continue  # 곧 이어지는 응답이 있다 — 아직 클라이언트에 turn_done을 보낼 때가 아니다
            if agent_text_parts:  # 이 턴에서 실제로 뭔가 말했으면 한 줄로 모아서 찍는다
                joined = "".join(agent_text_parts)
                log.info("[%s] 에이전트: %s", session_id, joined)
                state["_last_agent_utterance"] = joined
                agent_text_parts.clear()
            await sink.send_event({"type": "turn_done"})  # 다음 발화용 새 말풍선을 열라는 신호
            if ending:
                await sink.send_event({"type": "call_ended"})
                return

        elif etype == "error":
            log.error("upstream error: %s", event)
            await sink.send_event({
                "type": "error",
                "message": event.get("error", {}).get("message", "unknown error"),
            })


# --------------------------------------------------------------------------
# ws endpoint
# --------------------------------------------------------------------------

@app.websocket("/ws/browser")
async def browser_ws(client_ws: WebSocket) -> None:
    await client_ws.accept()
    session_id = f"session-{uuid.uuid4().hex[:8]}"  # 로그 상관관계용 id — 조회 용도로 쓰이진 않는다
    # 브라우저 데모엔 Spring이 실어주는 profile이 없다 — build_question_bank(None)이
    # static/recipient_profile.json으로 폴백한다.
    question_bank = brain.build_question_bank(None)
    state = brain.new_call_state(question_bank)
    state["_call_started_at"] = brain.spring_timestamp()
    model = client_ws.query_params.get("model")
    log.info("call started: %s (model=%s)", session_id, model or "(default)")

    try:
        if gemini_realtime.is_gemini_model(model):
            await gemini_realtime.run_browser_session(client_ws, state, question_bank, model, session_id)
            return

        async with websockets.connect(
            realtime_url(model),
            additional_headers={"Authorization": f"Bearer {require_env('XAI_API_KEY')}"},
        ) as upstream:
            await upstream.send(json.dumps({
                "type": "session.update",
                "session": {
                    "voice": os.environ.get("TTS_VOICE", "ara"),
                    "instructions": brain.build_full_instructions(question_bank),
                    "turn_detection": turn_detection_config(),
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "tools": brain.build_tools(question_bank),
                },
            }))

            # GREETING_PROMPT을 한 번만 슬쩍 끼워 넣어 통화의 첫 마디를 인사말로 유도한다
            # (SYSTEM_PROMPT에 영구히 섞어 넣으면 이후 턴에도 계속 "인사만 하라"고 남는다).
            await upstream.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": brain.GREETING_PROMPT}],
                },
            }))
            await upstream.send(json.dumps({"type": "response.create"}))

            await client_ws.send_json({"type": "ready"})

            upstream_task = asyncio.create_task(_pump_upstream(upstream, BrowserSink(client_ws), state, session_id))
            downstream_task = asyncio.create_task(_pump_downstream(client_ws, upstream))

            done, pending = await asyncio.wait(
                {upstream_task, downstream_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                    log.exception("call %s pump failed", session_id, exc_info=exc)

    except WebSocketDisconnect:
        log.info("client disconnected: %s", session_id)
    except Exception:
        log.exception("call %s failed", session_id)
        try:
            await client_ws.send_json({"type": "error", "message": "서버 오류로 통화가 종료되었습니다."})
        except Exception:
            pass
    finally:
        # 브라우저 데모는 recipient_id가 없어(Spring이 트리거한 통화가 아님) Spring으로
        # 보낼 대상이 없다 — 아웃박스 기록은 claw_stream_ws(실제 전화) 경로에서만 한다.
        try:
            await client_ws.close()
        except Exception:
            pass
        log.info("call ended: %s", session_id)


@app.websocket("/ws/claw-stream")
async def claw_stream_ws(client_ws: WebSocket) -> None:
    """claw-ops VoiceML의 <Connect><Stream>이 붙는 곳 — /ws/browser와 같은 구조지만

    상대가 브라우저가 아니라 claw-ops이므로 프로토콜과 오디오 포맷만 다르다
    (BrowserSink 대신 CallSink, _pump_downstream 대신 _pump_downstream_claw).
    """
    await client_ws.accept()
    session_id = f"claw-{uuid.uuid4().hex[:8]}"
    log.info("claw call started: %s", session_id)

    status = "FAILED"  # 아래서 정상 진행되면 COMPLETED로 덮어씀 — 중간에 예외로 빠지면 이 값 그대로 기록
    state: dict | None = None
    try:
        # session.update를 보내려면 이 통화의 profile(취미/근황)을 먼저 알아야 하는데,
        # 그건 POST /call이 PENDING_CALL_METADATA에 저장해둔 걸 claw-ops의 start
        # 이벤트(callId 포함)로만 꺼낼 수 있다 — 그래서 xAI 세션을 열기 전에 claw-ops의
        # start 이벤트부터 먼저 받는다(순서를 뒤집지 않으면 profile 없이 세션이 시작된다).
        call_id = None
        metadata: dict = {}
        while True:
            event = await client_ws.receive_json()
            if event.get("event") == "start":
                call_id = event["start"]["callId"]
                metadata = PENDING_CALL_METADATA.pop(call_id, {})
                break
            if event.get("event") == "stop":  # start도 오기 전에 끊긴 경우
                return
            # connected 등은 무시하고 start를 계속 기다린다.

        question_bank = brain.build_question_bank(metadata.get("profile"))
        state = brain.new_call_state(question_bank)
        state["_call_id"] = call_id
        state["_metadata"] = metadata
        state["_call_started_at"] = brain.spring_timestamp()

        model = metadata.get("model")
        log.info("claw call %s: model=%s", session_id, model or "(default)")

        if gemini_realtime.is_gemini_model(model):
            ok = await gemini_realtime.run_session(client_ws, state, question_bank, model, session_id)
            status = "COMPLETED" if ok else "FAILED"
        else:
            async with websockets.connect(
                realtime_url(model),
                additional_headers={"Authorization": f"Bearer {require_env('XAI_API_KEY')}"},
            ) as upstream:
                await upstream.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "voice": os.environ.get("TTS_VOICE", "ara"),
                        "instructions": brain.build_full_instructions(question_bank),
                        "turn_detection": turn_detection_config(),
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "tools": brain.build_tools(question_bank),
                    },
                }))

                await upstream.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": brain.GREETING_PROMPT}],
                    },
                }))
                await upstream.send(json.dumps({"type": "response.create"}))

                upstream_task = asyncio.create_task(
                    _pump_upstream(upstream, CallSink(client_ws), state, session_id)
                )
                downstream_task = asyncio.create_task(_pump_downstream_claw(client_ws, upstream, state))

                done, pending = await asyncio.wait(
                    {upstream_task, downstream_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                had_error = False
                for task in done:
                    exc = task.exception()
                    if exc and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                        log.exception("claw call %s pump failed", session_id, exc_info=exc)
                        had_error = True

            status = "FAILED" if had_error else "COMPLETED"

    except WebSocketDisconnect:
        log.info("claw-ops disconnected: %s", session_id)
    except Exception:
        log.exception("claw call %s failed", session_id)
    finally:
        if state is not None:
            call_id = state.get("_call_id")
            if call_id:
                # Stream 종료가 PSTN 통화 종료를 보장하는지 문서상 불명확 — REST로 명시 종료.
                try:
                    await claw_client().calls.update(call_id, status="completed")
                except Exception:
                    log.exception("failed to hang up claw call %s (%s)", session_id, call_id)
            brain.write_call_result_outbox(state, status)
        try:
            await client_ws.close()
        except Exception:
            pass
        log.info("claw call ended: %s", session_id)


async def _place_call(recipient_id, phone_number: str, profile: dict, model: str | None) -> None:
    """실제 claw-ops 발신 — Spring이 블로킹 대기 중인 POST /call 응답과 분리하기 위해

    BackgroundTasks로 여기서 실행된다. 실패하면 Spring 응답은 이미 나간 뒤라 알릴 방법이
    이거뿐이라, 웹훅 경로들과 똑같이 CALL_RESULT_OUTBOX에 FAILED로 바로 남긴다.
    """
    base = public_base_url()
    try:
        call = await claw_client().calls.create(
            to=phone_number,
            from_=require_env("CLAWOPS_FROM_NUMBER"),
            url=f"{base}/voiceml/answer",
            status_callback=f"{base}/webhooks/status",
            status_callback_event="initiated ringing answered completed",
            # 음성사서함이면 claw-ops가 알아서 끊게 한다 — AMD 결과가 통화 중 실시간으로
            # 오는지 문서상 불명확해서, 직접 판단하는 대신 내장 동작에 위임.
            machine_detection="Hangup",
        )
    except Exception:
        log.exception("발신 실패: recipient_id=%s", recipient_id)
        brain.write_minimal_call_result(recipient_id, "FAILED")
        return

    PENDING_CALL_METADATA[call.call_id] = {
        "recipient_id": recipient_id,
        "phone_number": phone_number,
        "profile": profile,
        "model": model,
    }


@app.post("/call")
async def create_call(
    request: Request,
    background_tasks: BackgroundTasks,
    x_api_key: str | None = Header(default=None, alias="X-API-KEY"),
) -> dict:
    """Spring이 이 서버에 대고 부르는 통화 트리거 엔드포인트.

    body: {"recipient_id": 1, "phone_number": "010-1234-5678",
           "profile": {"hobbies": [...], "recent_events": [{"id","text","added_at"}, ...]},
           "model": "grok-voice-think-fast-1.0"}
    (profile/model은 Spring 문서 원본 스펙엔 없는 확장 필드. model은 테스트 편의용 —
    생략하면 REALTIME_MODEL(.env) 기본값을 쓴다. profile은 Spring 팀과 별도 조율 필요.)

    Spring이 이 응답을 블로킹 대기하므로 즉시 accepted를 반환하고, 실제 발신(claw-ops
    REST 호출)은 백그라운드로 미룬다.
    """
    check_internal_key(x_api_key)
    body = await request.json()
    recipient_id = body.get("recipient_id")
    phone_number = (body.get("phone_number") or "").replace("-", "").replace(" ", "")
    if recipient_id is None or not phone_number:
        raise HTTPException(status_code=400, detail="'recipient_id' and 'phone_number' are required")

    background_tasks.add_task(
        _place_call, recipient_id, phone_number, body.get("profile") or {}, body.get("model"),
    )
    return {"status": "accepted"}


@app.post("/voiceml/answer")
async def voiceml_answer(request: Request, x_signature: str | None = Header(default=None)) -> Response:
    """claw-ops가 전화 연결 시 때리는 VoiceML 웹훅 — Stream으로 연결하라고 XML로 답한다."""
    form = dict(await request.form())
    base = public_base_url()
    verify_claw_signature(f"{base}/voiceml/answer", form, x_signature)

    stream_url = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/ws/claw-stream"
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response><Connect><Stream url="{stream_url}"/></Connect></Response>'
    )
    return Response(content=xml, media_type="application/xml")


_CLAW_TERMINAL_STATUSES = {"completed", "busy", "no-answer", "failed", "canceled", "rejected"}
# claw-ops 상태 -> Spring의 call_log.status. "completed"는 claw_stream_ws가 스트림을
# 실제로 붙였다는 뜻이라 거기서 이미 write_call_result_outbox를 부른다(여기선 다루지 않음).
_CLAW_STATUS_TO_SPRING = {
    "no-answer": "MISSED", "busy": "MISSED", "canceled": "MISSED", "rejected": "MISSED",
    "failed": "FAILED",
}


@app.post("/webhooks/status")
async def webhooks_status(request: Request, x_signature: str | None = Header(default=None)) -> Response:
    """claw-ops 통화 상태 콜백 — 로깅하고, 통화가 끝났는데 스트림까지 못 간(무응답/통화중/실패

    등) 메타데이터를 정리한다. 그 경우 claw_stream_ws가 아예 실행되지 않으므로(전화를
    안 받았으니 <Connect><Stream>까지 갈 일이 없음) Spring한테 결과를 알려줄 곳이 여기뿐이다.
    """
    form = dict(await request.form())
    base = public_base_url()
    verify_claw_signature(f"{base}/webhooks/status", form, x_signature)

    call_id = form.get("CallId")
    claw_status = form.get("CallStatus")
    log.info(
        "claw status: call=%s status=%s answered_by=%s hangup_cause=%s",
        call_id, claw_status, form.get("AnsweredBy"), form.get("HangupCause"),
    )
    if claw_status in _CLAW_TERMINAL_STATUSES:
        metadata = PENDING_CALL_METADATA.pop(call_id, None)
        spring_status = _CLAW_STATUS_TO_SPRING.get(claw_status)
        if metadata and spring_status and metadata.get("recipient_id") is not None:
            brain.write_minimal_call_result(metadata["recipient_id"], spring_status)

    return Response(status_code=204)

# Serve static/index.html at / (and index2.html at /index2.html) — mounted
# last so /ws/browser and /tts win the route match first.
app.mount("/", StaticFiles(directory="static", html=True), name="static")

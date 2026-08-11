from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import websockets
from clawops import AsyncClawOps, WebhookVerificationError
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks, FastAPI, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect,
)
from fastapi.staticfiles import StaticFiles

from agent import brain, tools as agent_tools
from integrations import dispatch, worker
from sinks import OutputSink, BrowserSink, CallSink, call_media_to_pcm24k

import gemini_loader


ROOT = Path(__file__).resolve().parent  # server.py가 이제 프로젝트 루트에 있어 .parent 한 번이면 됨

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dispatcher")

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # worker.py의 폴링 루프(runtime/ 알림·통화결과 파일 -> Spring)를 같은 프로세스의 백그라운드
    # 스레드로 돌린다 — Railway에 별도 서비스로 올리면 컨테이너가 갈려서 runtime/ 파일을
    # 서로 못 보게 된다. 같은 프로세스에 두면 파일시스템이 자동으로 공유된다.
    threading.Thread(target=worker.run_forever, daemon=True).start()
    yield

app = FastAPI(title="HYOPE AI voice agent", lifespan=_lifespan)

def require_env(name: str) -> str:
    # * 환경변수 받아오기

    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in; "
            f"model names and voices live at https://docs.x.ai."
        )
    return value

def realtime_url(model: str | None = None) -> str:
    # * 환경변수로 고정 x, 모델 선택

    base = os.environ.get("XAI_REALTIME_BASE_URL", "wss://api.x.ai/v1/realtime")
    model = model or os.environ.get("REALTIME_MODEL", "grok-voice-think-fast-2.0")
    return f"{base}?model={model}"

def check_internal_key(x_api_key: str | None) -> None:
    # * X-API-KEY: 와이어(HTTP 헤더) 이름. 
    # * INTERNAL_API_KEY: 서버 간(server-to-server) 내부 통신 비밀값

    expected = require_env("INTERNAL_API_KEY")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="unauthorized")

def turn_detection_config() -> dict:
    # * xAI Realtime의 server_vad 설정 — 어르신 통화 특성상 .env에서 "patient"하게 튜닝된 값.
    return {
        "type": "server_vad",
        "threshold": float(os.environ.get("VAD_THRESHOLD", "0.5")),
        "prefix_padding_ms": int(os.environ.get("PREFIX_MS", "300")),
        "silence_duration_ms": int(os.environ.get("SILENCE_MS", "900")),
        "idle_timeout_ms": int(os.environ.get("IDLE_TIMEOUT_MS", "30000")),
    }

# --------------------------------------------------------------------------
# call bridge (sink / 오디오 트랜스코딩 / pump 루프 — 브라우저·전화 공통)
# --------------------------------------------------------------------------

_call_client: AsyncClawOps | None = None

def call_client() -> AsyncClawOps:
    # * 프로세스당 한 번만 만들어서 재사용하는 calling 클라이언트 싱글턴.

    global _call_client
    if _call_client is None:
        _call_client = AsyncClawOps(
            api_key=require_env("CLAWOPS_API_KEY"),
            account_id=require_env("CLAWOPS_ACCOUNT_ID"),
        )
    return _call_client

def verify_claw_signature(url: str, params: dict, signature: str | None) -> None:
    # * claw-ops가 보낸 웹훅(voiceml/status)의 X-Signature를 검증 — 실패하면 401.

    if not signature:
        raise HTTPException(status_code=401, detail="missing X-Signature")
    try:
        call_client().webhooks.verify(
            url=url, params=params, signature=signature,
            signing_key=require_env("CLAWOPS_SIGNING_KEY"),
        )
    except WebhookVerificationError:
        raise HTTPException(status_code=401, detail="invalid signature")

def public_base_url() -> str:
    # * claw-ops에게 webhook/Stream URL을 알려줄 때 쓰는, 이 서버의 외부 접근(로컬은 ngrok) 가능한 base URL.

    return require_env("PUBLIC_BASE_URL").rstrip("/")

# ? POST /call에서 발신 시 채워두고, /ws/call-stream의 start 이벤트(callId)로 꺼내
# state["_metadata"]에 심는다. HTTP 요청(/call)과 WS 접속(claw-ops가 나중에 별도로
# 걸어옴)이 서로 다른 요청이라 callId로 이어줄 방법이 이거 말고 없다 — 단일 프로세스
# 배포를 전제로 한 인메모리 저장(다른 상태 저장과 동일한 방식).
PENDING_CALL_METADATA: dict[str, dict] = {}

async def _pump_downstream(client_ws: WebSocket, upstream) -> None:
    # * 브라우저(/ws/browser)에서 받아온 데이터 -> 모델

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

async def _pump_downstream_call(client_ws: WebSocket, upstream, state: dict) -> None:
    # * 통화 스트림(/ws/call-stream)에서 받아온 데이터 -> 오디오 포맷 변환 -> 모델

    resample_state = None
    while True:
        event = await client_ws.receive_json()
        etype = event.get("event")

        if etype == "media":
            b64_pcm16, resample_state = call_media_to_pcm24k(event["media"]["payload"], resample_state)
            await upstream.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": b64_pcm16,
            }))
        elif etype == "dtmf":
            log.info("phone call dtmf: %s", event.get("dtmf", {}).get("digit"))
        elif etype == "stop":
            return
        # connected/mark는 특별히 할 일이 없어 무시한다.

async def _pump_upstream(upstream, sink: OutputSink, state: dict, session_id: str) -> None:
    # * 1. xAI realtime ws 발화 -> 서버 -> sink(브라우저 or 전화망)
    # * or
    # * 2. xAI realtime ws 이벤트 -> 서버 실행 -> 결과 다시 xAI -> 응답 재요청

    ending = False

    # response.create 겹침을 막기 위해 "응답이 이어질 예정"을 추적, 진짜 끝난 response.done에서만 turn_done을 보낸다.
    awaiting_continuation = False

    # 지연시간 계측: 구간 시작 시각만 들고 있다가, 그 구간을 끝내는 이벤트가 오면
    # 그 시각과의 차이를 ms로 로그에 남긴다(perf_counter라 단조 증가만 보장하면 됨).
    timing = {
        "voice_in": None,   # 어르신 발화 STT 완료 시각 — ~툴 선정까지 구간의 시작점
        "tool_called": None,  # 가장 최근 툴 호출 시각 — ~다음 답변 생성 시작까지 구간의 시작점
    }

    # 대화 로깅용 — 에이전트 발화는 delta로 쪼개져 오므로 한 턴 끝날 때까지 모았다가 한 줄로 찍는다.
    agent_text_parts: list[str] = []

    # transcription.completed는 item당 status=in_progress -> completed 두 번 오고, 드물게
    # 같은 item_id로 completed가 통째로 재전송되기도 한다 — item당 최종본 한 번만 받아들인다.
    _completed_transcription_item_ids: set[str] = set()

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
            # status=in_progress인 중간 버전은 건너뛴다 — completed만 최종 텍스트다.
            if event.get("status") != "completed":
                continue
            # 같은 item_id로 completed가 또 오면(업스트림 재전송) 무시 — item당 한 번만 반영.
            item_id = event.get("item_id")
            if item_id in _completed_transcription_item_ids:
                continue
            if item_id is not None:
                _completed_transcription_item_ids.add(item_id)

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
            # 서버가 tool 실행 — flag_emergency(medium/low, 일반 신호)면 dispatch.dispatch_tool이
            # agent_tools.run_tool 부르기 전에 지오코딩으로 nearby_resource를 채워 넣는다.
            result = await dispatch.dispatch_tool(state, name, args)
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

            # checkin_start만 예외 — response.create를 걸면 모델이 인사와 첫 질문을 same 응답에 이어붙여 말해버림.
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
    # * 브라우저 데모용 엔드포인트 — 메인 서버 트리거 없이, 프론트 폼에서 받은 profile로 통화 세션을 연다.

    await client_ws.accept()
    session_id = f"session-{uuid.uuid4().hex[:8]}"  # 로그 상관관계용 id — 조회 용도로 쓰이진 않는다

    # 프론트(index.html)가 오디오 스트리밍 시작 전 {"type":"init", "phone_number", "profile"}를
    # 먼저 보낸다 — 실통화에서 PENDING_CALL_METADATA로 받는 것과 같은 모양이라, address 기반
    # 지오코딩/SMS 발송/취미·근황 카테고리까지 전화와 동일한 코드 경로로 테스트할 수 있다.
    init = await client_ws.receive_json()
    profile = init.get("profile") or None
    question_bank = brain.build_question_bank(profile)
    state = brain.new_call_state(question_bank)
    state["_call_started_at"] = brain.spring_timestamp()
    model = client_ws.query_params.get("model")
    state["_metadata"] = {
        "recipient_id": None,
        "phone_number": init.get("phone_number") or None,
        "profile": profile or {},
        "model": model,
    }
    log.info("call started: %s (model=%s)", session_id, model or "(default)")

    try:
        if gemini_loader.is_gemini_model(model):
            await gemini_loader.run_browser_session(client_ws, state, question_bank, model, session_id)
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
                    "tools": agent_tools.build_tools(question_bank),
                },
            }))

            # GREETING_PROMPT을 한 번만 끼워 넣어 통화의 첫 마디를 인사말로 유도한다 (시스템 프롬프트와 분리)
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
        # 브라우저 데모는 recipient_id가 없어 아웃박스 기록은 call_stream_ws(실제 전화) 경로에서만 한다.
        try:
            await client_ws.close()
        except Exception:
            pass
        log.info("call ended: %s", session_id)

@app.websocket("/ws/call-stream")
async def call_stream_ws(client_ws: WebSocket) -> None:
    # * claw-ops VoiceML의 <Connect><Stream>이 붙는 곳 — /ws/browser와 같은 구조에 CallSink/_pump_downstream_call만 다름.
    
    await client_ws.accept()
    session_id = f"phone call-{uuid.uuid4().hex[:8]}"
    log.info("phone call started: %s", session_id)

    status = "FAILED"  # 아래서 정상 진행되면 COMPLETED로 덮어씀 — 중간에 예외로 빠지면 이 값 그대로 기록
    state: dict | None = None
    try:
        # profile은 phone call start 이벤트(callId)로만 꺼낼 수 있어, xAI 세션을 열기 전에 그것부터 받는다.
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
        log.info("phone call %s: model=%s", session_id, model or "(default)")

        if gemini_loader.is_gemini_model(model):
            ok = await gemini_loader.run_session(client_ws, state, question_bank, model, session_id)
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
                        "tools": agent_tools.build_tools(question_bank),
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
                downstream_task = asyncio.create_task(_pump_downstream_call(client_ws, upstream, state))

                done, pending = await asyncio.wait(
                    {upstream_task, downstream_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                had_error = False
                for task in done:
                    exc = task.exception()
                    if exc and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                        log.exception("phone call %s pump failed", session_id, exc_info=exc)
                        had_error = True

            status = "FAILED" if had_error else "COMPLETED"

    except WebSocketDisconnect:
        log.info("phone call disconnected: %s", session_id)
    except Exception:
        log.exception("phone call %s failed", session_id)
    finally:
        if state is not None:
            call_id = state.get("_call_id")
            if call_id:
                # Stream 종료가 PSTN 통화 종료를 보장하는지 문서상 불명확 — REST로 명시 종료.
                try:
                    await call_client().calls.update(call_id, status="completed")
                except Exception:
                    log.exception("failed to hang up phone call %s (%s)", session_id, call_id)
            brain.write_call_result_outbox(state, status)
        try:
            await client_ws.close()
        except Exception:
            pass
        log.info("phone call ended: %s", session_id)


async def _place_call(recipient_id, phone_number: str, profile: dict, model: str | None) -> None:
    # * 실제 전화 발신 — POST /call 응답과 분리하려 BackgroundTasks로 실행, 실패 시 CALL_RESULT_OUTBOX에 FAILED로 남긴다.
    # * profile.address는 여기서 미리 안 쓰고 그대로 PENDING_CALL_METADATA에 실어 보낸다 —
    # * 지오코딩은 실제로 flag_emergency가 필요로 하는 순간에만 한다(integrations/dispatch.py의 dispatch_tool).

    base = public_base_url()
    try:
        call = await call_client().calls.create(
            to=phone_number,
            from_=require_env("CLAWOPS_FROM_NUMBER"),
            url=f"{base}/voiceml/answer",
            status_callback=f"{base}/webhooks/status",
            status_callback_event="initiated ringing answered completed",
            # 음성사서함이면 claw-ops가 알아서 끊게 한다 
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
    # * 메인 서버에서 부르는 통화 트리거. body: {recipient_id, phone_number, profile?, model?}
    # * 블로킹 대기 중인 Spring에 즉시 accepted를 반환하고 실제 발신은 백그라운드로.

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
    # * claw-ops가 전화 연결 시 때리는 VoiceML 웹훅 — Stream으로 연결하라고 XML로 답한다.
    
    form = dict(await request.form())
    base = public_base_url()
    verify_claw_signature(f"{base}/voiceml/answer", form, x_signature)

    stream_url = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/ws/call-stream"
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response><Connect><Stream url="{stream_url}"/></Connect></Response>'
    )
    return Response(content=xml, media_type="application/xml")

# * 전화 통신 상태(메인 서버의 call_log.status로 전달용)
_CALL_TERMINAL_STATUSES = {"completed", "busy", "no-answer", "failed", "canceled", "rejected"}
_CALL_STATUS_TO_SPRING = {
    "no-answer": "MISSED", "busy": "MISSED", "canceled": "MISSED", "rejected": "MISSED",
    "failed": "FAILED",
}

@app.post("/webhooks/status")
async def webhooks_status(request: Request, x_signature: str | None = Header(default=None)) -> Response:
    # * 통화 상태 콜백 — 스트림까지 못 간(무응답/통화중/실패) 경우 따로 정리.
    
    form = dict(await request.form())
    base = public_base_url()
    verify_claw_signature(f"{base}/webhooks/status", form, x_signature)

    call_id = form.get("CallId")
    call_status = form.get("CallStatus")
    log.info(
        "phone call status: call=%s status=%s answered_by=%s hangup_cause=%s",
        call_id, call_status, form.get("AnsweredBy"), form.get("HangupCause"),
    )
    if call_status in _CALL_TERMINAL_STATUSES:
        metadata = PENDING_CALL_METADATA.pop(call_id, None)
        spring_status = _CALL_STATUS_TO_SPRING.get(call_status)
        if metadata and spring_status and metadata.get("recipient_id") is not None:
            brain.write_minimal_call_result(metadata["recipient_id"], spring_status)

    return Response(status_code=204)

# static/index.html을 /에 서빙 — /ws/browser 등 다른 라우트가 먼저 매치되도록 맨 마지막에 mount.
app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")

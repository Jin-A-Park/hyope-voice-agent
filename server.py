from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
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

from agent import brain, prompts, tools as agent_tools
from integrations import dispatch, worker
from sinks import OutputSink, BrowserSink, CallSink, call_media_rms, call_media_to_pcm24k

import gemini_loader


ROOT = Path(__file__).resolve().parent  # server.py가 이제 프로젝트 루트에 있어 .parent 한 번이면 됨

load_dotenv()

# * 서드파티(websockets/httpx/watchfiles 등)는 ROOT_LOG_LEVEL로 조용히 시키고, 우리 로거
# * (dispatcher/tools/worker)만 LOG_LEVEL로 따로 살려서 요청 로그 소음과 우리 로그를 분리한다.
# * .env에서 값만 바꿔서 재시작하면 되고, 코드는 안 건드려도 된다.
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
ROOT_LOG_LEVEL = os.environ.get("ROOT_LOG_LEVEL", "WARNING").upper()

logging.basicConfig(level=ROOT_LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
for _name in ("dispatcher", "tools", "worker"):
    logging.getLogger(_name).setLevel(LOG_LEVEL)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

log = logging.getLogger("dispatcher")

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # worker.py의 폴링 루프(runtime/ 알림·통화결과 파일 -> Spring)를 같은 프로세스의 백그라운드
    # 스레드로 돌린다 — Railway에 별도 서비스로 올리면 컨테이너가 갈려서 runtime/ 파일을
    # 서로 못 보게 된다. 같은 프로세스에 두면 파일시스템이 자동으로 공유된다.
    # ENABLE_SPRING_WORKER=false면 이 스레드를 아예 안 띄운다 — Spring/hyope-serve 없이
    # 에이전트만 단독으로(웹 데모) 돌릴 때 사용. 기본값은 켜짐(기존과 동일한 동작).
    if os.environ.get("ENABLE_SPRING_WORKER", "true").strip().lower() != "false":
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
    # "think" 계열 변형은 reasoning 과정이 별도 채널 없이 음성 스트림에 그대로 새어나온다(오디오
    # 델타를 버퍼링 없이 즉시 전화로 흘려보내는 구조라 걸러낼 지점이 없다) — .env의 REALTIME_MODEL이
    # 안 잡혀있는 환경에서도 실수로 think 변형으로 안 떨어지도록 폴백 기본값을 non-think로 맞춘다.
    model = model or os.environ.get("REALTIME_MODEL", "grok-voice-latest")
    return f"{base}?model={model}"

def check_internal_key(x_api_key: str | None) -> None:
    # * X-API-KEY: 와이어(HTTP 헤더) 이름. 
    # * INTERNAL_API_KEY: 서버 간(server-to-server) 내부 통신 비밀값

    expected = require_env("INTERNAL_API_KEY")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


def turn_detection_config(threshold: float | None = None) -> dict:
    # * xAI Realtime의 server_vad 설정 — 어르신 통화 특성상 .env에서 "patient"하게 튜닝된 값.
    # * threshold를 넘기면 .env의 VAD_THRESHOLD 대신 그 값을 쓴다 — 통화별 주변 소음 보정용
    # * (_calibrate_vad_threshold 참고). 안 넘기면(브라우저 데모 등) 기존처럼 고정값을 쓴다.
    return {
        "type": "server_vad",
        "threshold": threshold if threshold is not None else float(os.environ.get("VAD_THRESHOLD", "0.5")),
        "prefix_padding_ms": int(os.environ.get("PREFIX_MS", "300")),
        "silence_duration_ms": int(os.environ.get("SILENCE_MS", "750")),
        "idle_timeout_ms": int(os.environ.get("IDLE_TIMEOUT_MS", "30000")),
    }


# _calibrate_vad_threshold의 보정 범위 — 검증된 기본값(VAD_THRESHOLD=0.5)에서 너무 멀리 벗어나면
# 오탐/미탐 위험이 커지므로, 측정한 주변 소음과 무관하게 이 범위 밖으로는 안 나가게 막는다.
# 절대적인 보정 공식이 아니라 시작점이다 — 실제 통화로 재보면서 재조정이 필요할 수 있다.
_VAD_THRESHOLD_MIN = 0.35
_VAD_THRESHOLD_MAX = 0.65
# 이 RMS(대략적인 조용한 방~약간 시끄러운 방 범위) 구간에서 위 min/max로 선형 보간한다.
_QUIET_RMS = 150
_NOISY_RMS = 1500
_CALIBRATION_TIMEOUT_S = 0.8  # 이 안에 충분한 샘플을 못 모으면 그냥 기본값(0.5)으로 넘어간다


async def _calibrate_vad_threshold(client_ws: WebSocket) -> float:
    # * 통화 연결 직후, 인사말을 시작하기 전 아주 짧게(최대 0.8초) 전화선 오디오를 들어서 주변
    # * 소음 수준을 재고 VAD threshold를 세션별로 보정한다 — 이 구간의 오디오는 모델에 전달하지
    # * 않고 버린다(아직 인사도 안 나갔으니 어르신 발화가 담겨있을 가능성이 거의 없다). 실패하거나
    # * 샘플이 부족하면 기존 고정 기본값(0.5)으로 조용히 넘어간다 — 이 보정은 있으면 좋은 것이지
    # * 통화 진행의 필수 조건이 아니다.
    samples: list[int] = []
    try:
        async def _collect():
            while True:
                event = await client_ws.receive_json()
                if event.get("event") == "media":
                    samples.append(call_media_rms(event["media"]["payload"]))
                elif event.get("event") == "stop":
                    return
        await asyncio.wait_for(_collect(), timeout=_CALIBRATION_TIMEOUT_S)
    except asyncio.TimeoutError:
        pass
    except Exception:
        log.exception("VAD 보정 중 오류 — 기본값으로 진행")

    if len(samples) < 5:  # 너무 적으면 노이즈 추정이 신뢰할 수 없다
        log.info("VAD 보정: 샘플 부족(%d개) — 기본값 0.5 사용", len(samples))
        return 0.5

    avg_rms = sum(samples) / len(samples)
    ratio = (avg_rms - _QUIET_RMS) / (_NOISY_RMS - _QUIET_RMS)
    ratio = max(0.0, min(1.0, ratio))
    threshold = _VAD_THRESHOLD_MIN + ratio * (_VAD_THRESHOLD_MAX - _VAD_THRESHOLD_MIN)
    log.info(
        "VAD 보정: 샘플 %d개, 평균 RMS %.0f -> threshold %.2f",
        len(samples), avg_rms, threshold,
    )
    return threshold

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

# --------------------------------------------------------------------------
# 필러(backchannel) 오디오 — 툴 호출 후 모델의 다음 응답이 늦어질 때(추론 지연) 침묵 대신
# 짧게 재생해서 통화가 끊긴 것처럼 느껴지지 않게 한다. static/fillers/*.b64는
# scratchpad/gen_fillers.py로 실제 통화 목소리(.env의 TTS_VOICE)에 맞춰 미리 합성해둔
# PCM16 24kHz 원본이다 — 모델과 무관하게 서버가 직접 재생하므로 "말하기 전에 함수부터
# 호출하라"는 기존 지시(중복 발화 방지)와 충돌하지 않는다.
FILLER_CLIPS: list[str] = []
_fillers_dir = ROOT / "static" / "fillers"
if _fillers_dir.exists():
    for _p in sorted(_fillers_dir.glob("*.b64")):
        FILLER_CLIPS.append(_p.read_text().strip())

FILLER_DELAY_S = 0.45  # 이 안에 모델의 진짜 응답 오디오가 안 오면 필러를 재생한다


async def _play_filler_after_delay(sink: OutputSink, delay: float) -> None:
    await asyncio.sleep(delay)
    if FILLER_CLIPS:
        await sink.send_audio(random.choice(FILLER_CLIPS))


def _cancel_filler(task: asyncio.Task | None) -> None:
    if task is not None and not task.done():
        task.cancel()


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

    # "가장 최근 툴 호출 이후로 뭔가 말했는지"(response_had_audio)로 이어서 response.create를 걸지
    # 판단한다. 아직 진행 중인 응답이 있는데 response.create를 또 걸면 두 응답이 겹쳐서
    # 오디오/transcript가 뒤섞이는(문장 중복) 문제가 생기므로, 반드시 response.done으로 그 응답이
    # 확정적으로 끝난 뒤에만 이어서 건다. response_had_audio는 응답 전체 기준이 아니라 매 툴 호출마다
    # 리셋된다 — 한 응답 안에서 "공감 발언 -> 툴 호출 -> (말 안 하고 끝)"처럼 이전에 뭔가 말했다는
    # 이유로 "이미 끝났다"고 오판하면 안 되기 때문. response_pending은 "이 응답이 끝나면 이어서
    # 걸어야 하는지".
    response_pending = False
    response_had_audio = False
    # 응답을 다 생성했다고(response.done) 클라이언트/통화 상대가 그걸 다 "재생"했다는 보장은 없다 —
    # end_call로 끊기 직전 마지막 발화가 재생될 시간을 벌어주기 위해 이번 응답에서 보낸 오디오
    # 바이트 수를 세어둔다(PCM16 24kHz mono = 초당 48000바이트).
    response_audio_bytes = 0

    # 드물게 모델이 응답 하나를 툴 호출도 오디오도 전혀 없이 완전히 빈 채로 끝내버릴 때가 있다 —
    # 이러면 response_pending도 안 걸려 있어서 그냥 조용히 턴이 끝나고, 어르신은 아무 질문도 못 받은
    # 채 서버는 어르신 말만 기다리는 상태가 된다. 이걸 감지해서 재시도한다(무한루프 방지용 상한 포함).
    response_had_activity = False
    blank_response_retries = 0
    MAX_BLANK_RESPONSE_RETRIES = 3

    # 업스트림이 같은 함수 호출 이벤트를 재전송하는 경우가 있다(아래 input_audio_transcription과
    # 동일한 현상) — call_id 기준으로 한 번만 처리해서 같은 툴이 중복 실행/중복 응답되는 걸 막는다.
    _handled_call_ids: set[str] = set()

    # 툴 호출 후 모델의 다음 응답(진짜 음성)이 FILLER_DELAY_S 안에 안 오면 필러를 재생한다 —
    # 진짜 오디오가 시작되거나 응답이 끝나면 아직 안 울린 필러는 취소한다.
    pending_filler_task: asyncio.Task | None = None

    # 지연시간 계측: 구간 시작 시각만 들고 있다가, 그 구간을 끝내는 이벤트가 오면
    # 그 시각과의 차이를 ms로 로그에 남긴다(perf_counter라 단조 증가만 보장하면 됨).
    timing = {
        "voice_in": None,   # 어르신 발화 STT 완료 시각 — ~툴 선정까지 구간의 시작점
        "tool_called": None,  # 가장 최근 툴 호출 시각 — ~다음 답변 생성 시작까지 구간의 시작점
    }

    # 대화 로깅용 — 에이전트 발화는 delta로 쪼개져 오므로 한 턴 끝날 때까지 모았다가 한 줄로 찍는다.
    agent_text_parts: list[str] = []
    # 어르신 발화도 마찬가지 — 통화 첫 start_call처럼 response.create를 안 거는 무음 구간에서는
    # input_audio_transcription.completed가 여러 번 올 수 있다(짧은 말을 여러 번 나눠 함).
    # 매번 바로 call_log_entries/말풍선을 확정하면 그 사이 이어 말한 내용이 조각조각 분리되어
    # 버린다 — 에이전트가 실제로 다음 응답을 낼 때(response.done)까지 모았다가 한 번에 확정한다.
    elder_text_parts: list[str] = []
    # 진단 결과: transcription.completed는 item당 status=in_progress -> completed 두 번 오고,
    # 드물게 같은 item_id가 completed까지 통째로 한 번 더 온다(업스트림 재전송으로 보임) —
    # 최종본만, item당 한 번만 받아들이기 위한 추적 세트.
    _completed_transcription_item_ids: set[str] = set()

    def _mark_answer_started() -> None:
        """response.output_audio(.transcript).delta가 오는 시점 = LLM 답변 생성 시작."""
        _cancel_filler(pending_filler_task)  # 진짜 응답이 시작됐으니 대기 중인 필러는 취소
        if timing["tool_called"] is None:
            return
        elapsed_ms = (time.perf_counter() - timing["tool_called"]) * 1000
        log.info("[%s] 툴 호출 -> LLM 답변 생성: %.0f ms", session_id, elapsed_ms)
        timing["tool_called"] = None

    async for raw in upstream:
        event = json.loads(raw)
        etype = event.get("type", "")

        if etype == "response.created":
            response_had_audio = False  # 새 응답이 시작됨 — 이 응답의 발화 여부를 새로 추적
            response_audio_bytes = 0
            response_had_activity = False

        elif etype == "response.output_audio.delta":
            _mark_answer_started()
            response_had_audio = True
            response_had_activity = True
            response_audio_bytes += len(base64.b64decode(event["delta"]))
            await sink.send_audio(event["delta"])

        elif etype == "response.output_audio_transcript.delta":
            _mark_answer_started()
            response_had_audio = True
            response_had_activity = True
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
            delta = (" " if elder_text_parts else "") + transcript
            elder_text_parts.append(transcript)
            await sink.send_event({
                "type": "transcript", "role": "elder", "delta": delta,
            })
        #모델이 tool 호출
        elif etype == "response.function_call_arguments.done":
            call_id = event["call_id"]
            if call_id in _handled_call_ids:
                continue  # 업스트림 재전송 — 이미 처리한 함수 호출은 건너뛴다(중복 실행/중복 발화 방지)
            _handled_call_ids.add(call_id)
            response_had_activity = True

            t_selected = time.perf_counter()
            name = event["name"]
            args = json.loads(event.get("arguments") or "{}")

            if timing["voice_in"] is not None:
                log.info(
                    "[%s] 음성 들어감 -> 툴 선정(%s): %.0f ms",
                    session_id, name, (t_selected - timing["voice_in"]) * 1000,
                )
                # 한 번 로그로 소비했으면 리셋 — 안 그러면 어르신의 새 발화 없이 툴이 연속 호출될 때
                # (예: check_in이 새 답변 없이 또 불릴 때) 예전 발화 시각을 재사용해서 마치 방금
                # 어르신이 말한 것처럼 보이는 오해의 소지가 있는 로그가 남는다.
                timing["voice_in"] = None

            t_called = time.perf_counter()
            log.info(
                "[%s] 툴 선정 -> 툴 호출(%s): %.1f ms",
                session_id, name, (t_called - t_selected) * 1000,
            )
            # 서버가 tool 실행 — check_external_api_necessity면 dispatch.dispatch_tool이
            # agent_tools.run_tool 부르기 전에 지오코딩/SMS 발송으로 nearby_resource/sms_sent를 채워 넣는다.
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

            # 지금 당장 response.create를 걸지 않는다 — 이 응답이 실제로 끝나는 시점(response.done)에
            # "이 툴 호출 이후로 뭔가 말했는지"(response_had_audio) 보고 판단한다(아래 response.done
            # 참고). 미리 걸면 아직 진행 중인 응답과 겹쳐서 두 응답이 동시에 말하는 문제가 생긴다.
            # 응답 전체가 아니라 "가장 최근 툴 호출 이후"를 봐야 한다 — 한 응답 안에서 공감 발언 뒤
            # 곧바로 다음 툴을 호출했는데 그 결과(다음 질문)를 마저 말하지 않고 응답이 끝나버리는
            # 경우가 있어서(이전 발언이 있었다는 이유로 "이미 끝났다"고 오판하면 안 됨), 매 툴
            # 호출마다 이 플래그를 리셋한다.
            #
            # 단, update_recipient_profile처럼 결과에 next_step이 없는 툴은 예외다 — 이 툴은
            # 설계상 "더 말할 것 없음"인 툴이라(어르신 답을 기다려야 하는 상황에서 조용히 기록만
            # 함), 이어서 강제로
            # response.create를 걸면 안 된다 — 그러면 어르신 답을 기다려야 할 타이밍에 서버가
            # 모델을 강제로 한 번 더 말하게 만들어버린다.
            #
            # end_call도 리셋에서 예외다 — check_in/end_call 스키마 둘 다 "먼저 작별 인사를 말한
            # 뒤에 end_call을 호출하라"고 지시하므로, 인사는 항상 이 툴 호출 *이전에* 이미 끝나 있다.
            # 무조건 리셋하면 방금 한 인사를 못 본 걸로 치고 아래 response.done에서 강제로
            # response.create를 걸어버려서, 모델이 end_call() 자신의 next_step을 보고 작별 인사를
            # 한 번 더 말하는 중복이 생긴다 — 리셋을 건너뛰면 이미 오디오가 나간 정상 케이스는
            # 그대로 끝나고, 모델이 인사 없이 바로 end_call부터 부른 예외 상황에서만
            # response_had_audio가 여전히 False라 안전망(강제 재시도)이 살아있다.
            if "next_step" in json.loads(result):
                response_pending = True
                if name != "end_call":
                    response_had_audio = False
                # 모델이 이 결과를 받고 이어서 말할 걸로 예상되는 경우에만 필러를 대기시킨다 —
                # update_recipient_profile처럼 next_step이 없는 툴(조용히 기록만 함)은 대상 아님.
                # start_call(통화의 첫 툴 호출, 아직 인사말도 안 나간 시점)도 제외한다 — 대화가
                # 시작되기도 전에 낯선 소리부터 나가면 오히려 어색하다. 이미 대기 중인 필러가
                # 있으면(짧은 시간 안에 툴이 연달아 불린 경우) 새로 하나로 교체한다 — 필러가
                # 겹쳐서 두 번 들리면 안 되니까.
                if name != "start_call":
                    _cancel_filler(pending_filler_task)
                    pending_filler_task = asyncio.create_task(_play_filler_after_delay(sink, FILLER_DELAY_S))
            timing["tool_called"] = t_called

            await sink.send_event({"type": "tool_used", "name": name})
            if name == "end_call":
                ending = True  # 작별 인사(다음 response)까지는 듣고 나서 끊는다

        elif etype == "response.done":
            if not response_had_activity:
                # 이 응답이 툴 호출도 오디오도 전혀 없이 완전히 빈 채로 끝났다 — 정상적인 흐름이
                # 아니다(모든 응답은 항상 말을 하거나 툴을 호출해야 한다). 상한 안에서 다시
                # 이어본다(무한루프로 API를 계속 두드리지 않도록 상한을 둔다).
                if blank_response_retries < MAX_BLANK_RESPONSE_RETRIES:
                    blank_response_retries += 1
                    log.warning(
                        "[%s] 응답이 아무것도 안 하고 끝남 — 재시도 %d/%d",
                        session_id, blank_response_retries, MAX_BLANK_RESPONSE_RETRIES,
                    )
                    response_pending = False
                    await upstream.send(json.dumps({"type": "response.create"}))
                    continue
                log.error("[%s] 응답이 계속 아무것도 안 함 — 재시도 포기하고 턴을 끝낸다", session_id)
            blank_response_retries = 0

            if response_pending and not response_had_audio:
                # 방금 응답이 함수 호출만 하고 끝났다(아무 말도 안 함) — 이어서 response.create를
                # 걸어야 실제로 말을 한다. 이미 뭔가 말했다면(response_had_audio) 정상적인 턴의
                # 끝이므로 아래로 내려가 그대로 turn_done을 보낸다.
                response_pending = False
                await upstream.send(json.dumps({"type": "response.create"}))
                continue
            response_pending = False
            if agent_text_parts:  # 이 턴에서 실제로 뭔가 말했으면 한 줄로 모아서 찍는다
                # 직전에 에이전트가 실제로 뭔가 물어봤을 때만 Q&A 쌍으로 기록한다 —
                # 통화 시작 직후처럼 아직 아무 질문도 안 나간 상태의 발화는 짝지을 게 없다.
                if elder_text_parts:
                    if state.get("_last_agent_utterance"):
                        state["call_log_entries"].append({
                            "sequence": len(state["call_log_entries"]) + 1,
                            "question": state["_last_agent_utterance"],
                            "answer": " ".join(elder_text_parts),
                            "asked_at": state.get("_last_agent_utterance_at") or brain.spring_timestamp(),
                        })
                    elder_text_parts.clear()
                joined = "".join(agent_text_parts)
                log.info("[%s] 에이전트: %s", session_id, joined)
                state["_last_agent_utterance"] = joined
                agent_text_parts.clear()
            await sink.send_event({"type": "turn_done"})  # 다음 발화용 새 말풍선을 열라는 신호
            if ending:
                # 오디오를 다 보냈다고 상대가 다 들었다는 보장은 없다 — 방금 응답의 실제 재생
                # 시간만큼(+여유) 기다렸다가 끊어야 마지막 작별 인사가 중간에 잘리지 않는다.
                playback_seconds = response_audio_bytes / (24000 * 2)  # PCM16 24kHz mono
                await asyncio.sleep(playback_seconds + 0.5)
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
    state = brain.new_call_state(session_id, profile)
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
            await gemini_loader.run_browser_session(client_ws, state, model, session_id)
            return

        async with websockets.connect(
            realtime_url(model),
            additional_headers={"Authorization": f"Bearer {require_env('XAI_API_KEY')}"},
        ) as upstream:
            await upstream.send(json.dumps({
                "type": "session.update",
                "session": {
                    "voice": os.environ.get("TTS_VOICE", "ara"),
                    "instructions": prompts.IDENTITY_PROMPT,
                    "turn_detection": turn_detection_config(),
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "tools": agent_tools.build_tools(),
                },
            }))

            # 인사말 문구 자체는 별도 주입하지 않는다 — 모델이 이 첫 response.create 안에서 스스로
            # start_call을 호출해 인사말 규칙(GREETING_PROMPT)을 받아 인사하고, 어르신이 답하면
            # check_in을 인자 없이 호출해 첫 질문을 얻어와 그대로 말한다.
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

        state = brain.new_call_state(call_id, metadata.get("profile"))
        state["_call_id"] = call_id
        state["_metadata"] = metadata
        state["_call_started_at"] = brain.spring_timestamp()

        model = metadata.get("model")
        log.info("phone call %s: model=%s", session_id, model or "(default)")

        if gemini_loader.is_gemini_model(model):
            ok = await gemini_loader.run_session(client_ws, state, model, session_id)
            status = "COMPLETED" if ok else "FAILED"
        else:
            # 인사말이 나가기 전 아주 짧게 전화선 소음을 재서 이번 통화의 VAD threshold를 보정한다
            # (조용한 집/시끄러운 환경에 따라 오탐지 위험이 다르므로) — xAI 세션을 열기 전에 해야
            # 그 결과를 session.update에 바로 실을 수 있다.
            vad_threshold = await _calibrate_vad_threshold(client_ws)
            async with websockets.connect(
                realtime_url(model),
                additional_headers={"Authorization": f"Bearer {require_env('XAI_API_KEY')}"},
            ) as upstream:
                await upstream.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "voice": os.environ.get("TTS_VOICE", "ara"),
                        "instructions": prompts.IDENTITY_PROMPT,
                        "turn_detection": turn_detection_config(vad_threshold),
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "tools": agent_tools.build_tools(),
                    },
                }))

                # 인사말 문구 자체는 별도 주입하지 않는다 — 모델이 이 첫 response.create 안에서
                # 스스로 start_call을 호출해 인사말 규칙(GREETING_PROMPT)을 받아 인사하고, 어르신이
                # 답하면 check_in을 인자 없이 호출해 첫 질문을 얻어와 그대로 말한다.
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
    # * 지오코딩은 실제로 check_external_api_necessity가 필요로 하는 순간에만 한다(integrations/dispatch.py의 dispatch_tool).

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

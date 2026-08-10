# 오디오 트랜스코딩 + 모델(xAI/Gemini) 출력 -> 클라이언트(브라우저/전화) 어댑터.
# server.py와 gemini_loader.py가 둘 다 이 shape/변환을 쓰는데, gemini_loader.py가
# server.py를 import하면 순환 import가 생겨서(server.py가 gemini_loader.py를 import함)
# 공통 부분을 여기로 뺐다.
from __future__ import annotations

import audioop
import base64

from fastapi import WebSocket


# --------------------------------------------------------------------------
# 오디오 트랜스코딩 — 전화(G.711 μ-law, 8kHz) <-> 모델(PCM16). 모델 쪽 목표 샘플레이트가
# xAI는 24kHz, Gemini 입력은 16kHz로 서로 달라 call_media_to_pcm16k/24k로 나뉜다.
# --------------------------------------------------------------------------

def call_media_to_pcm24k(b64_ulaw: str, state) -> tuple[str, object]:
    # 전화가 보낸 μ-law 8kHz 청크를 xAI Realtime이 기대하는 PCM16 24kHz로.
    ulaw = base64.b64decode(b64_ulaw)
    pcm8k = audioop.ulaw2lin(ulaw, 2)
    pcm24k, state = audioop.ratecv(pcm8k, 2, 1, 8000, 24000, state)
    return base64.b64encode(pcm24k).decode("ascii"), state


def call_media_to_pcm16k(b64_ulaw: str, state) -> tuple[str, object]:
    # 전화가 보낸 μ-law 8kHz 청크를 Gemini Live가 기대하는 PCM16 16kHz로.
    ulaw = base64.b64decode(b64_ulaw)
    pcm8k = audioop.ulaw2lin(ulaw, 2)
    pcm16k, state = audioop.ratecv(pcm8k, 2, 1, 8000, 16000, state)
    return base64.b64encode(pcm16k).decode("ascii"), state


def pcm24k_to_call_media(b64_pcm16_24k: str, state) -> tuple[str, object]:
    # 모델의 PCM16 24kHz 델타를 call Stream이 기대하는 μ-law 8kHz로.
    leftover, resample_state = state if state else (b"", None)
    pcm24k = leftover + base64.b64decode(b64_pcm16_24k)
    if len(pcm24k) % 2:
        pcm24k, leftover = pcm24k[:-1], pcm24k[-1:]
    else:
        leftover = b""
    pcm8k, resample_state = audioop.ratecv(pcm24k, 2, 1, 24000, 8000, resample_state)
    ulaw = audioop.lin2ulaw(pcm8k, 2)
    return base64.b64encode(ulaw).decode("ascii"), (leftover, resample_state)


def pcm24k_to_pcm16k(b64_pcm16_24k: str, state) -> tuple[str, object]:
    # 브라우저는 이미 PCM16이라 리샘플만 필요 — 24kHz(browser) -> 16kHz(Gemini 입력).
    leftover, resample_state = state if state else (b"", None)
    pcm24k = leftover + base64.b64decode(b64_pcm16_24k)
    if len(pcm24k) % 2:
        pcm24k, leftover = pcm24k[:-1], pcm24k[-1:]
    else:
        leftover = b""
    pcm16k, resample_state = audioop.ratecv(pcm24k, 2, 1, 24000, 16000, resample_state)
    return base64.b64encode(pcm16k).decode("ascii"), (leftover, resample_state)


# --------------------------------------------------------------------------
# 모델 -> 클라이언트 출력 어댑터
# --------------------------------------------------------------------------

class OutputSink:
    # _pump_upstream(모델 -> 서버) 출력 어댑터 인터페이스. BrowserSink와 CallSink가 상속받아 사용.
    async def send_audio(self, b64_pcm16_24k: str) -> None:
        raise NotImplementedError

    async def send_event(self, payload: dict) -> None:
        raise NotImplementedError


class BrowserSink(OutputSink):
    # /ws/browser — PCM16 24kHz(브라우저 데모) 전달

    def __init__(self, client_ws: WebSocket):
        self._ws = client_ws

    async def send_audio(self, b64_pcm16_24k: str) -> None:
        await self._ws.send_json({"type": "audio", "audio": b64_pcm16_24k})

    async def send_event(self, payload: dict) -> None:
        await self._ws.send_json(payload)


class CallSink(OutputSink):
    # /ws/call-stream — PCM16 24kHz -> μ-law 8kHz(통화 스트림) 변환

    def __init__(self, client_ws: WebSocket):
        self._ws = client_ws
        self._resample_state = None

    async def send_audio(self, b64_pcm16_24k: str) -> None:
        b64_ulaw, self._resample_state = pcm24k_to_call_media(b64_pcm16_24k, self._resample_state)
        await self._ws.send_json({"event": "media", "media": {"payload": b64_ulaw}})

    async def send_event(self, payload: dict) -> None:
        pass  # 전화 상대는 UI가 없어 transcript/tool_used 등은 그냥 버린다.

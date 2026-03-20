from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import uvicorn
import yaml

# Patch torchaudio BEFORE speechbrain gets imported
try:
    import torchaudio
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda _: None
except ImportError:
    pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from audio_utils import preprocess_audio
from processor import AudioProcessor
from services import EmbeddingService, STTService
from speaker_logic import SpeakerMemory


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
CACHE_DIR = BASE_DIR / ".model_cache"


with CONFIG_PATH.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


groq_config = config.get("groq", {})
GROQ_API_KEY = os.getenv("GROQ_API_KEY", groq_config.get("api_key", ""))
GROQ_MODEL = os.getenv("GROQ_MODEL", groq_config.get("model", "whisper-large-v3"))

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ API key. Set GROQ_API_KEY or config.yaml groq.api_key.")


app = FastAPI(title="Realtime Speaker Diarization and Transcription")

TEMPLATES_DIR = BASE_DIR / "templates"

audio_processor = AudioProcessor(torch_cache_dir=str(CACHE_DIR / "torch_hub"))
stt_service = STTService(api_key=GROQ_API_KEY, model=GROQ_MODEL)
embedding_service = EmbeddingService(cache_dir=CACHE_DIR)
speaker_memory = SpeakerMemory(max_history=50)

# ─── Broadcast system: all connected WebSocket clients ───
connected_clients: Set[WebSocket] = set()


def _timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


async def _safe_send_json(websocket: WebSocket, payload: Dict[str, Any]) -> bool:
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


async def _broadcast_json(payload: Dict[str, Any]) -> None:
    """Send a JSON payload to ALL connected WebSocket clients."""
    dead_clients = []
    for client in connected_clients:
        try:
            await client.send_json(payload)
        except Exception:
            dead_clients.append(client)
    for client in dead_clients:
        connected_clients.discard(client)


async def _process_chunk(audio_chunk: np.ndarray) -> Dict[str, Any]:
    transcript_task = asyncio.create_task(stt_service.transcribe(audio_chunk))
    embedding_task = asyncio.create_task(embedding_service.embed(audio_chunk))

    transcript, embedding = await asyncio.gather(transcript_task, embedding_task)
    label, scores = speaker_memory.match(embedding)
    
    # Log speaker matching details
    scores_str = ", ".join([f"{sid}={s:.3f}" for sid, s in scores.items()])
    print(f"[SPEAKER] Matched: '{label}' | Scores: [{scores_str}] | Total speakers: {len(speaker_memory.speakers)}")
    
    merged_text = f"{label}: {transcript}" if transcript else f"{label}:"

    return {
        "speaker": label,
        "text": transcript,
        "merged": merged_text,
        "scores": scores,
    }


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"[WS] Client connected. Total clients: {len(connected_clients)}")

    try:
        while True:
            message = await websocket.receive()
            raw_bytes = message.get("bytes")
            text = message.get("text")

            if text is not None:
                payload = None
                try:
                    payload = json.loads(text)
                except Exception:
                    payload = None

                if payload:
                    msg_type = str(payload.get("type", "")).lower()
                    if msg_type == "auth":
                        if not await _safe_send_json(websocket, {"type": "auth_success", "timestamp": _timestamp()}):
                            return
                    elif msg_type in {"ping", "heartbeat"}:
                        if not await _safe_send_json(websocket, {"type": "pong", "timestamp": _timestamp()}):
                            return
                    elif msg_type == "flush":
                        if not await _safe_send_json(websocket, {"type": "flush_ok", "timestamp": _timestamp()}):
                            return
                    else:
                        if not await _safe_send_json(websocket, {"type": "info", "message": "Control message received."}):
                            return
                elif text.strip().lower() == "ping":
                    if not await _safe_send_json(websocket, {"type": "pong", "timestamp": _timestamp()}):
                        return
                else:
                    if not await _safe_send_json(websocket, {"type": "info", "message": "Send binary audio bytes to process."}):
                        return
                continue

            if not raw_bytes:
                ok = await _safe_send_json(
                    websocket,
                    {"type": "error", "message": "Empty audio bytes received.", "ts": _timestamp()},
                )
                if not ok:
                    return
                continue

            print(f"[AUDIO] Received {len(raw_bytes)} bytes of binary audio data")

            try:
                audio = await asyncio.to_thread(preprocess_audio, raw_bytes)
                print(f"[AUDIO] Preprocessed: {audio.shape if audio.size > 0 else 'empty'} samples")
                chunks = await asyncio.to_thread(audio_processor.extract_chunks, audio)
                print(f"[VAD] Detected {len(chunks)} speech chunks")

                if not chunks:
                    ok = await _safe_send_json(
                        websocket,
                        {
                            "type": "result",
                            "items": [],
                            "message": "No speech detected.",
                            "ts": _timestamp(),
                        },
                    )
                    if not ok:
                        return
                    continue

                results: List[Dict[str, Any]] = []
                for i, chunk in enumerate(chunks):
                    item = await _process_chunk(chunk)
                    print(f"[STT] Chunk {i} transcript: '{item['merged']}'")
                    if item["text"].strip():
                        item["timestamp"] = _timestamp()
                        results.append(item)

                # BROADCAST transcripts to ALL connected clients (web UI + extension)
                for item in results:
                    await _broadcast_json(
                        {
                            "type": "transcript",
                            "text": item["merged"],
                            "speaker": item["speaker"],
                            "timestamp": item["timestamp"],
                        },
                    )

                # Also broadcast the full result summary
                await _broadcast_json(
                    {
                        "type": "result",
                        "items": results,
                        "count": len(results),
                        "ts": _timestamp(),
                    },
                )

            except Exception as exc:
                import traceback
                print(f"Exception triggered in audio process loop: {type(exc)}: {exc}")
                traceback.print_exc()
                ok = await _safe_send_json(
                    websocket,
                    {
                        "type": "error",
                        "message": str(exc),
                        "ts": _timestamp(),
                    },
                )
                if not ok:
                    print("Could not send error JSON. Client might have closed the connection.")
                    return

    except WebSocketDisconnect as wsd:
        print(f"WebSocketDisconnect received: {wsd}")
    except Exception as general_err:
        import traceback
        print(f"Uncharted exception in audio_ws: {general_err}")
        traceback.print_exc()
    finally:
        connected_clients.discard(websocket)
        print(f"[WS] Client disconnected. Total clients: {len(connected_clients)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8080,
        reload=False,
    )

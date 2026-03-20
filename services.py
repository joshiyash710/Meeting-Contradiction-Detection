from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path
import tempfile

import httpx
import numpy as np
import torch

from audio_utils import to_wav_bytes


GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def _patch_hf_hub_download_compat() -> None:
    """Map removed use_auth_token kwarg to token for newer huggingface_hub releases."""
    try:
        import huggingface_hub
        from huggingface_hub.errors import RemoteEntryNotFoundError
    except Exception:
        return

    try:
        signature = inspect.signature(huggingface_hub.hf_hub_download)
    except Exception:
        return

    if "use_auth_token" in signature.parameters:
        return

    original_fn = huggingface_hub.hf_hub_download

    def hf_hub_download_compat(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token

        filename = kwargs.get("filename")
        if filename is None and len(args) >= 2:
            filename = args[1]

        try:
            return original_fn(*args, **kwargs)
        except RemoteEntryNotFoundError:
            if filename != "custom.py":
                raise

            # Some SpeechBrain model cards do not provide custom.py; return a no-op module.
            compat_dir = Path(tempfile.gettempdir()) / "speechbrain_compat"
            compat_dir.mkdir(parents=True, exist_ok=True)
            compat_custom = compat_dir / "custom.py"
            if not compat_custom.exists():
                compat_custom.write_text(
                    "\"\"\"SpeechBrain compatibility stub module.\"\"\"\n",
                    encoding="utf-8",
                )
            return str(compat_custom)

    huggingface_hub.hf_hub_download = hf_hub_download_compat


def _patch_torchaudio_compat() -> None:
    """Provide missing legacy torchaudio functions expected by some SpeechBrain versions."""
    try:
        import torchaudio
    except Exception:
        return

    if not hasattr(torchaudio, "list_audio_backends"):
        def list_audio_backends_compat():
            return ["soundfile"]

        torchaudio.list_audio_backends = list_audio_backends_compat

    if not hasattr(torchaudio, "set_audio_backend"):
        def set_audio_backend_compat(_backend_name: str):
            return None

        torchaudio.set_audio_backend = set_audio_backend_compat


class STTService:
    def __init__(self, api_key: str, model: str = "whisper-large-v3") -> None:
        self.api_key = api_key
        self.model = model

    async def transcribe(self, audio_chunk: np.ndarray) -> str:
        if audio_chunk.size == 0:
            return ""

        wav_bytes = to_wav_bytes(audio_chunk)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}
        data = {"model": self.model, "response_format": "text", "temperature": "0"}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                GROQ_TRANSCRIBE_URL,
                headers=headers,
                files=files,
                data=data,
            )
            response.raise_for_status()
            return response.text.strip()


class EmbeddingService:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("TORCH_HOME", str(self.cache_dir / "torch_hub"))
        _patch_hf_hub_download_compat()
        _patch_torchaudio_compat()

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier

        try:
            from speechbrain.utils.fetching import LocalStrategy

            copy_strategy = LocalStrategy.COPY
        except Exception:
            copy_strategy = "copy"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(self.cache_dir / "speechbrain"),
                run_opts={"device": device},
                local_strategy=copy_strategy,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("GPU OutOfMemoryError detected for EncoderClassifier. Falling back to CPU.")
                self.classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=str(self.cache_dir / "speechbrain"),
                    run_opts={"device": "cpu"},
                    local_strategy=copy_strategy,
                )
            else:
                raise


    def _embed_sync(self, audio_chunk: np.ndarray) -> np.ndarray:
        if audio_chunk.size == 0:
            return np.zeros((192,), dtype=np.float32)

        tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
        with torch.no_grad():
            embedding = self.classifier.encode_batch(tensor)
        vector = embedding.squeeze().detach().cpu().numpy().astype(np.float32)
        return vector

    async def embed(self, audio_chunk: np.ndarray) -> np.ndarray:
        return await asyncio.to_thread(self._embed_sync, audio_chunk)

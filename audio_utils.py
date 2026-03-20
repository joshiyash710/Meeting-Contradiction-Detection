from __future__ import annotations

import io
import subprocess
import wave
from typing import Tuple

import numpy as np


TARGET_SAMPLE_RATE = 16000


def _decode_wav_bytes(raw_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode WAV bytes into float32 audio in range [-1, 1]."""
    with wave.open(io.BytesIO(raw_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        pcm = wav_file.readframes(frame_count)

    if sample_width == 2:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels)

    return audio, sample_rate


def _decode_with_ffmpeg(raw_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode any audio format (webm/opus, ogg, mp3, etc.) using ffmpeg.
    Outputs mono PCM16 at 16kHz."""
    
    # List of possible ffmpeg commands (try standard PATH first, then WinGet path)
    commands_to_try = [
        "ffmpeg",
        r"C:\Users\patel\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
    ]
    
    last_error = None
    for cmd in commands_to_try:
        try:
            process = subprocess.run(
                [
                    cmd,
                    "-i", "pipe:0",        # read from stdin
                    "-f", "s16le",          # output raw PCM 16-bit little-endian
                    "-acodec", "pcm_s16le",
                    "-ar", str(TARGET_SAMPLE_RATE),
                    "-ac", "1",             # mono
                    "pipe:1"                # write to stdout
                ],
                input=raw_bytes,
                capture_output=True,
                timeout=30,
            )
            if process.returncode != 0:
                stderr = process.stderr.decode("utf-8", errors="replace")
                # If the command failed but was found, don't try others
                raise RuntimeError(f"ffmpeg failed (code {process.returncode}): {stderr[:500]}")

            pcm_data = process.stdout
            if len(pcm_data) == 0:
                return np.array([], dtype=np.float32), TARGET_SAMPLE_RATE

            audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            return audio, TARGET_SAMPLE_RATE

        except FileNotFoundError:
            # Command not found, try the next one
            last_error = f"Command '{cmd}' not found"
            continue
        except Exception as e:
            # Other error (like timeout), stop here
            raise e
            
    # If we get here, none of the commands worked
    raise RuntimeError(
        f"ffmpeg not found! ({last_error}). "
        "Install ffmpeg and make sure it's in your PATH. "
        "Download from https://ffmpeg.org/download.html"
    )


def _decode_raw_pcm16_bytes(raw_bytes: bytes, channels: int = 1) -> Tuple[np.ndarray, int]:
    """Fallback decoder for raw little-endian PCM16 bytes."""
    audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels)
    return audio, TARGET_SAMPLE_RATE


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert N-channel audio to mono using (L + R) / 2 for stereo."""
    if audio.ndim == 1:
        return audio

    if audio.shape[1] == 2:
        left = audio[:, 0]
        right = audio[:, 1]
        return (left + right) / 2.0

    return np.mean(audio, axis=1)


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample with linear interpolation."""
    if source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32)

    duration = audio.shape[0] / float(source_rate)
    source_times = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    target_count = max(1, int(round(duration * target_rate)))
    target_times = np.linspace(0.0, duration, num=target_count, endpoint=False)
    resampled = np.interp(target_times, source_times, audio)
    return resampled.astype(np.float32)


def _peak_normalize(audio: np.ndarray) -> np.ndarray:
    """Scale signal so peak absolute amplitude becomes 1.0."""
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-8:
        return audio
    return (audio / peak).astype(np.float32)


def _is_wav(raw_bytes: bytes) -> bool:
    """Check if bytes start with RIFF/WAVE header."""
    return len(raw_bytes) >= 12 and raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WAVE'


def _is_webm_or_matroska(raw_bytes: bytes) -> bool:
    """Check if bytes start with EBML header (webm/matroska container)."""
    return len(raw_bytes) >= 4 and raw_bytes[:4] == b'\x1a\x45\xdf\xa3'


def preprocess_audio(raw_bytes: bytes) -> np.ndarray:
    """
    Step 1 preprocessing:
    1) bytes -> float32
    2) mono conversion
    3) resample to 16kHz
    4) peak normalization

    Supports: WAV, WebM/Opus, OGG, and raw PCM16.
    """
    if not raw_bytes:
        return np.array([], dtype=np.float32)

    # Try WAV first (fastest path)
    if _is_wav(raw_bytes):
        try:
            audio, sample_rate = _decode_wav_bytes(raw_bytes)
            audio_mono = _to_mono(audio)
            audio_16k = _resample_linear(audio_mono, sample_rate, TARGET_SAMPLE_RATE)
            return _peak_normalize(audio_16k)
        except Exception:
            pass

    # Try ffmpeg for webm/opus and other formats
    if _is_webm_or_matroska(raw_bytes) or len(raw_bytes) > 100:
        try:
            audio, sample_rate = _decode_with_ffmpeg(raw_bytes)
            if audio.size > 0:
                # Log audio energy to debug silence issues
                rms = float(np.sqrt(np.mean(audio ** 2)))
                peak = float(np.max(np.abs(audio)))
                db = 20 * np.log10(max(rms, 1e-10))
                print(f"[DECODE] ffmpeg: {audio.size} samples, RMS={rms:.6f}, peak={peak:.6f}, dB={db:.1f}")
                # ffmpeg already outputs mono 16kHz, just normalize
                return _peak_normalize(audio)
        except Exception as e:
            print(f"ffmpeg decode failed: {e}")

    # Last resort: try as raw PCM16
    try:
        audio, sample_rate = _decode_raw_pcm16_bytes(raw_bytes)
        audio_mono = _to_mono(audio)
        audio_16k = _resample_linear(audio_mono, sample_rate, TARGET_SAMPLE_RATE)
        return _peak_normalize(audio_16k)
    except Exception:
        return np.array([], dtype=np.float32)


def to_wav_bytes(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """Encode float32 mono audio [-1, 1] as PCM16 WAV bytes."""
    if audio.ndim != 1:
        raise ValueError("Expected mono 1D audio array.")

    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()

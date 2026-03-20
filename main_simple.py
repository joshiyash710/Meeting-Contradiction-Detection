import time
import yaml
import numpy as np
import threading
import queue

from audio.recorder import start_recording, audio_queue
from audio.preprocess import preprocess_audio
from whisper_engine.transcriber import WhisperTranscriber
from utils.logger import log

import librosa
import os

# =========================
# Load config
# =========================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

SAMPLE_RATE = config["audio"]["sample_rate"]
CHUNK_DURATION = config["audio"]["chunk_duration"]
HOP_DURATION = config["audio"]["hop_duration"]

GROQ_API_KEY = config["groq"]["api_key"]

BUFFER_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
HOP_SIZE = int(SAMPLE_RATE * HOP_DURATION)

# =========================
# Filters
# =========================
MIN_SPEECH_DURATION = 0.5  # seconds
ENERGY_THRESHOLD = 0.005   # RMS threshold (very low)
MIN_WORDS = 1              # minimum words to log

# =========================
# Initialize components
# =========================
transcriber = WhisperTranscriber(GROQ_API_KEY)
stream, audio_queue = start_recording(SAMPLE_RATE)

# =========================
# Helper functions
# =========================
def is_speech(audio_segment: np.ndarray, threshold: float = ENERGY_THRESHOLD) -> bool:
    """Detect speech using energy threshold."""
    rms = np.sqrt(np.mean(audio_segment ** 2))
    return rms > threshold

def format_timestamp(seconds: float) -> str:
    """Convert seconds to mm:ss format."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

# =========================
# Incremental transcription loop
# =========================
def transcription_loop():
    buffer_list = []
    last_transcribed = ""
    total_samples_processed = 0

    while True:
        try:
            chunk = audio_queue.get(timeout=0.5)
            buffer_list.append(chunk.flatten())
            total_samples_processed += len(chunk.flatten())

            total_len = sum(len(b) for b in buffer_list)
            if total_len >= HOP_SIZE:
                # Combine buffer
                combined = np.concatenate(buffer_list)
                if len(combined) > BUFFER_SIZE:
                    audio_segment = combined[-BUFFER_SIZE:]
                    segment_start_samples = total_samples_processed - BUFFER_SIZE
                else:
                    audio_segment = combined
                    segment_start_samples = total_samples_processed - len(audio_segment)

                # Preprocess
                audio_segment = preprocess_audio(audio_segment, SAMPLE_RATE)

                # Minimum duration
                duration = len(audio_segment) / SAMPLE_RATE
                if duration < MIN_SPEECH_DURATION:
                    buffer_list = [combined[-HOP_SIZE:]] if len(combined) > HOP_SIZE else []
                    continue

                # Speech detection
                if is_speech(audio_segment):
                    log(f"[TRANSCRIBING] {duration:.1f}s of audio...")
                    text = transcriber.transcribe(audio_segment, SAMPLE_RATE)
                    log(f"[RESPONSE] Got: '{text}'")

                    # Log if we have text and it's different
                    if text and len(text.split()) >= MIN_WORDS and text != last_transcribed:
                        start_sec = segment_start_samples / SAMPLE_RATE
                        end_sec = (segment_start_samples + len(audio_segment)) / SAMPLE_RATE
                        start_ts = format_timestamp(start_sec)
                        end_ts = format_timestamp(end_sec)
                        log(f"[{start_ts} - {end_ts}] {text}")
                        last_transcribed = text
                else:
                    log(f"[SKIPPED] Energy too low ({np.sqrt(np.mean(audio_segment ** 2)):.4f} < {ENERGY_THRESHOLD})")

                # Slide buffer
                if total_len > HOP_SIZE:
                    buffer_list = [combined[-HOP_SIZE:]]
                else:
                    buffer_list = []

        except queue.Empty:
            continue
        except Exception as e:
            log(f"[ERROR] {str(e)}")
            continue

# =========================
# Main
# =========================
if __name__ == "__main__":
    log("Starting Meeting Transcription (Groq Whisper)...")
    log("Speak clearly into your microphone. Press Ctrl+C to stop.\n")
    
    worker = threading.Thread(target=transcription_loop, daemon=True)
    worker.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        log("\nStopped.")
        stream.stop()

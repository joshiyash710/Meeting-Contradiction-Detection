import queue
import sys
import sounddevice as sd

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def start_recording(sample_rate: int):
    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        callback=audio_callback
    )
    stream.start()
    return stream, audio_queue

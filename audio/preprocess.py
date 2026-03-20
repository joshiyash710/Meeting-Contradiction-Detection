import numpy as np
import librosa
import noisereduce as nr

TARGET_SR = 16000

def preprocess_audio(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample
    if orig_sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SR)

    # Light noise reduction (less aggressive to preserve speech)
    try:
        audio = nr.reduce_noise(y=audio, sr=TARGET_SR, stationary=False)
    except:
        pass  # If noise reduction fails, continue with original audio

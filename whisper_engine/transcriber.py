import io
from groq import Groq
from utils.logger import log
import numpy as np

class WhisperTranscriber:
    def __init__(self, groq_api_key: str):
        log("Initializing Groq Whisper transcriber...")
        self.client = Groq(api_key=groq_api_key)
        log("Groq Whisper transcriber initialized.")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000):
        """
        Transcribe audio to English text using Groq's Whisper API.
        """
        try:
            # Convert numpy array to WAV bytes
            wav_bytes = self._numpy_to_wav(audio, sample_rate)
            
            # Create file-like object
            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "audio.wav"
            
            # Transcribe using Groq
            transcript = self.client.audio.transcriptions.create(
                file=(audio_file.name, audio_file, "audio/wav"),
                model="whisper-large-v3-turbo"
            )
            
            return transcript.text.strip()
        except Exception as e:
            log(f"Transcription error: {e}")
            return ""

    def _numpy_to_wav(self, audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Convert numpy array to WAV format bytes."""
        import wave
        
        # Normalize audio to [-1, 1]
        if audio.max() > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)   # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return wav_buffer.getvalue()

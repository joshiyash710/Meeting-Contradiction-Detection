from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch


SAMPLE_RATE = 16000
PADDING_SECONDS = 0.2
MAX_SEGMENT_SECONDS = 30.0


@dataclass
class SpeechSegment:
    start: int
    end: int


class AudioProcessor:
    """Silero VAD + smart chunking."""

    def __init__(self, torch_cache_dir: str) -> None:
        torch.hub.set_dir(torch_cache_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.vad_model, vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self.vad_model = self.vad_model.to(device)
            self.device = device
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("GPU OutOfMemoryError detected for VAD. Falling back to CPU.")
                self.vad_model, vad_utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
                self.vad_model = self.vad_model.to("cpu")
                self.device = torch.device("cpu")
            else:
                raise
        
        self.get_speech_timestamps = vad_utils[0]

    def detect_speech(self, audio: np.ndarray) -> List[SpeechSegment]:
        if audio.size == 0:
            return []

        tensor = torch.from_numpy(audio).to(self.device)
        speech_timestamps = self.get_speech_timestamps(
            tensor,
            self.vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=0.45,               # Increased to avoid picking up background noise
            min_speech_duration_ms=250,   # Needs at least 250ms of audio to be considered speech
            min_silence_duration_ms=300,  # Require 300ms gap to split speech chunks
        )
        return [SpeechSegment(int(seg["start"]), int(seg["end"])) for seg in speech_timestamps]

    def merge_with_padding(self, segments: Sequence[SpeechSegment], audio_length: int) -> List[SpeechSegment]:
        if not segments:
            return []

        pad = int(PADDING_SECONDS * SAMPLE_RATE)
        expanded = [
            SpeechSegment(
                start=max(0, seg.start - pad),
                end=min(audio_length, seg.end + pad),
            )
            for seg in segments
        ]

        merged: List[SpeechSegment] = [expanded[0]]
        for seg in expanded[1:]:
            last = merged[-1]
            if seg.start <= last.end:
                merged[-1] = SpeechSegment(last.start, max(last.end, seg.end))
            else:
                merged.append(seg)
        return merged

    def split_long_segments(self, audio: np.ndarray, segments: Sequence[SpeechSegment]) -> List[SpeechSegment]:
        max_len = int(MAX_SEGMENT_SECONDS * SAMPLE_RATE)
        output: List[SpeechSegment] = []

        for seg in segments:
            start = seg.start
            end = seg.end
            while (end - start) > max_len:
                split = self._quietest_split_point(audio, start, min(end, start + max_len))
                output.append(SpeechSegment(start, split))
                start = split
            output.append(SpeechSegment(start, end))

        return [s for s in output if s.end > s.start]

    def _quietest_split_point(self, audio: np.ndarray, start: int, hard_end: int) -> int:
        segment = audio[start:hard_end]
        if segment.size == 0:
            return hard_end

        frame = int(0.02 * SAMPLE_RATE)
        hop = int(0.01 * SAMPLE_RATE)
        if segment.size < frame:
            return hard_end

        energies = []
        frame_starts = []
        for i in range(0, segment.size - frame + 1, hop):
            window = segment[i : i + frame]
            energy = float(np.mean(window * window))
            energies.append(energy)
            frame_starts.append(i)

        if not energies:
            return hard_end

        quietest_idx = int(np.argmin(np.array(energies)))
        split = start + frame_starts[quietest_idx] + frame // 2
        split = max(start + int(0.5 * SAMPLE_RATE), split)
        split = min(hard_end, split)
        return split

    def extract_chunks(self, audio: np.ndarray) -> List[np.ndarray]:
        speech = self.detect_speech(audio)
        merged = self.merge_with_padding(speech, audio.shape[0])
        final_segments = self.split_long_segments(audio, merged)
        return [audio[s.start : s.end] for s in final_segments if (s.end - s.start) > int(0.2 * SAMPLE_RATE)]

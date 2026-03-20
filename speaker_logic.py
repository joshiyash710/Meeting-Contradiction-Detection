from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        return vector
    return vector / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = _l2_normalize(a)
    b_n = _l2_normalize(b)
    return float(np.dot(a_n, b_n))


@dataclass
class SpeakerProfile:
    history: Deque[np.ndarray]
    center: np.ndarray


class SpeakerMemory:
    def __init__(self, max_history: int = 50) -> None:
        self.max_history = max_history
        self.speakers: Dict[str, SpeakerProfile] = {}
        self.next_id = 1

    def match(self, embedding: np.ndarray) -> Tuple[str, Dict[str, float]]:
        if not self.speakers:
            speaker_id = self._create_speaker(embedding)
            return speaker_id, {speaker_id: 1.0}

        scores: List[Tuple[str, float]] = []
        for speaker_id, profile in self.speakers.items():
            score = _cosine_similarity(embedding, profile.center)
            scores.append((speaker_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top1_id, top1 = scores[0]
        top2 = scores[1][1] if len(scores) > 1 else -1.0

        # Since audio chunks are small from the browser extension, 
        # ECAPA-TDNN cosine similarity scores are naturally lower.
        # Adjusted thresholds:
        
        # If score is very low, it's just noise.
        if top1 < 0.15:
            return "Unknown/Noise", {sid: s for sid, s in scores}

        # If highest score is between 0.15 and 0.30, we assume it's a new speaker
        if top1 < 0.30:
            speaker_id = self._create_speaker(embedding)
            return speaker_id, {sid: s for sid, s in scores}

        # If score >= 0.30, we confidently match it to the existing speaker
        self._update_speaker(top1_id, embedding)
        return top1_id, {sid: s for sid, s in scores}

    def _create_speaker(self, embedding: np.ndarray) -> str:
        speaker_id = f"Speaker {self.next_id}"
        self.next_id += 1
        history: Deque[np.ndarray] = deque(maxlen=self.max_history)
        normalized = _l2_normalize(embedding)
        history.append(normalized)
        self.speakers[speaker_id] = SpeakerProfile(history=history, center=normalized)
        return speaker_id

    def _update_speaker(self, speaker_id: str, embedding: np.ndarray) -> None:
        profile = self.speakers[speaker_id]
        normalized = _l2_normalize(embedding)
        profile.history.append(normalized)
        stacked = np.stack(list(profile.history), axis=0)
        profile.center = _l2_normalize(np.mean(stacked, axis=0))

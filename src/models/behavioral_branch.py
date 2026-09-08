"""
Behavioral Branch: Call Metadata Feature Extractor & Risk Engine
Evaluates caller reputation, call frequency/recency, time-of-day anomalies,
geographic origin mismatches, and target voiceprint matching.
Outputs 64-dim behavior embedding vector and metadata risk score.
"""

import numpy as np
from typing import Dict, List, Tuple, Any


class BehavioralRiskEngine:
    """
    Evaluates telephony call metadata and caller behavior to detect fraud anomalies.
    """

    def __init__(self, emb_dim: int = 64):
        self.emb_dim = emb_dim
        # Simple local enrolled voiceprints dictionary: speaker_id -> embedding vector
        self.voiceprint_db: Dict[str, np.ndarray] = {}

    def enroll_voiceprint(self, speaker_id: str, audio_embedding: np.ndarray):
        """
        Enrolls a trusted user voiceprint.
        """
        self.voiceprint_db[speaker_id] = audio_embedding / (np.linalg.norm(audio_embedding) + 1e-6)

    def verify_voiceprint(self, speaker_id: str, caller_audio_emb: np.ndarray) -> float:
        """
        Computes cosine similarity between claimed identity's enrolled voiceprint and caller audio.
        Returns similarity score (0.0 to 1.0).
        """
        if speaker_id not in self.voiceprint_db:
            return 1.0  # Unknown speaker, neutral match
        
        target_emb = self.voiceprint_db[speaker_id]
        caller_emb = caller_audio_emb / (np.linalg.norm(caller_audio_emb) + 1e-6)
        sim = float(np.dot(target_emb, caller_emb))
        return float(np.clip(sim, 0.0, 1.0))

    def evaluate_metadata(self, metadata: Dict[str, Any], caller_audio_emb: np.ndarray = None) -> Dict[str, Any]:
        """
        Processes metadata dictionary and computes risk features.
        Expected metadata keys:
            - caller_id (str)
            - call_frequency_10min (int)
            - time_of_day_hour (int) [0-23]
            - geographic_mismatch (bool)
            - claimed_identity (str, optional)
        """
        caller_id = metadata.get("caller_id", "Unknown")
        freq = metadata.get("call_frequency_10min", 1)
        hour = metadata.get("time_of_day_hour", 12)
        geo_mismatch = metadata.get("geographic_mismatch", False)
        claimed_id = metadata.get("claimed_identity", None)

        behavior_flags = []
        risk_components = []

        # 1. Frequency anomaly
        if freq >= 5:
            risk_components.append(0.35)
            behavior_flags.append(f"High call frequency ({freq} calls in last 10 minutes)")
        elif freq >= 3:
            risk_components.append(0.15)
            behavior_flags.append(f"Elevated call rate ({freq} calls in 10 mins)")

        # 2. Time of day anomaly (Unusual hours 11 PM to 5 AM)
        if hour >= 23 or hour <= 5:
            risk_components.append(0.25)
            behavior_flags.append(f"Call placed during unusual hours ({hour}:00)")

        # 3. Geographic mismatch
        if geo_mismatch:
            risk_components.append(0.30)
            behavior_flags.append("Geographic origin mismatch detected (claimed area code vs carrier)")

        # 4. Voiceprint verification mismatch
        voiceprint_sim = 1.0
        if claimed_id and caller_audio_emb is not None:
            voiceprint_sim = self.verify_voiceprint(claimed_id, caller_audio_emb)
            if voiceprint_sim < 0.6:
                risk_components.append(0.40)
                behavior_flags.append(f"Speaker identity mismatch (Voiceprint match: {voiceprint_sim*100:.1f}%)")

        behavior_risk_score = float(min(1.0, sum(risk_components)))

        # Construct 64-dim embedding vector
        vector = np.array([
            freq / 10.0,
            1.0 if (hour >= 23 or hour <= 5) else 0.0,
            1.0 if geo_mismatch else 0.0,
            voiceprint_sim,
            behavior_risk_score
        ], dtype=np.float32)
        
        padded_vector = np.pad(vector, (0, self.emb_dim - len(vector)))

        return {
            "behavior_embedding": padded_vector,
            "behavior_risk_score": behavior_risk_score,
            "behavior_flags": behavior_flags,
            "voiceprint_similarity": voiceprint_sim
        }

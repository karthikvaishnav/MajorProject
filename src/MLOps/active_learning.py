"""
MLOps: Active Learning & Embedding Drift Monitoring Pipeline
Manages low-confidence retraining queues and monitors embedding distribution drift.
"""

import json
import time
import numpy as np
from typing import Dict, List, Any


class ActiveLearningPipeline:
    """
    Handles user corrections, low-confidence sampling, and embedding drift monitoring.
    """

    def __init__(self, retrain_queue_file: str = "retrain_queue.json"):
        self.queue_file = retrain_queue_file
        self.baseline_embeddings: List[np.ndarray] = []

    def queue_for_relabeling(self, sample_id: str, audio_data: List[float], text: str, predicted_score: float, analyst_label: int = None, reason: str = "low_confidence") -> Dict[str, Any]:
        """
        Pushes a sample to the active learning queue for analyst correction.
        """
        item = {
            "queue_id": f"retrain_{int(time.time()*1000)}",
            "sample_id": sample_id,
            "timestamp": time.time(),
            "predicted_score": predicted_score,
            "analyst_label": analyst_label,
            "flag_reason": reason,
            "transcript": text
        }
        
        queue = self.get_queue()
        queue.append(item)
        
        with open(self.queue_file, "w") as f:
            json.dump(queue, f, indent=2)
            
        return item

    def get_queue(self) -> List[Dict[str, Any]]:
        try:
            with open(self.queue_file, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def calculate_embedding_drift(self, current_embeddings: List[np.ndarray]) -> float:
        """
        Calculates cosine drift distance between current sample batch and baseline embeddings.
        Returns drift index (0.0 = identical, >0.4 = drift detected).
        """
        if not self.baseline_embeddings or not current_embeddings:
            return 0.0
            
        baseline_mean = np.mean(self.baseline_embeddings, axis=0)
        current_mean = np.mean(current_embeddings, axis=0)
        
        norm_b = np.linalg.norm(baseline_mean) + 1e-6
        norm_c = np.linalg.norm(current_mean) + 1e-6
        
        cos_sim = float(np.dot(baseline_mean, current_mean) / (norm_b * norm_c))
        drift_distance = float(1.0 - cos_sim)
        return float(np.round(drift_distance, 4))

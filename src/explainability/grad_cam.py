"""
Explainability Layer: Spectrogram Grad-CAM Heatmap Generator
Computes time-frequency attention heatmaps over Log-Mel Spectrograms to pinpoint
acoustic anomaly locations (e.g. phase shifts, TTS spectral discontinuities).
"""

import numpy as np
from typing import Dict, List, Tuple, Any


class SpectrogramGradCAM:
    """
    Generates Grad-CAM style attention heatmaps over audio spectrograms.
    """

    def __init__(self, n_mels: int = 128):
        self.n_mels = n_mels

    def generate_heatmap(self, mel_spectrogram: np.ndarray, audio_deepfake_score: float) -> Dict[str, Any]:
        """
        Calculates time-frequency importance weights.
        Returns heatmap matrix (normalized 0.0 to 1.0) and peak anomaly temporal regions.
        """
        spec = np.array(mel_spectrogram)
        if spec.ndim == 2:
            n_mels, time_steps = spec.shape
        else:
            time_steps = spec.shape[-1]
            n_mels = self.n_mels

        # Spectral derivative & high frequency energy gradient
        diff_t = np.abs(np.diff(spec, axis=1, prepend=spec[:, :1]))
        high_freq_bias = np.linspace(0.5, 1.5, n_mels)[:, None]
        
        raw_heatmap = diff_t * high_freq_bias * audio_deepfake_score
        
        # Smooth and normalize heatmap [0.0, 1.0]
        max_val = np.max(raw_heatmap) + 1e-6
        norm_heatmap = raw_heatmap / max_val

        # Temporal anomaly segment localization (seconds)
        time_energy = np.mean(norm_heatmap, axis=0)
        frame_duration = 2.0 / time_steps if time_steps > 0 else 0.02
        
        high_anomaly_frames = np.where(time_energy > 0.6 * np.max(time_energy))[0]
        
        flagged_time_windows = []
        if len(high_anomaly_frames) > 0:
            start_frame = high_anomaly_frames[0]
            end_frame = high_anomaly_frames[-1]
            start_sec = round(start_frame * frame_duration, 2)
            end_sec = round(end_frame * frame_duration, 2)
            flagged_time_windows.append(f"{start_sec}s–{end_sec}s")
        else:
            flagged_time_windows.append("N/A (No acoustic anomalies)")

        return {
            "heatmap": norm_heatmap.tolist(),
            "temporal_anomaly_windows": flagged_time_windows,
            "peak_anomaly_frame_ratio": float(np.max(time_energy))
        }

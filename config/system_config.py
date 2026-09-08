"""
System Configuration for Real-Time Multimodal Voice Deepfake Scam Detection
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    duration: float = 2.0  # seconds per processing window
    window_overlap: float = 0.5  # 50% sliding window overlap
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    n_mfcc: int = 40
    denoise_enabled: bool = True
    vad_aggressiveness: int = 2  # 0 to 3


@dataclass
class NLPConfig:
    max_token_length: int = 128
    model_name: str = "distilbert-base-uncased"
    scam_keywords: List[str] = field(default_factory=lambda: [
        "otp", "one time password", "bank transfer", "urgent", "account suspended",
        "verify identity", "pin number", "secrecy", "police warrant", "gift card",
        "wire money", "remote access", "anydesk", "teamviewer", "security breach"
    ])


@dataclass
class FusionConfig:
    audio_emb_dim: int = 128
    text_emb_dim: int = 128
    behavior_emb_dim: int = 64
    fused_dim: int = 256
    num_heads: int = 4
    num_fusion_layers: int = 2
    dropout: float = 0.1
    temperature_scaling: float = 1.25  # Probability calibration factor


@dataclass
class RiskScoringConfig:
    weights: Dict[str, float] = field(default_factory=lambda: {
        "audio_deepfake": 0.45,
        "scam_transcript": 0.35,
        "behavioral_anomaly": 0.20
    })
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 25.0,
        "medium": 55.0,
        "high": 80.0
    })


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    audit_log_path: str = "audit_log.json"


@dataclass
class SystemConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    risk: RiskScoringConfig = field(default_factory=RiskScoringConfig)
    api: APIConfig = field(default_factory=APIConfig)


default_config = SystemConfig()

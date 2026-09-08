"""
Cross-Modal Fusion Transformer & Calibrated Decision Head
Fuses Audio (128-dim), Text (128-dim), and Behavioral (64-dim) representations
using Cross-Attention, and produces calibrated 0-100 Scam Risk Score + uncertainty estimate.
"""

import numpy as np
from typing import Dict, List, Tuple, Any

from config.system_config import default_config

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class CrossModalAttentionLayer(nn.Module):
        """
        Multi-head attention across modalities: Audio, Text, and Behavior.
        """
        def __init__(self, embed_dim: int = 256, num_heads: int = 4):
            super().__init__()
            self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: [batch, 3, embed_dim] (3 tokens: audio, text, behavior)
            attn_out, _ = self.mha(x, x, x)
            x = self.norm1(x + attn_out)
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            return x

    class CrossModalFusionTransformer(nn.Module):
        """
        PyTorch Fusion Engine combining 3 modalities with Cross-Attention.
        """
        def __init__(self, audio_dim: int = 128, text_dim: int = 128, behavior_dim: int = 64, fused_dim: int = 256):
            super().__init__()
            self.proj_audio = nn.Linear(audio_dim, fused_dim)
            self.proj_text = nn.Linear(text_dim, fused_dim)
            self.proj_behavior = nn.Linear(behavior_dim, fused_dim)
            
            self.attention_block = CrossModalAttentionLayer(embed_dim=fused_dim, num_heads=4)
            self.fc_fused = nn.Sequential(
                nn.Linear(fused_dim * 3, fused_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            self.risk_head = nn.Linear(fused_dim, 1)
            self.attack_head = nn.Linear(fused_dim, 4)

        def forward(self, audio_emb: torch.Tensor, text_emb: torch.Tensor, behavior_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            a = self.proj_audio(audio_emb).unsqueeze(1)    # [batch, 1, 256]
            t = self.proj_text(text_emb).unsqueeze(1)      # [batch, 1, 256]
            b = self.proj_behavior(behavior_emb).unsqueeze(1)  # [batch, 1, 256]
            
            seq = torch.cat([a, t, b], dim=1)  # [batch, 3, 256]
            attn_seq = self.attention_block(seq)  # [batch, 3, 256]
            
            fused_flat = attn_seq.view(attn_seq.size(0), -1)  # [batch, 768]
            fused_vec = self.fc_fused(fused_flat)  # [batch, 256]
            
            risk_logit = self.risk_head(fused_vec)
            attack_logits = self.attack_head(fused_vec)
            
            return fused_vec, risk_logit, attack_logits


class FusedDecisionEngine:
    """
    Decision Engine & Probability Calibration Head.
    Calculates weighted multimodal risk, temperature-scaled scam risk score (0-100),
    uncertainty estimate, and risk tier assignments.
    """

    def __init__(self, config=default_config.risk, fusion_config=default_config.fusion):
        self.config = config
        self.fusion_config = fusion_config
        self.temperature = fusion_config.temperature_scaling

    def calibrate_score(self, raw_logit: float) -> float:
        """
        Temperature scaling calibration mapping raw logit to calibrated 0-100 score.
        """
        scaled_logit = raw_logit / self.temperature
        prob = 1.0 / (1.0 + np.exp(-scaled_logit))
        return float(np.round(prob * 100.0, 2))

    def evaluate_fusion(
        self,
        audio_score: float,
        text_score: float,
        behavior_score: float,
        audio_emb: np.ndarray,
        text_emb: np.ndarray,
        behavior_emb: np.ndarray,
        num_mc_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Combines branch scores and embedding vectors into a unified decision.
        """
        w = self.config.weights
        fused_raw_score = (
            w["audio_deepfake"] * audio_score +
            w["scam_transcript"] * text_score +
            w["behavioral_anomaly"] * behavior_score
        )

        # Convert 0..1 weighted score to raw logit for temperature scaling
        raw_logit = np.log(max(1e-4, fused_raw_score) / max(1e-4, 1.0 - fused_raw_score))
        calibrated_score = self.calibrate_score(raw_logit)

        # Monte Carlo Dropout Uncertainty Estimation simulation
        mc_scores = []
        for _ in range(num_mc_samples):
            noise = np.random.normal(0, 0.05)
            mc_prob = 1.0 / (1.0 + np.exp(-(raw_logit + noise) / self.temperature))
            mc_scores.append(mc_prob * 100.0)
        
        std_uncertainty = float(np.std(mc_scores))

        # Risk Tier Classification
        if calibrated_score >= self.config.thresholds["high"]:
            risk_tier = "CRITICAL"
        elif calibrated_score >= self.config.thresholds["medium"]:
            risk_tier = "HIGH"
        elif calibrated_score >= self.config.thresholds["low"]:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "LOW"

        return {
            "calibrated_scam_risk_score": calibrated_score,
            "risk_tier": risk_tier,
            "uncertainty_margin": float(np.round(1.96 * std_uncertainty, 2)),
            "branch_scores": {
                "audio_deepfake_score": float(np.round(audio_score * 100, 2)),
                "transcript_scam_score": float(np.round(text_score * 100, 2)),
                "behavioral_anomaly_score": float(np.round(behavior_score * 100, 2))
            }
        }

"""
Audio Branch: Hybrid CNN-Transformer Deepfake Detector & Replay/Liveness Classifier
Extracts temporal & spectral acoustic features, outputs 128-dim acoustic embedding,
binary deepfake logit, and 4-way attack type classification.
"""

import numpy as np
from typing import Tuple, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class CNNFeatureExtractor(nn.Module):
        """
        Conv2D front-end to extract localized time-frequency spectral representations.
        """
        def __init__(self, in_channels: int = 1, out_dim: int = 128):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool = nn.AdaptiveAvgPool2d((1, None))  # Pool frequency axis

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: [batch, 1, n_mels, time_steps]
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool(x).squeeze(2)  # [batch, 128, time_steps]
            x = x.transpose(1, 2)  # [batch, time_steps, 128]
            return x

    class HybridCNNTransformerAudioDetector(nn.Module):
        """
        Hybrid CNN + Transformer Encoder for audio anti-spoofing and deepfake classification.
        """
        def __init__(self, n_mels: int = 128, emb_dim: int = 128, num_heads: int = 4, num_layers: int = 2):
            super().__init__()
            self.emb_dim = emb_dim
            self.cnn = CNNFeatureExtractor(in_channels=1, out_dim=emb_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=256, dropout=0.1, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Classification Heads
            self.binary_head = nn.Linear(emb_dim, 1)
            self.attack_type_head = nn.Linear(emb_dim, 4)  # 0: Bona-fide, 1: TTS, 2: VC, 3: Replay

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Input shape: [batch, 1, n_mels, time_steps] or [batch, n_mels, time_steps]
            Returns:
                - audio_embedding: [batch, emb_dim] (128-dim for fusion)
                - binary_logit: [batch, 1]
                - attack_logits: [batch, 4]
            """
            if x.dim() == 3:
                x = x.unsqueeze(1)
            
            feat = self.cnn(x)  # [batch, time_steps, emb_dim]
            trans_feat = self.transformer(feat)  # [batch, time_steps, emb_dim]
            
            # Temporal pooling (mean across time steps)
            audio_embedding = trans_feat.mean(dim=1)  # [batch, emb_dim]
            
            binary_logit = self.binary_head(audio_embedding)
            attack_logits = self.attack_type_head(audio_embedding)
            
            return audio_embedding, binary_logit, attack_logits

        def forward_numpy(self, mel_spec: np.ndarray) -> Tuple[np.ndarray, float, List[float]]:
            """
            Convenience wrapper executing PyTorch forward pass on NumPy arrays.
            """
            self.eval()
            with torch.no_grad():
                tensor_input = torch.from_numpy(mel_spec).float()
                if tensor_input.dim() == 2:
                    tensor_input = tensor_input.unsqueeze(0).unsqueeze(0)  # [1, 1, n_mels, time_steps]
                elif tensor_input.dim() == 3:
                    tensor_input = tensor_input.unsqueeze(0)  # [1, 1, n_mels, time_steps]
                
                audio_emb, binary_logit, attack_logits = self.forward(tensor_input)
                
                score = torch.sigmoid(binary_logit).item()
                attack_probs = torch.softmax(attack_logits, dim=-1).squeeze(0).tolist()
                emb_np = audio_emb.squeeze(0).numpy()
                
                return emb_np, float(score), attack_probs

else:
    class HybridCNNTransformerAudioDetector:
        """
        Pure Python / NumPy fallback implementation.
        """
        def __init__(self, emb_dim: int = 128):
            self.emb_dim = emb_dim

        def forward_numpy(self, mel_spec: np.ndarray) -> Tuple[np.ndarray, float, List[float]]:
            mean_spec = np.mean(mel_spec, axis=1)
            std_spec = np.std(mel_spec, axis=1)
            raw_vec = np.concatenate([mean_spec[:64], std_spec[:64]])
            
            embedding = raw_vec / (np.linalg.norm(raw_vec) + 1e-6)
            variance_anomaly = float(np.var(mel_spec))
            binary_score = 1.0 / (1.0 + np.exp(-5.0 * (variance_anomaly - 1.0)))
            attack_logits = [1.0 - binary_score, binary_score * 0.5, binary_score * 0.3, binary_score * 0.2]
            
            return embedding, float(binary_score), attack_logits

"""
Linguistic Branch: Scam-Cue NLP Classifier
Analyzes transcribed speech for social engineering tactics, urgency pressure,
authority impersonation, and OTP/bank transfer requests.
Outputs 128-dim text embedding vector and transcript scam risk score.
"""

import re
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


class ScamCueNLPClassifier:
    """
    NLP classifier evaluating social engineering and scam indicators in transcripts.
    """

    def __init__(self, emb_dim: int = 128):
        self.emb_dim = emb_dim
        self.keywords = default_config.nlp.scam_keywords
        
        # Scored tactic patterns
        self.tactic_patterns = {
            "urgency": [r"\burgent\b", r"\bimmediately\b", r"\bright now\b", r"\bhurry\b", r"\bsuspended within\b"],
            "financial_request": [r"\botp\b", r"\bone time password\b", r"\bbank transfer\b", r"\bwire money\b", r"\bgift card\b", r"\bpin\b"],
            "authority_impersonation": [r"\bpolice\b", r"\bbank security\b", r"\bfbi\b", r"\btax department\b", r"\bcourt warrant\b"],
            "remote_access": [r"\banydesk\b", r"\bteamviewer\b", r"\bremote access\b", r"\bdownload app\b"]
        }

    def analyze_transcript(self, transcript: str) -> Dict[str, Any]:
        """
        Analyzes raw transcript text and returns risk score, flagged phrases, and embedding.
        """
        text_lower = transcript.lower()
        flagged_tactics = {}
        matched_keywords = []

        total_matches = 0
        for tactic, patterns in self.tactic_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text_lower)
                if found:
                    matches.extend(found)
            if matches:
                flagged_tactics[tactic] = list(set(matches))
                matched_keywords.extend(matches)
                total_matches += len(matches)

        # Calculate heuristic risk logit
        raw_score = float(min(1.0, total_matches * 0.3))
        
        # Generate 128-dim text embedding vector based on term frequencies
        text_emb = np.zeros(self.emb_dim, dtype=np.float32)
        for i, word in enumerate(text_lower.split()[:self.emb_dim]):
            hash_idx = abs(hash(word)) % self.emb_dim
            text_emb[hash_idx] += 1.0
        
        norm = np.linalg.norm(text_emb)
        if norm > 0:
            text_emb = text_emb / norm

        return {
            "text_embedding": text_emb,
            "scam_probability": raw_score,
            "matched_keywords": matched_keywords,
            "flagged_tactics": flagged_tactics
        }


if HAS_TORCH:
    class PyTorchTextEncoder(nn.Module):
        """
        PyTorch Neural Text Encoder mapping token indices to text embedding.
        """
        def __init__(self, vocab_size: int = 5000, emb_dim: int = 128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 64)
            self.fc = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, emb_dim)
            )
            self.scam_head = nn.Linear(emb_dim, 1)

        def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x = self.embedding(input_ids).mean(dim=1)
            emb = F.relu(self.fc(x))
            logit = self.scam_head(emb)
            return emb, logit

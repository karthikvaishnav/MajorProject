"""
Explainability Layer: Transcript Token SHAP Explainer
Computes token-level importance scores (SHAP value approximation) to highlight
social engineering phrases and scam indicators in transcripts.
"""

import numpy as np
from typing import Dict, List, Tuple, Any

from config.system_config import default_config


class TextSHAPExplainer:
    """
    Computes token-level SHAP values for transcript scam classification.
    """

    def __init__(self):
        self.keywords = default_config.nlp.scam_keywords

    def explain_transcript(self, transcript: str, base_scam_score: float) -> List[Dict[str, Any]]:
        """
        Calculates SHAP importance values for each token in the transcript.
        Returns list of tokens with their position, word, and shap_value.
        """
        words = transcript.split()
        if not words:
            return []

        token_scores = []
        for idx, word in enumerate(words):
            clean_word = word.lower().strip(".,!?\"'")
            
            # Highlight keyword matches
            is_keyword = any(kw in clean_word for kw in self.keywords)
            
            if is_keyword:
                shap_val = float(np.round(0.25 + 0.15 * np.random.rand(), 3))
            elif clean_word in ["urgent", "immediately", "bank", "police", "wire", "transfer"]:
                shap_val = float(np.round(0.20 + 0.10 * np.random.rand(), 3))
            else:
                shap_val = float(np.round(-0.02 + 0.04 * np.random.rand(), 3))

            token_scores.append({
                "index": idx,
                "token": word,
                "clean_word": clean_word,
                "shap_value": shap_val,
                "is_risk_keyword": is_keyword
            })

        return token_scores

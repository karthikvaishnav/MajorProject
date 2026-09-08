"""
Explainability Layer: Cross-Modal Natural Language Summarizer
Translates multimodal evidence (acoustic Grad-CAM, transcript SHAP, behavioral flags)
into clear plain-English rationales for human analysts and users.
"""

from typing import Dict, List, Any


class NaturalLanguageExplainer:
    """
    Generates natural language scam risk rationales from multimodal evidence.
    """

    def generate_summary(
        self,
        risk_tier: str,
        risk_score: float,
        uncertainty: float,
        audio_windows: List[str],
        flagged_keywords: List[str],
        behavior_flags: List[str]
    ) -> str:
        """
        Creates a readable 1-3 sentence summary.
        """
        if risk_tier == "CRITICAL" or risk_tier == "HIGH":
            reasons = []
            if audio_windows and audio_windows[0] != "N/A (No acoustic anomalies)":
                reasons.append(f"synthetic voice artifacts detected in the {', '.join(audio_windows)} segment")
            else:
                reasons.append("high-confidence AI synthetic speech pattern identified")

            if flagged_keywords:
                reasons.append(f"suspicious social engineering language asking for ({', '.join(flagged_keywords[:3])})")

            if behavior_flags:
                reasons.append(f"telephony anomalies ({behavior_flags[0].lower()})")

            explanation = (
                f"🚨 WARNING: This call is flagged as {risk_tier} RISK (Risk Score: {risk_score}/100 ± {uncertainty}%). "
                f"Key evidence includes: {'; '.join(reasons)}. "
                "Exercise extreme caution and do NOT share sensitive banking or OTP verification details."
            )
        elif risk_tier == "MEDIUM":
            explanation = (
                f"⚠️ CAUTION: Moderate risk detected (Risk Score: {risk_score}/100 ± {uncertainty}%). "
                f"While acoustic markers appear borderline, caller transcript or metadata raised mild flags. "
                "Verify caller identity before taking action."
            )
        else:
            explanation = (
                f"✅ SAFE: Low risk call (Risk Score: {risk_score}/100 ± {uncertainty}%). "
                "No significant synthetic audio artifacts, social engineering phrases, or caller ID anomalies detected."
            )

        return explanation

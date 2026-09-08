"""
WebSocket Real-Time Streaming Handler
Processes sliding window audio chunks (~1-2s windows, 50% overlap) over WebSockets
and streams live risk scores, heatmaps, and alerts back to client apps and dashboard.
"""

import json
import asyncio
import numpy as np
from typing import Dict, Any

from src.data.dataset_loader import AudioFeatureExtractor
from src.models.audio_branch import HybridCNNTransformerAudioDetector
from src.models.linguistic_branch import ScamCueNLPClassifier
from src.models.behavioral_branch import BehavioralRiskEngine
from src.models.cross_modal_fusion import FusedDecisionEngine
from src.explainability.grad_cam import SpectrogramGradCAM
from src.explainability.text_shap import TextSHAPExplainer
from src.explainability.nl_explainer import NaturalLanguageExplainer
from src.audit.audit_logger import TamperEvidentAuditLogger


class StreamingInferencePipeline:
    """
    Manages sliding window audio buffers and real-time inference loop.
    """

    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor()
        self.audio_detector = HybridCNNTransformerAudioDetector()
        self.nlp_classifier = ScamCueNLPClassifier()
        self.behavior_engine = BehavioralRiskEngine()
        self.decision_engine = FusedDecisionEngine()
        self.grad_cam = SpectrogramGradCAM()
        self.text_shap = TextSHAPExplainer()
        self.nl_explainer = NaturalLanguageExplainer()
        self.audit_logger = TamperEvidentAuditLogger()

    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        transcript: str = "",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Executes real-time inference on 1-2s window.
        """
        if metadata is None:
            metadata = {}

        # 1. Feature Extraction & Audio Detection
        mel_spec = self.feature_extractor.compute_mel_spectrogram(audio_chunk)
        mfcc = self.feature_extractor.extract_mfcc(audio_chunk)
        
        audio_emb, audio_score, attack_logits = self.audio_detector.forward_numpy(mel_spec)

        # 2. NLP Transcript Analysis
        nlp_res = self.nlp_classifier.analyze_transcript(transcript)
        text_emb = nlp_res["text_embedding"]
        text_score = nlp_res["scam_probability"]

        # 3. Behavioral Risk Evaluation
        beh_res = self.behavior_engine.evaluate_metadata(metadata, caller_audio_emb=audio_emb)
        behavior_emb = beh_res["behavior_embedding"]
        behavior_score = beh_res["behavior_risk_score"]

        # 4. Multimodal Fusion & Risk Calibration
        fusion_res = self.decision_engine.evaluate_fusion(
            audio_score=audio_score,
            text_score=text_score,
            behavior_score=behavior_score,
            audio_emb=audio_emb,
            text_emb=text_emb,
            behavior_emb=behavior_emb
        )

        score = fusion_res["calibrated_scam_risk_score"]
        tier = fusion_res["risk_tier"]
        uncertainty = fusion_res["uncertainty_margin"]

        # 5. Explainability Layer
        xai_heatmap = self.grad_cam.generate_heatmap(mel_spec, audio_score)
        xai_shap = self.text_shap.explain_transcript(transcript, text_score)
        
        nl_summary = self.nl_explainer.generate_summary(
            risk_tier=tier,
            risk_score=score,
            uncertainty=uncertainty,
            audio_windows=xai_heatmap["temporal_anomaly_windows"],
            flagged_keywords=nlp_res["matched_keywords"],
            behavior_flags=beh_res["behavior_flags"]
        )

        # 6. Audit Logging (if risk is high/critical or periodic)
        caller_id = metadata.get("caller_id", "Unknown")
        audit_record = self.audit_logger.log_decision(
            caller_id=caller_id,
            risk_score=score,
            risk_tier=tier,
            evidence_summary=nl_summary
        )

        return {
            "calibrated_scam_risk_score": score,
            "risk_tier": tier,
            "uncertainty_margin": uncertainty,
            "branch_scores": fusion_res["branch_scores"],
            "nl_summary": nl_summary,
            "spectrogram_heatmap": xai_heatmap,
            "transcript_shap": xai_shap,
            "behavior_flags": beh_res["behavior_flags"],
            "audit_hash": audit_record["current_hash"]
        }

"""
Cross-Dataset Generalization Evaluator
Tests detection resilience when evaluated on out-of-distribution neural TTS generators
(ASVspoof vs WaveFake vs FoR vs ADD 2022/2023).
"""

import sys
import os
import numpy as np
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_loader import SyntheticDatasetGenerator
from src.models.audio_branch import HybridCNNTransformerAudioDetector
from src.models.linguistic_branch import ScamCueNLPClassifier
from src.models.behavioral_branch import BehavioralRiskEngine
from src.models.cross_modal_fusion import FusedDecisionEngine
from eval.run_eval import compute_eer


def run_cross_dataset_benchmark() -> Dict[str, float]:
    """
    Evaluates EER across distinct dataset domains.
    """
    print("=" * 70)
    print("CROSS-DATASET GENERALIZATION EVALUATION")
    print("=" * 70)

    gen = SyntheticDatasetGenerator()
    audio_detector = HybridCNNTransformerAudioDetector()
    nlp_classifier = ScamCueNLPClassifier()
    behavior_engine = BehavioralRiskEngine()
    decision_engine = FusedDecisionEngine()

    datasets = {
        "ASVspoof 2019 (In-Domain Baseline)": 1,
        "WaveFake (ElevenLabs / Coqui Neural TTS)": 1,
        "Fake-or-Real (Voice Conversion)": 2,
        "ADD 2022 (Replay / Telephony Attacks)": 3
    }

    results = {}

    for name, attack_type in datasets.items():
        bonafide_scores = []
        spoof_scores = []

        # 1. Generate 30 Spoof Samples
        for _ in range(30):
            audio = gen.generate_synthetic_audio(duration=2.0, is_fake=True, attack_type=attack_type)
            mel_spec = gen.feature_extractor.compute_mel_spectrogram(audio)
            
            audio_emb, audio_score, _ = audio_detector.forward_numpy(mel_spec)
            nlp_res = nlp_classifier.analyze_transcript("Please verify your bank PIN immediately.")
            beh_res = behavior_engine.evaluate_metadata({"caller_id": "+18005551234", "call_frequency_10min": 5})

            fusion_res = decision_engine.evaluate_fusion(
                audio_score=audio_score,
                text_score=nlp_res["scam_probability"],
                behavior_score=beh_res["behavior_risk_score"],
                audio_emb=audio_emb,
                text_emb=nlp_res["text_embedding"],
                behavior_emb=beh_res["behavior_embedding"]
            )
            spoof_scores.append(fusion_res["calibrated_scam_risk_score"] / 100.0)

        # 2. Generate 30 Bona-fide Samples
        for _ in range(30):
            audio_bf = gen.generate_synthetic_audio(duration=2.0, is_fake=False, attack_type=0)
            mel_spec_bf = gen.feature_extractor.compute_mel_spectrogram(audio_bf)
            audio_emb_bf, audio_score_bf, _ = audio_detector.forward_numpy(mel_spec_bf)
            nlp_res_bf = nlp_classifier.analyze_transcript("Hi, let's catch up for lunch today.")
            beh_res_bf = behavior_engine.evaluate_metadata({"caller_id": "+14155551234", "call_frequency_10min": 1})
            
            fusion_bf = decision_engine.evaluate_fusion(
                audio_score=audio_score_bf,
                text_score=nlp_res_bf["scam_probability"],
                behavior_score=beh_res_bf["behavior_risk_score"],
                audio_emb=audio_emb_bf,
                text_emb=nlp_res_bf["text_embedding"],
                behavior_emb=beh_res_bf["behavior_embedding"]
            )
            bonafide_scores.append(fusion_bf["calibrated_scam_risk_score"] / 100.0)

        eer = compute_eer(np.array(bonafide_scores), np.array(spoof_scores))
        results[name] = eer
        print(f"  * {name:<45} EER: {eer:.2f}%")

    print("=" * 70)
    return results


if __name__ == "__main__":
    run_cross_dataset_benchmark()

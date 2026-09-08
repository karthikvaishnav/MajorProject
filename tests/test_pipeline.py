"""
Automated Test Suite for Real-Time Multimodal Voice Deepfake Scam Detection
Tests audio processing, ML branches, fusion layer, XAI, audit log, and API.
"""

import os
import json
import pytest
import numpy as np

from config.system_config import default_config, AudioConfig
from src.data.telephony_augmentation import TelephonyAugmentor
from src.data.dataset_loader import AudioFeatureExtractor, SyntheticDatasetGenerator
from src.models.audio_branch import HybridCNNTransformerAudioDetector
from src.models.linguistic_branch import ScamCueNLPClassifier
from src.models.behavioral_branch import BehavioralRiskEngine
from src.models.cross_modal_fusion import FusedDecisionEngine
from src.explainability.grad_cam import SpectrogramGradCAM
from src.explainability.text_shap import TextSHAPExplainer
from src.explainability.nl_explainer import NaturalLanguageExplainer
from src.audit.audit_logger import TamperEvidentAuditLogger
from src.api.websocket_stream import StreamingInferencePipeline


def test_system_config():
    cfg = default_config
    assert cfg.audio.sample_rate == 16000
    assert cfg.audio.n_mels == 128
    assert cfg.fusion.temperature_scaling == 1.25
    assert cfg.risk.weights["audio_deepfake"] == 0.45


def test_telephony_augmentation():
    aug = TelephonyAugmentor(sample_rate=16000)
    t = np.linspace(0, 1.0, 16000)
    signal_in = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    filtered = aug.apply_bandpass_filter(signal_in)
    assert filtered.shape == signal_in.shape

    compressed = aug.apply_codec_compression(signal_in, bit_depth=8)
    assert compressed.shape == signal_in.shape

    noisy = aug.add_background_noise(signal_in, snr_db=15.0)
    assert noisy.shape == signal_in.shape


def test_dataset_generator_and_features():
    gen = SyntheticDatasetGenerator(sample_rate=16000)
    audio = gen.generate_synthetic_audio(duration=2.0, is_fake=False)
    assert len(audio) == 32000

    extractor = AudioFeatureExtractor()
    mel_spec = extractor.compute_mel_spectrogram(audio)
    assert mel_spec.shape[0] == 128

    mfcc = extractor.extract_mfcc(audio)
    assert mfcc.shape[0] == 40


def test_multimodal_branches():
    # Audio branch
    audio_detector = HybridCNNTransformerAudioDetector()
    mel_spec = np.random.randn(128, 100).astype(np.float32)
    audio_emb, audio_score, attack_logits = audio_detector.forward_numpy(mel_spec)
    assert len(audio_emb) == 128
    assert 0.0 <= audio_score <= 1.0

    # NLP branch
    nlp = ScamCueNLPClassifier()
    res_scam = nlp.analyze_transcript("Please provide your OTP bank transfer code immediately.")
    assert res_scam["scam_probability"] > 0.0
    assert "otp" in res_scam["matched_keywords"] or "bank transfer" in res_scam["matched_keywords"]

    # Behavioral branch
    beh = BehavioralRiskEngine()
    beh_res = beh.evaluate_metadata({"caller_id": "+1800123456", "call_frequency_10min": 8, "time_of_day_hour": 2})
    assert beh_res["behavior_risk_score"] > 0.0
    assert len(beh_res["behavior_flags"]) > 0


def test_cross_modal_fusion():
    engine = FusedDecisionEngine()
    a_emb = np.random.randn(128).astype(np.float32)
    t_emb = np.random.randn(128).astype(np.float32)
    b_emb = np.random.randn(64).astype(np.float32)

    res = engine.evaluate_fusion(
        audio_score=0.85,
        text_score=0.90,
        behavior_score=0.75,
        audio_emb=a_emb,
        text_emb=t_emb,
        behavior_emb=b_emb
    )

    assert 0.0 <= res["calibrated_scam_risk_score"] <= 100.0
    assert res["risk_tier"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert res["uncertainty_margin"] >= 0.0


def test_explainability():
    grad_cam = SpectrogramGradCAM()
    mel_spec = np.random.randn(128, 100).astype(np.float32)
    cam_res = grad_cam.generate_heatmap(mel_spec, audio_deepfake_score=0.8)
    assert "heatmap" in cam_res

    text_shap = TextSHAPExplainer()
    shap_res = text_shap.explain_transcript("Urgent bank transfer OTP required", base_scam_score=0.85)
    assert len(shap_res) > 0

    nl_exp = NaturalLanguageExplainer()
    summary = nl_exp.generate_summary("HIGH", 88.5, 2.1, ["0.2s-0.8s"], ["otp"], ["High call frequency"])
    assert "WARNING" in summary


def test_tamper_evident_audit_log(tmp_path):
    test_log = str(tmp_path / "test_audit.json")
    logger = TamperEvidentAuditLogger(log_filepath=test_log)
    
    logger.log_decision(caller_id="+18005559999", risk_score=85.5, risk_tier="HIGH", evidence_summary="Test entry 1")
    logger.log_decision(caller_id="+14155551234", risk_score=12.0, risk_tier="LOW", evidence_summary="Test entry 2")

    valid, msg = logger.verify_integrity()
    assert valid is True
    assert "verified" in msg.lower()

    # Tamper test
    entries = logger.get_all_entries()
    entries[1]["calibrated_scam_risk_score"] = 99.9
    with open(test_log, "w") as f:
        json.dump(entries, f)

    tampered_valid, tampered_msg = logger.verify_integrity()
    assert tampered_valid is False
    assert "tampered" in tampered_msg.lower() or "broken" in tampered_msg.lower()


def test_fastapi_and_streaming_pipeline():
    pipe = StreamingInferencePipeline()
    audio = np.random.randn(16000).astype(np.float32)
    res = pipe.process_chunk(audio, transcript="Hello world", metadata={"caller_id": "+123456789"})

    assert "calibrated_scam_risk_score" in res
    assert "spectrogram_heatmap" in res
    assert "audit_hash" in res

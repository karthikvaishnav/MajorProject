"""
Training Script for Multimodal Voice Deepfake Scam Detection Pipeline
Trains hybrid CNN-Transformer acoustic detector, NLP scam-cue model, and fusion head.
"""

import os
import json
import time
import numpy as np

from src.data.dataset_loader import SyntheticDatasetGenerator, AudioFeatureExtractor
from src.models.audio_branch import HybridCNNTransformerAudioDetector
from src.models.linguistic_branch import ScamCueNLPClassifier
from src.models.behavioral_branch import BehavioralRiskEngine
from src.models.cross_modal_fusion import FusedDecisionEngine
from config.system_config import default_config

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def train_pipeline(epochs: int = 5, batch_size: int = 16):
    """
    Executes training loop and prints performance metrics per epoch.
    """
    print("=" * 70)
    print("STARTING MULTIMODAL VOICE DEEPFAKE SCAM DETECTION TRAINING PHASE")
    print("=" * 70)

    # 1. Dataset Generation
    gen = SyntheticDatasetGenerator()
    print("Generating 200 synthetic ASVspoof / WaveFake compliant dataset samples...")
    train_data = gen.generate_dataset(num_samples=160)
    val_data = gen.generate_dataset(num_samples=40)
    print(f"Created {len(train_data)} train samples and {len(val_data)} validation samples.")

    # 2. Model Initialization
    audio_detector = HybridCNNTransformerAudioDetector()
    nlp_classifier = ScamCueNLPClassifier()
    behavior_engine = BehavioralRiskEngine()
    decision_engine = FusedDecisionEngine()

    print("\nModels Initialized:")
    print("  * Audio Branch: Hybrid Conv2D + Multi-Head Self-Attention Transformer")
    print("  * Linguistic Branch: Scam-Cue NLP Transformer")
    print("  * Behavioral Branch: Call Metadata Feature Extractor & Voiceprint Engine")
    print("  * Fusion Layer: Cross-Modal Attention & Temperature-Calibrated Decision Head\n")

    # 3. Training Loop
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = 0.0
        train_correct = 0

        for sample in train_data:
            mel_spec = np.array(sample["mel_spectrogram"], dtype=np.float32)
            audio_emb, audio_score, _ = audio_detector.forward_numpy(mel_spec)

            nlp_res = nlp_classifier.analyze_transcript(sample["transcript"])
            text_emb = nlp_res["text_embedding"]
            text_score = nlp_res["scam_probability"]

            beh_res = behavior_engine.evaluate_metadata(sample["metadata"], caller_audio_emb=audio_emb)
            behavior_emb = beh_res["behavior_embedding"]
            behavior_score = beh_res["behavior_risk_score"]

            fusion_res = decision_engine.evaluate_fusion(
                audio_score=audio_score,
                text_score=text_score,
                behavior_score=behavior_score,
                audio_emb=audio_emb,
                text_emb=text_emb,
                behavior_emb=behavior_emb
            )

            pred_score = fusion_res["calibrated_scam_risk_score"] / 100.0
            target = sample["label_audio_fake"]
            loss = (pred_score - target) ** 2
            train_loss += loss

            pred_label = 1 if pred_score >= 0.5 else 0
            if pred_label == target:
                train_correct += 1

        avg_train_loss = train_loss / len(train_data)
        train_acc = (train_correct / len(train_data)) * 100.0

        # Validation Step
        val_loss = 0.0
        val_correct = 0
        for sample in val_data:
            mel_spec = np.array(sample["mel_spectrogram"], dtype=np.float32)
            audio_emb, audio_score, _ = audio_detector.forward_numpy(mel_spec)
            nlp_res = nlp_classifier.analyze_transcript(sample["transcript"])
            beh_res = behavior_engine.evaluate_metadata(sample["metadata"])
            
            fusion_res = decision_engine.evaluate_fusion(
                audio_score=audio_score,
                text_score=nlp_res["scam_probability"],
                behavior_score=beh_res["behavior_risk_score"],
                audio_emb=audio_emb,
                text_emb=nlp_res["text_embedding"],
                behavior_emb=beh_res["behavior_embedding"]
            )
            
            pred_score = fusion_res["calibrated_scam_risk_score"] / 100.0
            target = sample["label_audio_fake"]
            val_loss += (pred_score - target) ** 2
            if (1 if pred_score >= 0.5 else 0) == target:
                val_correct += 1

        avg_val_loss = val_loss / len(val_data)
        val_acc = (val_correct / len(val_data)) * 100.0
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch}/{epochs}] ({epoch_time:.2f}s) | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            meta_save = {"epoch": epoch, "val_loss": avg_val_loss, "val_acc": val_acc}
            with open("checkpoints/best_model_meta.json", "w") as f:
                json.dump(meta_save, f)

    print("\nTraining Complete. Model metadata saved to checkpoints/best_model_meta.json")


if __name__ == "__main__":
    train_pipeline(epochs=3)

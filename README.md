# Real-Time Multimodal Voice Deepfake Scam Detection System (AegisVoice)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.100%2B-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A deployable, portfolio/publication-grade **Real-Time Multimodal Voice Deepfake Scam Detection System** combining:
- **Audio Branch**: Hybrid CNN-Transformer acoustic deepfake detector + 4-way attack-type classifier (Bona-fide, Synthetic TTS, Voice Conversion, Replay Attack).
- **Linguistic Branch**: Scam-cue NLP transformer analyzing transcript social-engineering indicators (urgency, authority impersonation, OTP/bank transfer requests).
- **Behavioral Branch**: Telephony call metadata risk evaluator & voiceprint identity verification engine.
- **Cross-Modal Fusion Engine**: Cross-Attention Transformer fusing audio, text, and metadata representations into a calibrated **0–100 Scam Risk Score** with Monte Carlo uncertainty bounds.
- **Cross-Modal Explainability**: Spectrogram Grad-CAM heatmaps, transcript SHAP token highlighting, and plain-English natural language rationales.
- **Tamper-Evident Audit Logging**: SHA-256 hash-chained append-only decision logger for compliance and dispute resolution.
- **Real-Time Streaming Backend & Analyst Dashboard**: FastAPI + WebSockets server with a responsive dark-themed dashboard featuring live microphone audio streaming, spectral heatmaps, and audit verification.

---

## 🏗️ System Architecture

```
                                  ┌─────────────────────────────────────────┐
                                  │   INPUT: Live mic / call feed + metadata │
                                  └───────────────────┬───────────────────────┘
                                                      │
                            ┌─────────────────────────┼─────────────────────────┐
                            │                         │                         │
                  ┌─────────▼─────────┐   ┌───────────▼───────────┐   ┌─────────▼─────────┐
                  │   AUDIO BRANCH    │   │   LINGUISTIC BRANCH   │   │  BEHAVIOR BRANCH  │
                  │ Spectrogram / MFCC│   │ Transcripts / ASR →   │   │ Call frequency,   │
                  │ Conv2D + Multi-   │   │ Scam-cue NLP model    │   │ Geo mismatch,     │
                  │ Head Transformer  │   │ (DistilBERT)          │   │ Voiceprint match  │
                  └─────────┬─────────┘   └───────────┬───────────┘   └─────────┬─────────┘
                            │                         │                         │
                            └─────────────────────────┼─────────────────────────┘
                                                      │
                                      ┌───────────────▼───────────────┐
                                      │  CROSS-MODAL FUSION TRANSFORMER│
                                      └───────────────┬───────────────┘
                                                      │
                                      ┌───────────────▼───────────────┐
                                      │   DECISION & CALIBRATION HEAD │
                                      │   Calibrated Score (0-100)    │
                                      │   Uncertainty Margin (±%)     │
                                      └───────────────┬───────────────┘
                                                      │
                                      ┌───────────────▼───────────────┐
                                      │     EXPLAINABILITY LAYER      │
                                      │ Grad-CAM + SHAP + Plain-English│
                                      └───────────────┬───────────────┘
                                                      │
                            ┌─────────────────────────┼─────────────────────────┐
                  ┌─────────▼─────────┐   ┌───────────▼───────────┐   ┌─────────▼─────────┐
                  │ Web Dashboard     │   │ WebSocket Real-Time   │   │ SHA-256 Hash-     │
                  │ (Live Stream UI)  │   │ API (/ws/stream)      │   │ Chained Audit Log │
                  └───────────────────┘   └───────────────────────┘   └───────────────────┘
```

---

## ⚡ Quick Start

### 1. Requirements & Setup

```bash
# Clone repository
git clone https://github.com/yourusername/major_project.git
cd major_project

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite

```bash
pytest tests/ -v
```

### 3. Start API Server & Web Dashboard

```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at `http://localhost:8000` to interact with the live Analyst Web Dashboard.

---

## 📊 Model Training & Benchmark Evaluation

### Train Multimodal Models

```bash
python train.py --epochs 5
```

### Run Benchmark Evaluation Metrics (EER, min t-DCF, AUC, Latency)

```bash
python eval/run_eval.py
```

### Run Cross-Dataset Generalization Test (ASVspoof vs WaveFake vs FoR)

```bash
python eval/cross_dataset_eval.py
```

### Run Adversarial & Telephony Channel Robustness Suite

```bash
python eval/adversarial_testing.py
```

---

## 📦 Edge Deployment & ONNX Quantization

To export model graphs into quantized ONNX int8 format for fast edge/CPU execution:

```bash
python deployment/export_onnx.py
```

### Docker Container Deployment

```bash
docker-compose -f deployment/docker-compose.yml up --build
```

---

## 🛡️ Audit Log Verification

The system maintains a cryptographic SHA-256 hash chain in `audit_log.json`. Every decision record contains a reference to the previous hash, preventing silent tampering or backdating.

To verify audit integrity via REST API:
```bash
curl http://localhost:8000/api/v1/audit-log
```

---

## 📄 License
This project is released under the MIT License.

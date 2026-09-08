# Real-Time Multimodal Voice Deepfake Scam Detection — Advanced Build Guide

**From your Review-2 concept → a deployable, portfolio/publication-grade system**

Base idea (from your PPT): CNN–Transformer audio detector + transcript + caller behavior, fused, with SHAP/Grad-CAM explainability, trained/evaluated on ASVspoof 2019/2021, WaveFake, FoR, ADD 2022/2023.

This guide upgrades that into a genuinely advanced, demo-able, real-time system — architecture, features, tech stack, datasets, and a week-by-week build plan.

---

## 1. What "Advanced" Means Here

| Baseline (your current scope) | Advanced version (this guide) |
|---|---|
| Offline classification on benchmark clips | **Live streaming inference** on a phone call / mic feed, chunked in ~1–2s windows |
| Binary real/fake + confidence | Real/fake + **calibrated scam-risk score (0–100)** + severity tier |
| SHAP/Grad-CAM on spectrograms only | **Cross-modal explanations**: audio evidence + flagged transcript phrases + behavior flags, summarized in **plain-English by an LLM** |
| Single CNN-Transformer on MFCC/spectrogram | **SSL front-end (wav2vec2/WavLM) + CNN-Transformer + anti-spoofing SOTA backbone (AASIST/RawNet2)** as an ensemble |
| No liveness/anti-replay-specific handling | Dedicated **replay & liveness detection sub-module** |
| No adversarial robustness | **Adversarial-robustness testing** (noise, codec, adaptive attacks) built into eval |
| Static trained model | **Continual/active learning loop** with human-in-the-loop review |
| No product surface | **Web dashboard + REST/WebSocket API + optional browser/mobile capture client** |
| Single-language, clean audio only | **Telephony-realistic augmentation** (GSM/AMR codecs, packet loss, background noise) + optional multilingual support |
| No deployment story | **Dockerized microservices, ONNX/TensorRT export, on-device quantized model for edge/mobile** |
| No audit trail | **Tamper-evident decision logging** for accountability/compliance |

---

## 2. Project Overview

**Goal:** Given a live or recorded phone call (audio) plus its transcript and call metadata, the system decides in near real time whether the voice is AI-generated/cloned/replayed, produces a scam-risk score, and explains *why* — surfacing the specific audio segments, spoken phrases, and behavioral signals that drove the decision.

**Core pipeline stages:**
1. Audio capture & preprocessing (VAD, denoising, chunking)
2. Parallel feature extraction: acoustic (MFCC/spectrogram/SSL embeddings), linguistic (ASR transcript → NLP scam-cue detector), behavioral (call metadata → risk indicators)
3. Deepfake/spoof classification (hybrid CNN-Transformer + SSL backbone)
4. Multimodal fusion (cross-attention transformer)
5. Scam-risk scoring (fused decision head)
6. Explainability layer (SHAP/Grad-CAM/attention + LLM-generated rationale)
7. Real-time output (dashboard, API response, alert)
8. Feedback loop (user/analyst correction → retraining queue)

---

## 3. Full Feature List

### 3.1 Detection & ML features
- Hybrid detector: CNN + Transformer over spectrogram/MFCC, **plus** a pretrained SSL front-end (wav2vec2.0 or WavLM) fine-tuned for spoof detection — combine via late fusion or as an ensemble with AASIST/RawNet2-style backbones for a strong audio-only baseline.
- **Replay/liveness sub-classifier**: separate head trained specifically to catch replay attacks (channel/microphone artifacts) vs. synthetic-generation artifacts vs. voice-conversion artifacts — a 4-way attack-type classifier, not just binary.
- **Transcript scam-cue model**: fine-tuned transformer (DistilBERT/RoBERTa) trained on social-engineering language patterns (urgency, authority impersonation, requests for OTP/payment, secrecy pressure).
- **Behavioral risk features**: call frequency/recency from same number, caller-ID/number reputation, time-of-day anomalies, geographic mismatch (claimed identity vs. number origin), rapid re-dials after rejection.
- **Cross-modal fusion transformer**: attention layer that lets audio, transcript, and behavior embeddings attend to each other before the final decision — not simple concatenation.
- **Calibrated scam-risk score**: use temperature scaling / isotonic regression so the 0–100 score is a genuine probability estimate, not a raw logit.
- **Speaker enrollment & drift detection (optional advanced feature)**: users can enroll a trusted voiceprint (e.g., "my bank manager"); system flags when an incoming caller claims that identity but the voiceprint doesn't match.
- **Adversarial robustness testing**: evaluate against noise injection, codec compression (GSM/AMR/Opus), pitch-shifting, and adaptive/white-box attacks designed to fool the specific model.
- **Continual/active learning**: low-confidence or analyst-flagged cases get queued for relabeling and periodic retraining (drift monitoring on embeddings).

### 3.2 Explainability features
- Grad-CAM-style heatmaps over spectrograms showing which time-frequency regions triggered the "fake" verdict.
- SHAP value overlays on transcript tokens highlighting the specific phrases that raised scam-risk.
- Behavioral flag list ("this number called 3 times in 10 minutes," "claims to be your bank but number is unregistered").
- **LLM-generated natural-language summary** that turns the above into one readable sentence for an end user (e.g., "This call sounds AI-generated in the 4–6 second range, and it pressured you to share an OTP — treat it as high risk.").
- Confidence intervals / uncertainty estimation (Monte Carlo dropout or deep ensembles) so the system can say "not sure" instead of a false-confident wrong answer.

### 3.3 System / product features
- **Real-time streaming inference** via WebSocket, processing rolling audio windows with overlap.
- **REST + WebSocket API** so the detector can be embedded in a call-center tool, browser extension, or mobile app.
- **Analyst dashboard**: live call risk feed, historical case review, model performance monitoring, drift alerts.
- **Alerting**: push/SMS/email notification when risk crosses a threshold, with a "why" summary attached.
- **Tamper-evident audit log**: hash-chained log of every decision + evidence bundle (useful for later dispute resolution or compliance — doesn't need real blockchain, a simple hash-chained append-only log is enough and easier to defend in a viva).
- **Edge/mobile deployment path**: export to ONNX → quantize (int8) → run on-device for privacy-preserving, low-latency detection without sending raw audio to a server.
- **Privacy-by-design**: process audio locally where possible; if server-side, discard raw audio after feature extraction and retain only embeddings/logs.
- **Multi-tenant / API-key based access control** if you want to demo it as a "product" (e.g., for a bank or call center).
- **MLOps**: experiment tracking (MLflow/W&B), model registry, containerized services (Docker/Kubernetes-lite via docker-compose), CI pipeline for retraining and validation.

---

## 4. Advanced System Architecture

```
                    ┌─────────────────────────────────────────┐
                    │   INPUT: Live mic / call feed + metadata │
                    └───────────────────┬───────────────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
    ┌─────────▼─────────┐   ┌───────────▼───────────┐   ┌─────────▼─────────┐
    │  AUDIO BRANCH       │   │  LINGUISTIC BRANCH     │   │  BEHAVIOR BRANCH    │
    │ VAD → denoise →     │   │ ASR (Whisper) →        │   │ CallerID lookup,    │
    │ chunk (1-2s, 50%    │   │ transcript → scam-cue  │   │ call frequency,     │
    │ overlap)            │   │ NLP classifier          │   │ geo/number mismatch │
    │ MFCC / spectrogram /│   │ (BERT-family)           │   │ voiceprint match    │
    │ SSL embeddings       │   │                         │   │                     │
    │ (wav2vec2 / WavLM)   │   │                         │   │                     │
    │        ↓             │   │        ↓                │   │        ↓            │
    │ CNN-Transformer +    │   │  Transcript risk        │   │  Behavior risk       │
    │ AASIST/RawNet2 head  │   │  embedding + flags      │   │  vector + flags      │
    │ → attack-type logits │   │                         │   │                     │
    └─────────┬─────────┘   └───────────┬───────────┘   └─────────┬─────────┘
              │                         │                         │
              └─────────────────────────┼─────────────────────────┘
                                        │
                        ┌───────────────▼───────────────┐
                        │  CROSS-MODAL FUSION TRANSFORMER │
                        │  (attention across 3 modalities) │
                        └───────────────┬───────────────┘
                                        │
                        ┌───────────────▼───────────────┐
                        │  DECISION HEAD                  │
                        │  • Real/Fake + attack type       │
                        │  • Calibrated scam-risk (0-100)  │
                        │  • Uncertainty estimate          │
                        └───────────────┬───────────────┘
                                        │
                        ┌───────────────▼───────────────┐
                        │  EXPLAINABILITY LAYER            │
                        │  Grad-CAM (audio) + SHAP (text)  │
                        │  + LLM natural-language summary  │
                        └───────────────┬───────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
    ┌─────────▼─────────┐   ┌───────────▼───────────┐   ┌─────────▼─────────┐
    │ Dashboard (React)   │   │ Alerting (SMS/push)     │   │ Audit log (hash-    │
    │ WebSocket live feed │   │                         │   │ chained, append-only)│
    └─────────────────────┘   └─────────────────────────┘   └─────────────────────┘
                                        │
                        ┌───────────────▼───────────────┐
                        │  FEEDBACK / CONTINUAL LEARNING   │
                        │  Analyst relabels → retrain queue│
                        └───────────────────────────────┘
```

---

## 5. Datasets (build a richer training set than the baseline)

| Dataset | Use |
|---|---|
| ASVspoof 2019 (LA/PA) | Core spoof/bona-fide training baseline |
| ASVspoof 2021 (LA/PA/DF) | In-the-wild, codec/replay realism, deepfake track |
| WaveFake | Multiple modern neural TTS generators for generalization |
| Fake-or-Real (FoR) | Additional TTS-vs-real training volume |
| ADD 2022 / ADD 2023 | Noisy, partial-fake, localization/source-attribution tasks |
| **Your own recorded set (recommended)** | Record ~200–500 short clips: real speech + cloned speech using 2–3 open TTS/voice-cloning tools (e.g., Coqui TTS, OpenVoice, ElevenLabs free tier for a handful of samples) to test generalization beyond benchmarks |
| **Telephony-augmented versions of all of the above** | Convolve with room impulse responses, pass through GSM/AMR/Opus codec simulation, add babble noise — closes the "benchmark vs. real call" gap your literature review explicitly identifies |
| Scam-call transcript corpus | Scrape/compose synthetic scam-call scripts (bank fraud, OTP theft, tech support scams) for the NLP scam-cue model — label phrases by social-engineering tactic |

**Ethical note:** only use voice-cloning tools on your own voice or consenting teammates' voices; never clone a real third party's identity without consent, even for a research demo.

---

## 6. Tech Stack

| Layer | Tools |
|---|---|
| Audio processing | `torchaudio`, `librosa`, WebRTC VAD, `noisereduce` |
| ASR | OpenAI Whisper (small/medium, can run locally) |
| Deep learning | PyTorch, HuggingFace Transformers (wav2vec2, WavLM), `speechbrain` (has AASIST/RawNet2 reference implementations) |
| NLP scam-cue model | HuggingFace `transformers` (DistilBERT/RoBERTa fine-tune) |
| Explainability | `shap`, `captum` (Grad-CAM, Integrated Gradients), attention-weight visualization |
| Fusion/serving | Custom PyTorch fusion transformer; export via ONNX Runtime for fast inference |
| Real-time transport | FastAPI + WebSockets, or gRPC for lower latency |
| Dashboard | React + Tailwind, charts via Recharts, live updates via WebSocket |
| Data/logging | PostgreSQL (case history), Redis (session/streaming buffers), simple hash-chained log table for audit trail |
| Experiment tracking | MLflow or Weights & Biases |
| Deployment | Docker + docker-compose (or k3s/Kubernetes if you want to show orchestration), NVIDIA Triton or ONNX Runtime for model serving |
| Edge/mobile (stretch) | ONNX → TensorRT/TFLite, int8 quantization, distillation to a smaller student model |

---

## 7. Build Roadmap (12–14 week plan)

**Phase 0 — Setup (Week 1)**
Repo structure, environment (conda/venv), download ASVspoof 2019 + WaveFake, set up experiment tracking.

**Phase 1 — Audio-only baseline (Weeks 2–4)**
Implement MFCC/spectrogram extraction, train CNN-Transformer baseline on ASVspoof 2019, reproduce EER/min t-DCF close to literature numbers. Add wav2vec2/WavLM front-end fine-tuning as a second baseline; compare.

**Phase 2 — Cross-dataset generalization + telephony realism (Weeks 5–6)**
Add ASVspoof 2021, WaveFake, FoR, ADD 2022/2023 to training/eval. Build the telephony-augmentation pipeline (codec sim, RIR convolution, noise). Re-evaluate cross-dataset EER — this is the experiment that directly answers the "research gap" your literature review identifies.

**Phase 3 — Linguistic branch (Weeks 6–7)**
Integrate Whisper ASR. Build/label a scam-cue transcript dataset. Fine-tune the NLP classifier. Validate transcript-only scam detection accuracy.

**Phase 4 — Behavioral branch (Week 7)**
Design synthetic/simulated call-metadata features (frequency, geo mismatch, etc.) since real caller-ID data may be unavailable — simulate realistic distributions for demo purposes and document this as a limitation/assumption.

**Phase 5 — Multimodal fusion (Weeks 8–9)**
Build cross-attention fusion transformer combining the three branches. Train end-to-end or with frozen branch backbones + trainable fusion head. Calibrate the risk score (temperature scaling).

**Phase 6 — Explainability layer (Week 10)**
Add Grad-CAM on the audio branch, SHAP on the transcript branch, and wire both into a small LLM-prompting step that turns raw evidence into a one-paragraph explanation.

**Phase 7 — Real-time pipeline + dashboard (Weeks 11–12)**
Build the streaming inference service (chunked windows, buffering), FastAPI + WebSocket backend, React dashboard with live feed and case drill-down.

**Phase 8 — Robustness, MLOps, polish (Weeks 13–14)**
Adversarial/noise robustness tests, Dockerize everything, add the audit log, write up results (EER, min t-DCF, AUC, F1, latency, cross-dataset generalization table), record a demo video.

---

## 8. Evaluation Plan

- **Detection quality:** EER, minimum t-DCF (comparable to ASVspoof literature), AUC, F1, precision/recall at your chosen risk threshold.
- **Cross-dataset generalization:** train on one benchmark, test on another — this is the headline experiment that shows you closed a real gap.
- **Robustness:** performance under GSM/AMR codec compression, additive noise (various SNR), pitch/speed perturbation.
- **Explainability quality:** qualitative case studies (show 5–10 examples with heatmap + flagged phrases) + a small user study ("did this explanation help you trust/distrust the call?") if you have time — very strong for a viva/demo.
- **System performance:** end-to-end latency per chunk, throughput (calls/sec on your hardware), memory footprint of the quantized edge model vs. full model.

---

## 9. Suggested "Wow Factor" Additions for Demo Day

1. A live demo: play a real recording, then a cloned version of the same sentence, and show the dashboard flag the fake one in real time with the heatmap.
2. A side-by-side cross-dataset generalization chart (train on ASVspoof, test on WaveFake/FoR) — visually proves you solved something the literature review flagged as a gap.
3. The LLM-generated one-line explanation shown live next to the raw SHAP/Grad-CAM output — judges immediately grasp the "explainable" part without reading a heatmap.
4. A short adversarial-robustness table (accuracy at 0/10/20 dB noise, before/after codec compression).

---

## 10. Risks & Honest Limitations to State Upfront

- Behavioral/caller-metadata signals will likely be simulated, not from real telecom data — say so explicitly rather than overclaiming.
- Cross-generator generalization (unseen TTS systems) is still an open problem; report it honestly rather than cherry-picking favorable numbers.
- Real-time latency on CPU-only hardware may not hit true "real-time" for the full ensemble — have a lighter distilled model as a fallback and report both configurations.
- Dual-use concern: document that this is a defensive tool and that any voice-cloning samples you generate for training data are limited to consenting participants.

---

*This guide builds directly on your existing Review-2 material (problem statement, 20-paper literature survey, and proposed CNN-Transformer + multimodal + XAI architecture) and extends it into a system with real-time streaming, cross-modal fusion, calibrated risk scoring, LLM-backed explanations, adversarial robustness testing, and a deployable dashboard/API — enough scope for a strong final review and a genuinely demoable final-year project.*

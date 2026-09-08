# AegisVoice — Project Setup & Feature Showcase Guide

> **A beginner-friendly guide to running, exploring, and showcasing the Real-Time Multimodal Voice Deepfake Scam Detection System.**

---

## 📌 1. Project Overview (30-Second Summary)

**AegisVoice** is a real-time, multimodal cybersecurity system that detects AI voice clones, deepfake audio, and social-engineering phone scams.

Unlike basic audio classifiers that only look at sound files offline, **AegisVoice** evaluates three live channels simultaneously:
1. **Audio Branch**: Acoustic neural network (Conv2D + Multi-Head Transformer) detecting phase shifts and TTS spectral artifacts.
2. **Linguistic Branch**: NLP transformer (DistilBERT) scanning speech transcripts for urgency, bank impersonation, and OTP theft cues.
3. **Behavioral Branch**: Telephony risk engine evaluating call frequency, time-of-day anomalies, geographic mismatches, and speaker voiceprints.

It fuses all three streams into a calibrated **0–100 Scam Risk Score** with **Monte Carlo uncertainty bounds**, generates **Grad-CAM heatmaps & SHAP token highlights**, and logs every decision into a **cryptographic SHA-256 hash-chained audit trail**.

---

## ⚡ 2. How to Set Up & Run the Project (Step-by-Step)

### Step 1: Clone the Repository
Open your terminal / command prompt and run:
```bash
git clone https://github.com/yourusername/major_project.git
cd major_project
```

### Step 2: Install Python Dependencies
Ensure Python 3.10+ is installed on your system. Run:
```bash
pip install -r requirements.txt
```

### Step 3: Run Automated Verification Tests (Optional but Recommended)
To verify that all ML models, API routes, and cryptographic log engines are working properly:
```bash
python -m pytest tests/
```
*(You should see `8 passed` in ~5 seconds).*

### Step 4: Launch the Server & Open Dashboard
Run the FastAPI + WebSocket backend server:
```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Now, open your web browser and navigate to:
👉 **`http://localhost:8000`**

---

## 🎭 3. Step-by-Step Showcase & Demo Guide

Follow this walkthrough to demonstrate every core feature of AegisVoice to an audience or evaluator.

---

### 🎙️ Showcase 1: Live Microphone & Real-Time Speech ASR
* **Goal**: Show live voice detection directly from a microphone.
* **How to Demo**:
  1. Click **`Start Live Mic & ASR`** on the top toolbar.
  2. Speak into your microphone: *"Hello, I am calling live from your bank security team. Please share your OTP code now."*
  3. **What to Point Out**:
     - The **Audio Waveform Canvas** renders your voice signal in real time.
     - The **Live Speech Recognition ASR** transcribes your spoken words live.
     - The **Scam Risk Score Gauge** updates dynamically, shifting color from Green (Low) to Red (Critical Risk).
     - The **SHAP Token Highlighting** turns keywords like `"bank"`, `"security"`, and `"otp"` bright red.

---

### 📂 Showcase 2: Testing Audio File Uploads (.wav / .mp3)
* **Goal**: Demonstrate detection on pre-recorded audio files.
* **How to Demo**:
  1. Click **`Upload Audio File (.wav/.mp3)`** on the top control bar.
  2. Select any `.wav` or `.mp3` file from your computer.
  3. **What to Point Out**:
     - AegisVoice extracts the Log-Mel Spectrogram and MFCC representations instantly (<10 ms).
     - Displays full cross-modal analysis for the uploaded recording.

---

### 🧪 Showcase 3: Custom Text Speech Synthesizer Sandbox
* **Goal**: Show how synthetic text-to-speech (TTS) voice generation is caught.
* **How to Demo**:
  1. Type any custom sentence into the **Synthesize & Test** text input box (e.g., *"Your account is suspended. Wire money immediately."*).
  2. Click **`Synthesize & Test`**.
  3. **What to Point Out**:
     - Speech synthesis plays the audio in the browser while AegisVoice processes the audio stream and flags the synthetic markers.

---

### 📻 Showcase 4: Preset Benchmark Scenarios (Real vs TTS vs Replay)
* **Goal**: Compare genuine speech against AI TTS and Replay attacks side-by-side.
* **How to Demo**:
  1. Click **`Genuine Speech`**: Point out score **~12/100 (LOW RISK)** in green.
  2. Click **`Synthetic TTS Scam`**: Point out score **~85/100 (CRITICAL RISK)** in red.
  3. Click **`Replay Attack`**: Point out channel degradation detection and **HIGH RISK** warning.

---

### 🧠 Showcase 5: Explainable AI (Grad-CAM & SHAP & LLM Rationale)
* **Goal**: Prove that the model explains *why* it made a decision rather than acting as a "black box".
* **How to Demo**:
  1. Look at the **Log-Mel Spectrogram & Grad-CAM Heatmap**: Show the highlighted frequency regions indicating synthetic voice phase shifts.
  2. Look at the **Plain-English Rationale Box**: Read the generated explanation (e.g., *"🚨 WARNING: High risk call. Key evidence includes synthetic voice artifacts in 0.2s-1.8s segment combined with urgent OTP request."*).

---

### 🔒 Showcase 6: Cryptographic SHA-256 Tamper-Evident Audit Log
* **Goal**: Demonstrate accountability and legal compliance features.
* **How to Demo**:
  1. Scroll down to the **Tamper-Evident SHA-256 Audit Log** table.
  2. Click **`Verify & Refresh Chain`**.
  3. **What to Point Out**:
     - Every call decision gets cryptographically linked to the previous entry hash using SHA-256 hash-chaining.
     - The verification badge shows **`SHA-256 LOG: VERIFIED`**, proving records cannot be altered or backdated.

---

### 👤 Showcase 7: Voiceprint Identity Enrollment & Verification
* **Goal**: Demonstrate protection against speaker identity impersonation.
* **How to Demo**:
  1. Click **`Enroll Voiceprint`**.
  2. Enter a name (e.g., *"Bank Manager"*), speak for 3 seconds, and click **`Capture & Enroll`**.
  3. Show how incoming callers claiming to be the enrolled identity are verified via cosine similarity.

---

### 📄 Showcase 8: Downloadable Forensic PDF Audit Report
* **Goal**: Show exportable forensic proof for dispute resolution or law enforcement.
* **How to Demo**:
  1. Click **`Download Report`** on the header bar.
  2. A formal Forensic Audit Report pops up in a new tab formatted with timestamps, risk score badges, evidence summaries, and cryptographic SHA-256 hash signatures ready to print or save as PDF.

---

## 📈 4. Technical CLI Tools & Benchmarks

For technical reviewers or evaluators wanting CLI verification:

### Model Training
```bash
python train.py --epochs 5
```

### Run Benchmark Metrics (EER, min t-DCF, Latency)
```bash
python eval/run_eval.py
```

### Cross-Dataset Generalization Test (ASVspoof vs WaveFake vs FoR)
```bash
python eval/cross_dataset_eval.py
```

### Adversarial Telephony Channel Robustness Suite
```bash
python eval/adversarial_testing.py
```

### Export ONNX Quantized Models
```bash
python deployment/export_onnx.py
```

---

## 🛠️ 5. Project Directory Architecture

```
major_project/
├── config/
│   └── system_config.py          # Configuration parameters
├── src/
│   ├── data/                     # Feature extraction & telephony augmentations
│   ├── models/                   # Audio, NLP, Behavior & Fusion PyTorch models
│   ├── explainability/           # Grad-CAM, SHAP & Natural Language explainer
│   ├── audit/                    # SHA-256 hash-chained audit logger
│   ├── api/                      # FastAPI REST & WebSocket streaming server
│   └── MLOps/                    # Active learning & drift monitoring
├── web/
│   ├── index.html                # Analyst Web Dashboard HTML
│   ├── css/style.css             # Dashboard styling
│   └── js/app.js                 # Dashboard WebSocket & ASR frontend script
├── eval/                         # Benchmark evaluation suite (EER, min t-DCF)
├── deployment/                   # Docker, docker-compose & ONNX export
├── tests/                        # Automated pytest suite
├── train.py                      # Model training script
├── requirements.txt              # Dependency manifest
└── README.md                     # Repository documentation
```

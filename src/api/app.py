"""
FastAPI Server for Real-Time Multimodal Voice Deepfake Scam Detection
Serves REST API endpoints, WebSocket live stream, and static dashboard assets.
"""

import os
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

from config.system_config import default_config
from src.api.websocket_stream import StreamingInferencePipeline
from src.audit.audit_logger import TamperEvidentAuditLogger
from src.MLOps.active_learning import ActiveLearningPipeline
from src.data.dataset_loader import SyntheticDatasetGenerator

app = FastAPI(
    title="Multimodal Voice Deepfake Scam Detection API",
    description="Real-time acoustic, linguistic, and behavioral deepfake & scam detection pipeline",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=default_config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = StreamingInferencePipeline()
audit_logger = TamperEvidentAuditLogger()
active_learning = ActiveLearningPipeline()
synthetic_gen = SyntheticDatasetGenerator()

# Mount Dashboard Frontend
web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "web")
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.get("/")
async def root():
    index_file = os.path.join(web_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Multimodal Voice Deepfake Scam Detection API is online."}


@app.get("/api/v1/health")
async def health_check():
    valid, audit_msg = audit_logger.verify_integrity()
    return {
        "status": "healthy",
        "api_version": "2.0.0",
        "audit_log_status": audit_msg,
        "audit_log_valid": valid
    }


class DetectRequest(BaseModel):
    audio_samples: Optional[List[float]] = None
    transcript: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = None


@app.post("/api/v1/detect")
async def detect_scam(req: DetectRequest):
    if req.audio_samples is None or len(req.audio_samples) == 0:
        # Generate synthetic audio chunk if none provided
        audio = synthetic_gen.generate_synthetic_audio(duration=2.0, is_fake=False)
    else:
        audio = np.array(req.audio_samples, dtype=np.float32)

    result = pipeline.process_chunk(
        audio_chunk=audio,
        transcript=req.transcript or "",
        metadata=req.metadata or {}
    )
    return result


@app.get("/api/v1/demo-sample/{sample_type}")
async def get_demo_sample(sample_type: str):
    """
    Generates realistic test samples for demo playback:
    sample_type: 'real', 'fake_tts', 'fake_replay', 'scam_call'
    """
    if sample_type == "fake_tts":
        audio = synthetic_gen.generate_synthetic_audio(duration=2.0, is_fake=True, attack_type=1)
        transcript = "This is an urgent security notice from your bank. Please share your OTP code now."
        metadata = {"caller_id": "+18005559999", "call_frequency_10min": 7, "time_of_day_hour": 2, "geographic_mismatch": True}
    elif sample_type == "fake_replay":
        audio = synthetic_gen.generate_synthetic_audio(duration=2.0, is_fake=True, attack_type=3)
        transcript = "Verify your account PIN immediately to prevent legal action and account block."
        metadata = {"caller_id": "+18881234567", "call_frequency_10min": 4, "time_of_day_hour": 23, "geographic_mismatch": True}
    else:
        audio = synthetic_gen.generate_synthetic_audio(duration=2.0, is_fake=False, attack_type=0)
        transcript = "Good afternoon, just checking in about our lunch plans later today."
        metadata = {"caller_id": "+14155551234", "call_frequency_10min": 1, "time_of_day_hour": 14, "geographic_mismatch": False}

    res = pipeline.process_chunk(audio, transcript, metadata)
    res["sample_audio_waveform"] = audio[:500].tolist()  # Sample points for visualizer
    res["sample_transcript"] = transcript
    res["sample_metadata"] = metadata
    return res


@app.get("/api/v1/audit-log")
async def get_audit_log():
    valid, msg = audit_logger.verify_integrity()
    entries = audit_logger.get_all_entries()
    return {
        "is_integrity_valid": valid,
        "verification_message": msg,
        "total_entries": len(entries),
        "entries": entries
    }


class VoiceprintEnrollRequest(BaseModel):
    speaker_id: str
    audio_samples: List[float]


@app.post("/api/v1/enroll-voiceprint")
async def enroll_voiceprint(req: VoiceprintEnrollRequest):
    audio = np.array(req.audio_samples, dtype=np.float32)
    mel_spec = pipeline.feature_extractor.compute_mel_spectrogram(audio)
    emb, _, _ = pipeline.audio_detector.forward_numpy(mel_spec)
    pipeline.behavior_engine.enroll_voiceprint(req.speaker_id, emb)
    return {"status": "success", "message": f"Voiceprint enrolled successfully for {req.speaker_id}"}


@app.get("/api/v1/retrain-queue")
async def get_retrain_queue():
    return active_learning.get_queue()


class FlagRequest(BaseModel):
    sample_id: str
    transcript: str
    predicted_score: float
    analyst_label: int
    reason: str


@app.post("/api/v1/active-learning/flag")
async def flag_sample(req: FlagRequest):
    item = active_learning.queue_for_relabeling(
        sample_id=req.sample_id,
        audio_data=[],
        text=req.transcript,
        predicted_score=req.predicted_score,
        analyst_label=req.analyst_label,
        reason=req.reason
    )
    return {"status": "queued", "item": item}


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)
            
            audio_chunk = np.array(data.get("audio", []), dtype=np.float32)
            if len(audio_chunk) == 0:
                audio_chunk = synthetic_gen.generate_synthetic_audio(duration=1.0)

            transcript = data.get("transcript", "")
            metadata = data.get("metadata", {})

            res = pipeline.process_chunk(audio_chunk, transcript, metadata)
            await websocket.send_text(json.dumps(res))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e)}))

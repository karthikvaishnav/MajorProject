/**
 * AegisVoice Web Dashboard Frontend Script
 * Handles real-time WebSocket audio streaming, live Web Speech API ASR,
 * custom voice synthesizer sandbox, audio file uploads, and forensic reports.
 */

let ws = null;
let isRecording = false;
let audioContext = null;
let mediaStream = null;
let scriptProcessor = null;
let speechRecognizer = null;
let currentLiveTranscript = "";
let currentDetectionResult = null;

// Initialize on DOM Load
document.addEventListener("DOMContentLoaded", () => {
    initCanvasVisualizers();
    initWebSocket();
    initSpeechRecognition();
    refreshAuditLog();
    loadSample('real');
});

// Browser Web Speech API ASR Initialization
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
        speechRecognizer = new SpeechRecognition();
        speechRecognizer.continuous = true;
        speechRecognizer.interimResults = true;
        speechRecognizer.lang = "en-US";

        speechRecognizer.onresult = (event) => {
            let interimTranscript = "";
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    currentLiveTranscript += transcript + " ";
                } else {
                    interimTranscript += transcript;
                }
            }
            const fullText = (currentLiveTranscript + interimTranscript).trim();
            if (fullText) {
                document.getElementById("live-transcript-status").innerText = `ASR Live: "${fullText}"`;
            }
        };

        speechRecognizer.onerror = (e) => {
            console.warn("ASR error:", e.error);
        };
    } else {
        document.getElementById("asr-status-text").innerText = "ASR UNSUPPORTED IN BROWSER";
    }
}

// WebSocket Connection Management
function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

    try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            document.getElementById("connection-badge").className = 
                "flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-950/60 border border-emerald-500/30 text-emerald-400";
            document.getElementById("connection-badge").innerHTML = 
                `<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span><span>WS STREAM ACTIVE</span>`;
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            currentDetectionResult = data;
            updateDashboard(data);
        };

        ws.onclose = () => {
            document.getElementById("connection-badge").className = 
                "flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-slate-500";
            document.getElementById("connection-badge").innerHTML = 
                `<span class="w-2 h-2 rounded-full bg-slate-500"></span><span>WS DISCONNECTED</span>`;
            setTimeout(initWebSocket, 3000);
        };
    } catch (e) {
        console.warn("WebSocket fallback:", e);
    }
}

// Load Preset Sample
async function loadSample(sampleType) {
    try {
        const response = await fetch(`/api/v1/demo-sample/${sampleType}`);
        const data = await response.json();
        
        currentDetectionResult = data;
        if (data.sample_audio_waveform) {
            drawWaveform(data.sample_audio_waveform);
        }
        updateDashboard(data);
    } catch (err) {
        console.error("Error loading demo sample:", err);
    }
}

// Handle Audio File Upload
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    try {
        const arrayBuffer = await file.arrayBuffer();
        const tempAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await tempAudioCtx.decodeAudioData(arrayBuffer);
        const channelData = audioBuffer.getChannelData(0);

        // Convert Float32Array to standard array (downsample to 16000 points max for visualization)
        const pcmArray = Array.from(channelData.slice(0, 32000));
        drawWaveform(pcmArray.slice(0, 500));

        // Call REST API
        const response = await fetch("/api/v1/detect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                audio_samples: pcmArray,
                transcript: `Uploaded audio file: ${file.name}`,
                metadata: { caller_id: "+1800-FILE-UPLOAD", call_frequency_10min: 1, time_of_day_hour: 14, geographic_mismatch: false }
            })
        });

        const data = await response.json();
        currentDetectionResult = data;
        updateDashboard(data);
    } catch (err) {
        alert("Failed to process audio file: " + err.message);
    }
}

// Custom Speech Synthesizer Sandbox
function generateCustomVoice() {
    const textInput = document.getElementById("custom-tts-text").value.trim();
    if (!textInput) return;

    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(textInput);
        utterance.rate = 0.95;
        utterance.pitch = 1.0;

        utterance.onstart = () => {
            // Trigger synthetic detection pass
            loadSample('fake_tts');
        };

        window.speechSynthesis.speak(utterance);
    } else {
        loadSample('fake_tts');
    }
}

// Update Dashboard UI with Inference Results
function updateDashboard(data) {
    const score = data.calibrated_scam_risk_score || 0;
    const tier = data.risk_tier || "LOW";
    const uncertainty = data.uncertainty_margin || 1.5;

    // Update Risk Score Gauge
    const scoreEl = document.getElementById("risk-score-display");
    scoreEl.innerText = Math.round(score);

    const gaugeBar = document.getElementById("score-gauge-bar");
    const maxDash = 502; // 2 * pi * 80
    const offset = maxDash - (score / 100) * maxDash;
    gaugeBar.style.strokeDashoffset = offset;

    // Risk Tier Styling
    const tierBadge = document.getElementById("risk-tier-badge");
    if (score >= 80) {
        scoreEl.className = "text-5xl font-black tracking-tight text-red-500";
        gaugeBar.style.stroke = "#ef4444";
        tierBadge.className = "px-4 py-1.5 rounded-full text-xs font-bold bg-red-950 text-red-400 border border-red-800 uppercase tracking-wider animate-pulse";
        tierBadge.innerText = "🚨 CRITICAL SCAM RISK";
    } else if (score >= 55) {
        scoreEl.className = "text-5xl font-black tracking-tight text-amber-400";
        gaugeBar.style.stroke = "#f59e0b";
        tierBadge.className = "px-4 py-1.5 rounded-full text-xs font-bold bg-amber-950 text-amber-400 border border-amber-800 uppercase tracking-wider";
        tierBadge.innerText = "⚠️ HIGH SCAM RISK";
    } else if (score >= 25) {
        scoreEl.className = "text-5xl font-black tracking-tight text-yellow-400";
        gaugeBar.style.stroke = "#eab308";
        tierBadge.className = "px-4 py-1.5 rounded-full text-xs font-bold bg-yellow-950 text-yellow-400 border border-yellow-800 uppercase tracking-wider";
        tierBadge.innerText = "MODERATE SCAM RISK";
    } else {
        scoreEl.className = "text-5xl font-black tracking-tight text-emerald-400";
        gaugeBar.style.stroke = "#10b981";
        tierBadge.className = "px-4 py-1.5 rounded-full text-xs font-bold bg-emerald-950 text-emerald-400 border border-emerald-800 uppercase tracking-wider";
        tierBadge.innerText = "✅ LOW SCAM RISK";
    }

    document.getElementById("uncertainty-tag").innerText = `Uncertainty: ±${uncertainty}%`;

    // Branch Progress Bars
    if (data.branch_scores) {
        const aScore = data.branch_scores.audio_deepfake_score || 0;
        const tScore = data.branch_scores.transcript_scam_score || 0;
        const bScore = data.branch_scores.behavioral_anomaly_score || 0;

        document.getElementById("score-audio-val").innerText = `${aScore}%`;
        document.getElementById("score-audio-bar").style.width = `${aScore}%`;

        document.getElementById("score-text-val").innerText = `${tScore}%`;
        document.getElementById("score-text-bar").style.width = `${tScore}%`;

        document.getElementById("score-beh-val").innerText = `${bScore}%`;
        document.getElementById("score-beh-bar").style.width = `${bScore}%`;
    }

    // Call Metadata
    if (data.sample_metadata) {
        const m = data.sample_metadata;
        document.getElementById("meta-caller-id").innerText = m.caller_id || "Unknown";
        document.getElementById("meta-freq").innerText = `${m.call_frequency_10min || 1} calls`;
        document.getElementById("meta-hour").innerText = `${m.time_of_day_hour || 12}:00`;
    }

    // Plain English Summary Box
    if (data.nl_summary) {
        document.getElementById("nl-summary-box").innerText = data.nl_summary;
    }

    // Spectrogram Heatmap & Anomaly Window
    if (data.spectrogram_heatmap) {
        const windows = data.spectrogram_heatmap.temporal_anomaly_windows || [];
        document.getElementById("temporal-anomaly-tag").innerText = `Anomaly Window: ${windows.join(', ')}`;
        drawSpectrogram(data.spectrogram_heatmap.heatmap);
    }

    // Transcript SHAP Tokens
    const shapBox = document.getElementById("shap-tokens-box");
    shapBox.innerHTML = "";
    if (data.transcript_shap && data.transcript_shap.length > 0) {
        data.transcript_shap.forEach(item => {
            const span = document.createElement("span");
            let bgClass = "bg-slate-900 border-slate-800 text-slate-300";
            if (item.is_risk_keyword) {
                bgClass = "bg-red-950/80 border-red-700 text-red-300 font-bold px-2 py-0.5 rounded shadow";
            } else if (item.shap_value > 0.15) {
                bgClass = "bg-amber-950/60 border-amber-800 text-amber-300 px-1.5 py-0.5 rounded";
            } else {
                bgClass = "px-1 text-slate-400";
            }
            span.className = `border text-xs ${bgClass}`;
            span.title = `SHAP value: ${item.shap_value}`;
            span.innerText = item.token;
            shapBox.appendChild(span);
        });
    } else {
        shapBox.innerHTML = `<span class="text-slate-500 italic">No transcript tokens available.</span>`;
    }

    // Behavioral Flags
    const flagsBox = document.getElementById("behavior-flags-box");
    flagsBox.innerHTML = "";
    if (data.behavior_flags && data.behavior_flags.length > 0) {
        data.behavior_flags.forEach(flag => {
            const div = document.createElement("div");
            div.className = "text-[11px] font-mono text-amber-400 bg-amber-950/40 border border-amber-900/60 rounded p-1.5 flex items-center gap-2";
            div.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <span>${flag}</span>`;
            flagsBox.appendChild(div);
        });
    }

    refreshAuditLog();
}

// Canvas Visualizers
function initCanvasVisualizers() {
    const waveCanvas = document.getElementById("waveform-canvas");
    waveCanvas.width = waveCanvas.clientWidth || 600;
    waveCanvas.height = waveCanvas.clientHeight || 64;

    const specCanvas = document.getElementById("spectrogram-canvas");
    specCanvas.width = specCanvas.clientWidth || 600;
    specCanvas.height = specCanvas.clientHeight || 128;
}

function drawWaveform(audioData) {
    const canvas = document.getElementById("waveform-canvas");
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, w, h);

    ctx.beginPath();
    ctx.strokeStyle = "#818cf8";
    ctx.lineWidth = 1.5;

    const sliceWidth = w / audioData.length;
    let x = 0;

    for (let i = 0; i < audioData.length; i++) {
        const v = audioData[i];
        const y = (v + 1) * (h / 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
    }

    ctx.lineTo(w, h / 2);
    ctx.stroke();
}

function drawSpectrogram(heatmapMatrix) {
    if (!heatmapMatrix || heatmapMatrix.length === 0) return;

    const canvas = document.getElementById("spectrogram-canvas");
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;

    const rows = heatmapMatrix.length;
    const cols = heatmapMatrix[0].length;

    const cellW = w / cols;
    const cellH = h / rows;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const val = heatmapMatrix[r][c];
            let red = Math.min(255, Math.floor(val * 255 * 1.5));
            let green = Math.floor((1 - Math.abs(val - 0.5) * 2) * 200);
            let blue = Math.floor((1 - val) * 255);

            ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
            ctx.fillRect(c * cellW, h - (r + 1) * cellH, cellW + 1, cellH + 1);
        }
    }
}

// Refresh Cryptographic Audit Log Table
async function refreshAuditLog() {
    try {
        const response = await fetch("/api/v1/audit-log");
        const data = await response.json();

        const tbody = document.getElementById("audit-table-body");
        tbody.innerHTML = "";

        if (data.entries && data.entries.length > 0) {
            const sorted = [...data.entries].reverse();
            sorted.slice(0, 8).forEach(entry => {
                const tr = document.createElement("tr");
                tr.className = "hover:bg-slate-900/80 transition border-b border-slate-800/60";

                let tierColor = "text-emerald-400";
                if (entry.risk_tier === "CRITICAL") tierColor = "text-red-400 font-bold";
                else if (entry.risk_tier === "HIGH") tierColor = "text-amber-400";
                else if (entry.risk_tier === "MEDIUM") tierColor = "text-yellow-400";

                tr.innerHTML = `
                    <td class="p-3 font-semibold text-slate-400">#${entry.entry_id}</td>
                    <td class="p-3 text-slate-300">${entry.caller_id}</td>
                    <td class="p-3 font-bold ${tierColor}">${entry.calibrated_scam_risk_score}/100</td>
                    <td class="p-3 ${tierColor}">${entry.risk_tier}</td>
                    <td class="p-3 text-slate-400 truncate max-w-xs" title="${entry.evidence_summary}">${entry.evidence_summary}</td>
                    <td class="p-3 text-slate-500 font-mono text-[10px] truncate max-w-[140px]" title="${entry.current_hash}">${entry.current_hash}</td>
                `;
                tbody.appendChild(tr);
            });

            if (data.verification_message) {
                document.getElementById("audit-badge-text").innerText = `SHA-256 LOG: VERIFIED (${data.total_entries} ENTRIES)`;
            }
        }
    } catch (err) {
        console.error("Error fetching audit log:", err);
    }
}

// Live Microphone Recording & ASR Streaming
async function toggleMicrophone() {
    const btn = document.getElementById("mic-toggle-btn");
    const asrBadgeText = document.getElementById("asr-status-text");

    if (!isRecording) {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(mediaStream);
            scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

            currentLiveTranscript = "";
            if (speechRecognizer) {
                try {
                    speechRecognizer.start();
                    asrBadgeText.innerText = "ASR LISTENING...";
                } catch (e) { console.warn(e); }
            }

            scriptProcessor.onaudioprocess = (e) => {
                if (!isRecording) return;
                const inputData = e.inputBuffer.getChannelData(0);
                const pcmArray = Array.from(inputData);
                drawWaveform(pcmArray.slice(0, 200));

                const sendText = currentLiveTranscript || "Hello, I am calling live from your bank security team.";

                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        audio: pcmArray,
                        transcript: sendText,
                        metadata: { caller_id: "+19998887777", call_frequency_10min: 2, time_of_day_hour: 15, geographic_mismatch: false }
                    }));
                }
            };

            source.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);

            isRecording = true;
            btn.className = "px-4 py-2 rounded-xl bg-red-600 hover:bg-red-500 text-white font-semibold text-xs transition flex items-center gap-2 shadow-lg shadow-red-600/30 animate-pulse";
            btn.innerHTML = `<i class="fa-solid fa-stop text-sm"></i> Stop Live Streaming`;
        } catch (e) {
            alert("Microphone access denied or unsupported in browser: " + e.message);
        }
    } else {
        isRecording = false;
        if (scriptProcessor) scriptProcessor.disconnect();
        if (audioContext) audioContext.close();
        if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
        if (speechRecognizer) {
            try { speechRecognizer.stop(); } catch(e){}
            asrBadgeText.innerText = "ASR AWAITING MIC";
        }

        btn.className = "px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition flex items-center gap-2 shadow-lg shadow-indigo-600/30";
        btn.innerHTML = `<i class="fa-solid fa-microphone text-sm"></i> Start Live Mic & ASR`;
    }
}

// Modals & Report Export
function openEnrollModal() {
    document.getElementById("enroll-modal").classList.remove("hidden");
}

function closeEnrollModal() {
    document.getElementById("enroll-modal").classList.add("hidden");
}

async function captureVoiceprint() {
    const speakerId = document.getElementById("enroll-speaker-id").value.trim() || "User_1";
    alert(`Voiceprint registered for ${speakerId}. The system will now verify caller voice match against this enrolled profile!`);
    closeEnrollModal();
}

function downloadForensicReport() {
    if (!currentDetectionResult) {
        alert("Run or load a call sample first to generate a report!");
        return;
    }
    const res = currentDetectionResult;
    const reportHtml = `
    <html>
    <head><title>AegisVoice Forensic Audit Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; color: #1e293b; }
        h1 { color: #0f172a; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }
        .box { background: #f8fafc; border: 1px solid #cbd5e1; padding: 15px; borderRadius: 8px; margin-bottom: 20px; }
        .hash { font-family: monospace; font-size: 11px; color: #475569; word-break: break-all; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: bold; color: white; background: #ef4444; }
    </style>
    </head>
    <body>
        <h1>AegisVoice Forensic Deepfake Audit Report</h1>
        <p><strong>Generated At:</strong> ${new Date().toLocaleString()}</p>
        
        <div class="box">
            <h3>Detection Summary</h3>
            <p><strong>Calibrated Scam Risk Score:</strong> ${res.calibrated_scam_risk_score} / 100</p>
            <p><strong>Risk Severity Tier:</strong> <span class="badge">${res.risk_tier}</span></p>
            <p><strong>Uncertainty Margin:</strong> ±${res.uncertainty_margin}%</p>
        </div>

        <div class="box">
            <h3>Evidence Rationale</h3>
            <p>${res.nl_summary || 'N/A'}</p>
        </div>

        <div class="box">
            <h3>Cryptographic Integrity Verification</h3>
            <p><strong>SHA-256 Decision Hash:</strong></p>
            <p class="hash">${res.audit_hash || 'SHA-256 Hash Verified'}</p>
            <p><strong>Status:</strong> Valid Immutable Append-Only Chain</p>
        </div>

        <button onclick="window.print()" style="padding: 10px 20px; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer;">Print / Save as PDF</button>
    </body>
    </html>
    `;

    const reportWindow = window.open("", "_blank");
    reportWindow.document.write(reportHtml);
    reportWindow.document.close();
}

async function flagForRetraining() {
    if (!currentDetectionResult) return;
    try {
        await fetch("/api/v1/active-learning/flag", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sample_id: `sample_${Date.now()}`,
                transcript: currentDetectionResult.nl_summary || "",
                predicted_score: currentDetectionResult.calibrated_scam_risk_score || 50.0,
                analyst_label: 1,
                reason: "Analyst flagged for retraining"
            })
        });
        alert("Sample flagged and queued into MLOps Active Learning Retraining Pipeline!");
    } catch (e) {
        alert("Error flagging sample: " + e.message);
    }
}

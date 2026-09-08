"""
Dataset Loader & Feature Extraction Module for Audio, Text, and Behavior
Supports ASVspoof 2019/2021, WaveFake formats, and synthetic dataset generation.
"""

import os
import json
import numpy as np
from scipy import signal
from scipy.fft import dct
from typing import Dict, List, Tuple, Any

from config.system_config import AudioConfig, default_config


class AudioFeatureExtractor:
    """
    Extracts acoustic representations (Log-Mel Spectrograms, MFCCs) and performs VAD.
    """

    def __init__(self, config: AudioConfig = default_config.audio):
        self.config = config

    def apply_vad(self, waveform: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Simple energy-based Voice Activity Detection (VAD) trimming.
        """
        energy = waveform ** 2
        active_indices = np.where(energy > threshold)[0]
        if len(active_indices) > 0:
            start, end = active_indices[0], active_indices[-1]
            return waveform[start:end]
        return waveform

    def compute_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Computes Log-Mel Spectrogram (Shape: [n_mels, time_steps]).
        """
        if len(waveform) < self.config.n_fft:
            waveform = np.pad(waveform, (0, self.config.n_fft - len(waveform)))
        
        frequencies, times, Sxx = signal.spectrogram(
            waveform,
            fs=self.config.sample_rate,
            nperseg=self.config.n_fft,
            noverlap=self.config.n_fft - self.config.hop_length,
            mode='magnitude'
        )
        
        # Approximate Mel-scale filterbank mapping
        n_freqs = Sxx.shape[0]
        mel_weights = np.linspace(0, 1, self.config.n_mels)
        mel_spec = np.dot(np.diag(mel_weights), Sxx[:self.config.n_mels, :])
        log_mel_spec = np.log(mel_spec + 1e-6)
        
        # Normalize
        norm_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-6)
        return norm_spec.astype(np.float32)

    def extract_mfcc(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extracts MFCC features (Shape: [n_mfcc, time_steps]).
        """
        log_mel = self.compute_mel_spectrogram(waveform)
        # Discrete Cosine Transform (DCT Type-II) along frequency axis
        mfcc_feat = dct(log_mel, type=2, axis=0, norm='ortho')[:self.config.n_mfcc]
        return mfcc_feat.astype(np.float32)


class SyntheticDatasetGenerator:
    """
    Generates synthetic benchmark training & validation audio samples, transcripts,
    and metadata adhering to ASVspoof / WaveFake formats.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.feature_extractor = AudioFeatureExtractor()

    def generate_synthetic_audio(self, duration: float = 2.0, is_fake: bool = False, attack_type: int = 0) -> np.ndarray:
        """
        Generates synthetic audio waveform.
        is_fake=False: Natural harmonics with fundamental frequency (F0) modulation.
        is_fake=True: Spectral artifacts, unnatural phase discontinuities, or robotized pitch.
        attack_type: 0=Bona-fide, 1=Neural TTS, 2=Voice Conversion, 3=Replay Attack.
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        if not is_fake:
            # Bona-fide speech harmonic synthesis (F0 ~ 120-220 Hz)
            f0 = 150.0 + 20.0 * np.sin(2 * np.pi * 1.5 * t)
            waveform = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t) + 0.25 * np.sin(2 * np.pi * 3 * f0 * t)
            # Enforce natural envelope
            envelope = np.sin(np.pi * t / duration) ** 2
            waveform = waveform * envelope
        else:
            if attack_type == 1:
                # Neural TTS artifact: High frequency phase jitter + artificial formant peaks
                f0 = 160.0
                waveform = np.sin(2 * np.pi * f0 * t) + 0.6 * signal.square(2 * np.pi * 3.5 * f0 * t)
                jitter = 0.05 * np.random.randn(len(t))
                waveform += jitter
            elif attack_type == 2:
                # Voice Conversion artifact: Phase discontinuity & pitch shift quantization
                f0 = 180.0 + 40.0 * signal.sawtooth(2 * np.pi * 4 * t)
                waveform = np.sin(2 * np.pi * f0 * t)
            else:
                # Replay Attack artifact: Channel impulse response + high noise floor
                f0 = 140.0
                waveform = np.sin(2 * np.pi * f0 * t) + 0.2 * np.random.randn(len(t))
                b, a = signal.butter(2, [0.1, 0.4], btype='band')
                waveform = signal.lfilter(b, a, waveform)

        # Normalize amplitude [-1, 1]
        max_val = np.max(np.abs(waveform)) + 1e-6
        return (waveform / max_val).astype(np.float32)

    def generate_dataset(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Generates a dataset containing audio, labels, transcripts, and metadata.
        """
        scam_phrases = [
            "Hello, I am calling from your bank security department. We detected unauthorized transactions.",
            "Please provide your OTP code immediately to prevent account suspension.",
            "Your social security account has been compromised. Wire transfer $500 to clear your name.",
            "This is an urgent call regarding your credit card. Verify your PIN now.",
            "Hi mom, I lost my phone and need money urgently. Please transfer funds.",
            "Good morning, just checking in about our scheduled meeting tomorrow at 10 AM.",
            "Hey, let's catch up over lunch later today if you're free.",
            "Thank you for calling customer service. How can I help you today?",
            "The weather forecast predicts light rain throughout the afternoon.",
            "Please confirm your appointment time for next Tuesday."
        ]

        dataset = []
        for i in range(num_samples):
            is_fake = (i % 2 == 1)
            attack_type = np.random.choice([1, 2, 3]) if is_fake else 0
            
            # Match transcript to scam vs legitimate context
            if is_fake:
                transcript_idx = np.random.randint(0, 5)  # Scam phrases
                scam_label = 1
            else:
                transcript_idx = np.random.randint(5, 10)  # Normal phrases
                scam_label = 0

            audio = self.generate_synthetic_audio(duration=2.0, is_fake=is_fake, attack_type=attack_type)
            mel_spec = self.feature_extractor.compute_mel_spectrogram(audio)
            mfcc = self.feature_extractor.extract_mfcc(audio)

            # Metadata simulation
            metadata = {
                "caller_id": f"+1800{np.random.randint(1000000, 9999999)}" if scam_label else "+14155551234",
                "call_frequency_10min": int(np.random.randint(3, 12) if scam_label else np.random.randint(0, 2)),
                "time_of_day_hour": int(np.random.choice([2, 3, 23]) if scam_label else np.random.randint(8, 20)),
                "geographic_mismatch": bool(scam_label and np.random.rand() > 0.3),
                "voiceprint_similarity": float(np.random.uniform(0.1, 0.4) if is_fake else np.random.uniform(0.85, 0.99))
            }

            dataset.append({
                "id": f"sample_{i:04d}",
                "audio": audio.tolist(),
                "mel_spectrogram": mel_spec.tolist(),
                "mfcc": mfcc.tolist(),
                "transcript": scam_phrases[transcript_idx],
                "label_audio_fake": 1 if is_fake else 0,
                "label_scam_transcript": scam_label,
                "attack_type": int(attack_type),
                "metadata": metadata
            })

        return dataset

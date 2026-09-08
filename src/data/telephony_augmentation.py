"""
Telephony Augmentation Pipeline for Realism in Audio Deepfake Detection
Simulates real telecom conditions: GSM/AMR codec compression, bandpass filtering,
room impulse response (RIR) reverberation, and additive babble noise.
"""

import numpy as np
from scipy import signal


class TelephonyAugmentor:
    """
    Applies realistic telephony artifacts to input audio waveforms.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def apply_bandpass_filter(self, waveform: np.ndarray, lowcut: float = 300.0, highcut: float = 3400.0) -> np.ndarray:
        """
        Applies Narrowband (300Hz-3400Hz) or Wideband telephony filtering.
        """
        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = min(highcut / nyquist, 0.99)
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.lfilter(b, a, waveform)
        return filtered.astype(np.float32)

    def apply_codec_compression(self, waveform: np.ndarray, bit_depth: int = 8) -> np.ndarray:
        """
        Simulates G.711 / GSM codec quantization loss (mu-law / A-law quantization).
        """
        # Mu-law companding simulation
        mu = 255.0
        x = np.clip(waveform, -1.0, 1.0)
        companded = np.sign(x) * np.log(1.0 + mu * np.abs(x)) / np.log(1.0 + mu)
        # Quantization
        quantized = np.round(companded * (2 ** (bit_depth - 1))) / (2 ** (bit_depth - 1))
        # De-companding
        expanded = np.sign(quantized) * (1.0 / mu) * ((1.0 + mu) ** np.abs(quantized) - 1.0)
        return expanded.astype(np.float32)

    def add_background_noise(self, waveform: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
        """
        Adds synthetic babble / ambient channel noise at specified Signal-to-Noise Ratio (SNR).
        """
        signal_power = np.mean(waveform ** 2)
        if signal_power == 0:
            return waveform
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), size=waveform.shape)
        noisy_signal = waveform + noise
        return np.clip(noisy_signal, -1.0, 1.0).astype(np.float32)

    def apply_reverberation(self, waveform: np.ndarray, rt60: float = 0.2) -> np.ndarray:
        """
        Simulates acoustic room impulse response (RIR).
        """
        decay_length = int(self.sample_rate * rt60)
        if decay_length <= 0:
            return waveform
        rir = np.exp(-3.0 * np.linspace(0, 1, decay_length))
        rir = rir / np.sum(rir)
        reverberated = np.convolve(waveform, rir, mode='same')
        return reverberated.astype(np.float32)

    def augment(self, waveform: np.ndarray, snr_db: float = 15.0, codec_bits: int = 8) -> np.ndarray:
        """
        Applies full telephony augmentation chain.
        """
        audio = self.apply_bandpass_filter(waveform)
        audio = self.apply_codec_compression(audio, bit_depth=codec_bits)
        audio = self.add_background_noise(audio, snr_db=snr_db)
        return audio

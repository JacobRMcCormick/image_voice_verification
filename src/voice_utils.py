"""
Shared voice helpers so enrollment and verification process audio identically.

Anything that changes the embedding (recording, trimming, normalizing, encoding)
lives here. If these differ between enroll and verify, similarity scores drop.
"""

import re

import numpy as np
import sounddevice as sd
import torch
from speechbrain.inference.speaker import EncoderClassifier

from tts import speak_blocking  # noqa: F401  (re-exported for enroll/verify)

torch.set_num_threads(4)

SAMPLE_RATE = 16000
EMBED_DIM = 192         
MIN_SPEECH_SEC = 0.4     


def record(seconds):
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def trim_speech(audio, sr=SAMPLE_RATE, frame_ms=30, pad_ms=150):
    """Energy-gate the clip down to the speech. Returns None if no speech found."""
    frame = int(sr * frame_ms / 1000)
    n = len(audio) // frame
    if n == 0:
        return None
    frames = audio[:n * frame].reshape(n, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    if rms.max() < 0.005:  
        return None
    thresh = max(rms.max() * 0.1, np.median(rms) * 3)
    idx = np.where(rms > thresh)[0]
    if len(idx) == 0:
        return None
    pad = pad_ms // frame_ms
    start = max(idx[0] - pad, 0) * frame
    end = min(idx[-1] + pad + 1, n) * frame
    trimmed = audio[start:end]
    if len(trimmed) < MIN_SPEECH_SEC * sr:
        return None
    return trimmed


def peak_normalize(audio, target=0.9):
    peak = np.abs(audio).max()
    return audio if peak < 1e-6 else (audio * (target / peak)).astype(np.float32)


def prepare(audio):
    """Raw recording -> trimmed, peak-normalized speech (or None)."""
    trimmed = trim_speech(audio)
    if trimmed is None:
        return None
    return peak_normalize(trimmed)


def load_encoder():
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )


def embed(encoder, audio):
    """Prepared audio -> L2-normalized (192,) float32 embedding."""
    tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        emb = encoder.encode_batch(tensor).squeeze().cpu().numpy().flatten()
    return (emb / np.linalg.norm(emb)).astype(np.float32)


def cosine_sim(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def normalize_text(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
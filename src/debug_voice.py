import sounddevice as sd
import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier
from scipy.spatial.distance import cosine

# 1. Initialize Engine
print("Loading SpeechBrain Model...")
encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
samplerate = 16000
duration = 4

print("\n--- TEST 1: SPEAK SAMPLES ---")
input("Press Enter and say 'Open Sesame' cleanly...")
audio1 = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
sd.wait()
audio1_flat = audio1.flatten()

input("Press Enter and say 'Open Sesame' exactly the same way...")
audio2 = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
sd.wait()
audio2_flat = audio2.flatten()

# Check for dead audio streams (silent microphones)
print(f"\nAudio 1 Max Amplitude: {np.max(np.abs(audio1_flat)):.4f}")
print(f"Audio 2 Max Amplitude: {np.max(np.abs(audio2_flat)):.4f}")

# Extract Vectors
t1 = torch.from_numpy(audio1_flat).unsqueeze(0)
t2 = torch.from_numpy(audio2_flat).unsqueeze(0)

with torch.no_grad():
    v1 = encoder.encode_batch(t1).squeeze().cpu().numpy()
    v2 = encoder.encode_batch(t2).squeeze().cpu().numpy()

# Mathematical analysis
similarity = 1.0 - cosine(v1, v2)
print(f"\nResulting Similarity: {similarity:.4f}")

print("\n--- TEST 2: CHECK IMPOSTOR ---")
input("Press Enter and have SOMEONE ELSE speak, or make a random clapping noise...")
audio3 = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
sd.wait()
audio3_flat = audio3.flatten()

t3 = torch.from_numpy(audio3_flat).unsqueeze(0)
with torch.no_grad():
    v3 = encoder.encode_batch(t3).squeeze().cpu().numpy()

impostor_sim = 1.0 - cosine(v1, v3)
print(f"Impostor/Noise Similarity: {impostor_sim:.4f}")

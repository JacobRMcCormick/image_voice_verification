"""Blocking text-to-speech whose wait ends exactly when the audio ends."""

import os
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np
import sounddevice as sd

_RENDER_CODE = """
import sys, pyttsx3
e = pyttsx3.init()
for v in e.getProperty("voices"):
    if any(k in v.name.lower() for k in ["natural", "zira", "david", "en-us"]):
        e.setProperty("voice", v.id)
        break
e.setProperty("rate", 175)
e.save_to_file(sys.argv[1], sys.argv[2])
e.runAndWait()
"""

_LIVE_CODE = """
import sys, pyttsx3
e = pyttsx3.init()
for v in e.getProperty("voices"):
    if any(k in v.name.lower() for k in ["natural", "zira", "david", "en-us"]):
        e.setProperty("voice", v.id)
        break
e.setProperty("rate", 175)
e.say(sys.argv[1])
e.runAndWait()
"""


def render(text, timeout=30):
    """
    Turn text into audio WITHOUT playing it. Returns (samples, sample_rate), or None
    on failure. Call this ahead of time so playing later has no delay.
    """
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run([sys.executable, "-c", _RENDER_CODE, text, path],
                       timeout=timeout, check=False)
        with wave.open(path, "rb") as w:
            if w.getsampwidth() != 2:
                raise ValueError(f"unsupported sample width {w.getsampwidth()}")
            rate, channels = w.getframerate(), w.getnchannels()
            frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            data = data.reshape(-1, channels)
        if data.size == 0:
            raise ValueError("empty audio")
        return data, rate
    except Exception as e:
        print(f"[tts] render failed: {e}")
        return None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def play(audio, tail=0.8):
    """Play pre-rendered audio and return only when it has finished, plus `tail` seconds."""
    data, rate = audio
    sd.play(data, rate)
    sd.wait()  
    time.sleep(tail)  


def speak_blocking(text, tail=0.8, timeout=30, fallback_sec_per_word=0.45):
    """
    Say `text` and return only once it has finished playing, so the mic never
    hears the TTS.
    """
    audio = render(text, timeout)
    if audio is not None:
        try:
            play(audio, tail)
            return
        except Exception as e:
            print(f"[tts] playback failed ({e}); falling back to live speech")

    start = time.time()
    try:
        subprocess.run([sys.executable, "-c", _LIVE_CODE, text], timeout=timeout, check=False)
    except Exception as e:
        print(f"[tts] skipped: {e}")
    remaining = len(text.split()) * fallback_sec_per_word - (time.time() - start)
    if remaining > 0:
        time.sleep(remaining)
    time.sleep(tail)
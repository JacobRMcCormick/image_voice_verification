import argparse
import csv
import os
import sys
from datetime import datetime

import cv2
import numpy as np
from faster_whisper import WhisperModel

from database import FaceDB
from voice_utils import (cosine_sim, embed, load_encoder, normalize_text,
                         prepare, record, speak_blocking)


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Voice & Passphrase Verification")
    parser.add_argument("--db", default="people.db", help="Path to SQLite database")
    parser.add_argument("--id", type=int, required=True, help="Database ID of the recognized person")
    parser.add_argument("--passphrase", default="open sesame", help="Required passphrase")
    parser.add_argument("--duration", type=float, default=4.0, help="Recording duration in seconds")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Minimum cosine similarity to accept (calibrate with the log!)")
    parser.add_argument("--log", default="verify_log.csv", help="CSV file where every score is appended")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Skip the spoken welcome (use when a wrapper already greeted the user)")
    parser.add_argument("--label", default="", choices=["", "genuine", "impostor"],
                        help="Tag this attempt in the log, to calibrate the threshold later")
    return parser.parse_args()


def draw_box(img, title, subtitle, box_color=(0, 140, 255)):
    """Draws a clean UI card on a neutral background."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (w // 8, h // 3), (7 * w // 8, 2 * h // 3), (30, 30, 30), -1)
    cv2.rectangle(img, (w // 8, h // 3), (7 * w // 8, 2 * h // 3), box_color, 3)
    cv2.putText(img, title, (w // 8 + 30, h // 3 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2, cv2.LINE_AA)
    cv2.putText(img, subtitle, (w // 8 + 30, h // 3 + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def show(window, title, subtitle, color=(0, 140, 255), wait=50):
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    draw_box(canvas, title, subtitle, color)
    cv2.imshow(window, canvas)
    cv2.waitKey(wait)


def log_attempt(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "id", "centroid_sim", "sample_mean_sim", "score",
                        "passphrase_ok", "granted", "label"])
        w.writerow(row)


def load_models():
    """
    Load and warm up both models. Call this early (e.g. at program start) so
    verify_voice() has no loading delay. Returns (encoder, whisper_model).
    """
    print("[verify] Loading AI models...")
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    encoder = load_encoder()

    try:
        noise = (np.random.randn(16000) * 0.01).astype(np.float32)
        embed(encoder, noise)
        segments, _ = whisper_model.transcribe(noise, beam_size=5, language="en")
        list(segments) 
    except Exception as e:
        print(f"[verify] Warm-up skipped: {e}")

    print("[verify] Models ready.")
    return encoder, whisper_model


def verify_voice(person_id, encoder, whisper_model, db_path="people.db",
                 passphrase="open sesame", duration=4.0, threshold=0.45,
                 log_path="verify_log.csv", label="", no_prompt=False):
    """Run one verification attempt with already-loaded models. Returns True if granted."""
    db = FaceDB(db_path)
    if person_id not in db.people:
        db.close()
        print(f"Error: ID {person_id} not found in database.")
        return False

    person = db.people[person_id]
    target_name = db.label(person_id)
    ref_centroid = person.get("voice_vec")
    ref_samples = person.get("voice_samples")
    db.close()

    if ref_centroid is None:
        print(f"[verify] Error: {target_name} has no voice embedding enrolled!")
        return False
    ref_centroid = np.asarray(ref_centroid, dtype=np.float32).flatten()

    window = f"Security Verification - {target_name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 360)
    show(window, f"WELCOME BACK, {target_name.upper()}", "Preparing...", wait=100)

    if not no_prompt:
        show(window, f"WELCOME BACK, {target_name.upper()}", "Listen to the prompt...")
        speak_blocking(f"Welcome back {target_name}, please speak the passphrase.")

    # 4. Record
    show(window, "LISTENING...", f"Say: '{passphrase}' into the microphone...", (0, 255, 255))
    print(f"[verify] Recording audio for {duration} seconds...")
    raw = record(duration)

    show(window, "PROCESSING...", "Analyzing voiceprint and transcription...", (255, 165, 0))

    peak = float(np.abs(raw).max())
    if peak >= 0.99:
        print("[verify] Warning: audio clipped. Move back from the mic or lower input gain.")
    elif peak < 0.05:
        print(f"[verify] Warning: recording is very quiet (peak {peak:.3f}).")

    speech = prepare(raw)

    centroid_sim = sample_sim = score = 0.0
    is_correct_speaker = False
    is_correct_passphrase = False
    spoken_text = ""

    if speech is None:
        print("[verify] No speech detected in the recording.")
    else:
        user_embed = embed(encoder, speech)
        centroid_sim = cosine_sim(ref_centroid, user_embed)
        if ref_samples is not None and len(ref_samples) > 0:
            sample_sim = float(np.mean([cosine_sim(s, user_embed) for s in ref_samples]))
        else:
            sample_sim = centroid_sim
        score = max(centroid_sim, sample_sim)
        is_correct_speaker = score >= threshold

        segments, _ = whisper_model.transcribe(raw, beam_size=5, language="en")
        spoken_text = normalize_text(" ".join(seg.text for seg in segments))
        is_correct_passphrase = normalize_text(passphrase) in spoken_text

    print(f"[verify] Speech: {0 if speech is None else len(speech) / 16000:.2f}s | "
          f"centroid: {centroid_sim:.4f} | per-sample mean: {sample_sim:.4f} | "
          f"score: {score:.4f} (threshold {threshold}) | Heard: '{spoken_text}'")

    success = is_correct_speaker and is_correct_passphrase

    log_attempt(log_path, [
        datetime.now().isoformat(timespec="seconds"), person_id,
        f"{centroid_sim:.4f}", f"{sample_sim:.4f}", f"{score:.4f}",
        is_correct_passphrase, success, label,
    ])

    if speech is None:
        result_msg, sub_msg = "ACCESS DENIED", "No speech detected"
    else:
        result_msg = "ACCESS GRANTED" if success else "ACCESS DENIED"
        sub_msg = f"Voice: {score:.2f} (need {threshold:.2f}) | Passphrase: {is_correct_passphrase}"
    show(window, result_msg, sub_msg, (0, 200, 0) if success else (0, 0, 255), wait=3500)
    cv2.destroyAllWindows()
    return success


def main():
    args = parse_args()
    encoder, whisper_model = load_models()
    ok = verify_voice(args.id, encoder, whisper_model, db_path=args.db,
                      passphrase=args.passphrase, duration=args.duration,
                      threshold=args.threshold, log_path=args.log,
                      label=args.label, no_prompt=args.no_prompt)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
import argparse
import sys
import time

import cv2
import numpy as np

from database import FaceDB
from voice_utils import (cosine_sim, embed, load_encoder, prepare, record,
                         speak_blocking)

MIN_GOOD_TAKES = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Voice Enrollment Tool")
    parser.add_argument("--db", default="people.db", help="Path to SQLite database")
    parser.add_argument("--id", type=int, required=True, help="Database ID of the person to enroll")
    parser.add_argument("--phrase", default="Open sesame.", help="Phrase to read each take")
    parser.add_argument("--takes", type=int, default=6, help="Number of short takes to record")
    parser.add_argument("--duration", type=float, default=3.0, help="Seconds recorded per take")
    return parser.parse_args()


def draw_box(img, title, subtitle, box_color=(0, 140, 255)):
    """Draws a clean UI card on a neutral background."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (w // 8, h // 3), (7 * w // 8, 2 * h // 3), (30, 30, 30), -1)
    cv2.rectangle(img, (w // 8, h // 3), (7 * w // 8, 2 * h // 3), box_color, 3)
    cv2.putText(img, title, (w // 8 + 30, h // 3 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2, cv2.LINE_AA)
    cv2.putText(img, subtitle, (w // 8 + 30, h // 3 + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def show(window, title, subtitle, color=(0, 140, 255), wait=50):
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    draw_box(canvas, title, subtitle, color)
    cv2.imshow(window, canvas)
    cv2.waitKey(wait)


def main():
    args = parse_args()

    db = FaceDB(args.db)
    if args.id not in db.people:
        db.close()
        sys.exit(f"Error: ID {args.id} not found in database. Please ensure the person record exists first.")
    target_name = db.label(args.id)
    db.close()

    window = f"Voice Enrollment - {target_name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 360)
    show(window, f"ENROLLING: {target_name.upper()}", "Loading voice model...", wait=100)

    print("[enroll] Loading SpeechBrain ECAPA-TDNN model onto CPU...")
    encoder = load_encoder()

    # Speak the instructions (without the phrase) and WAIT until finished
    show(window, f"ENROLLING: {target_name.upper()}", f"Phrase to say: '{args.phrase}'")
    speak_blocking(f"Hello {target_name}. When the screen turns yellow, "
                   f"say the phrase shown on screen.")

    embeddings = []
    attempts = 0
    max_attempts = args.takes * 2

    while len(embeddings) < args.takes and attempts < max_attempts:
        attempts += 1
        n = len(embeddings) + 1

        show(window, f"GET READY ({n}/{args.takes})", f"Next: '{args.phrase}'", (255, 165, 0), wait=800)
        show(window, f"SPEAK NOW ({n}/{args.takes})", f"Say: '{args.phrase}'", (0, 255, 255))

        print(f"[enroll] Take {n}/{args.takes} ({args.duration}s)...")
        raw = record(args.duration)
        speech = prepare(raw)

        if speech is None:
            print("[enroll] No clear speech detected, retrying this take.")
            show(window, "NOT HEARD", "Speak louder or closer to the mic", (0, 0, 255), wait=1200)
            continue

        print(f"[enroll]   speech length: {len(speech) / 16000:.2f}s")
        embeddings.append(embed(encoder, speech))
        time.sleep(0.3)

    if len(embeddings) < MIN_GOOD_TAKES:
        cv2.destroyAllWindows()
        sys.exit(f"Error: only {len(embeddings)} usable takes (need {MIN_GOOD_TAKES}). "
                 f"Check your microphone and try again.")

    show(window, "PROCESSING...", "Computing voice fingerprint...", (255, 165, 0))

    samples = np.stack(embeddings)
    centroid = samples.mean(axis=0)
    centroid = (centroid / np.linalg.norm(centroid)).astype(np.float32)

    # Consistency check: how well does each take agree with the centroid?
    sims = [cosine_sim(centroid, e) for e in samples]
    print("[enroll] Take-vs-centroid similarity: " + ", ".join(f"{s:.2f}" for s in sims))
    if min(sims) < 0.5:
        print("[enroll] Warning: at least one take is an outlier. Consider re-enrolling in a quieter room.")

    print(f"[enroll] Saving {centroid.shape[0]}-dim voice vector for ID {args.id} ({target_name})...")
    db = FaceDB(args.db)
    ok = db.update_voice_embedding(args.id, centroid, samples)
    db.close()

    if not ok:
        cv2.destroyAllWindows()
        sys.exit(f"Error: Failed to save voice embedding for ID {args.id}.")

    show(window, "ENROLLMENT SUCCESS", f"Voice profile saved for {target_name}!", (0, 200, 0), wait=3500)
    cv2.destroyAllWindows()
    print("[enroll] Done!")
    sys.exit(0)


if __name__ == "__main__":
    main()
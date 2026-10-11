import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from database import FaceDB
from identity import Recognizer, check_face_model, load_face_app

#should share params with detect_people

def resolve_device(choice: str) -> str:
    """Turn 'auto' / 'cpu' / 'gpu' / '0' / 'cuda:1' into a torch device string."""
    choice = choice.lower().strip()
    cuda_ok = torch.cuda.is_available()

    if choice == "auto":
        return "cuda:0" if cuda_ok else "cpu"
    if choice == "cpu":
        return "cpu"
    if choice in ("gpu", "cuda"):
        choice = "0"
    if choice.isdigit():
        choice = f"cuda:{choice}"

    if choice.startswith("cuda"):
        if not cuda_ok:
            print("[warn] GPU requested but CUDA is not available. Falling back to CPU.")
            return "cpu"
        return choice
    return choice

def parse_args():
    p = argparse.ArgumentParser(description="Check if a specific person is visible on a camera feed or video.")
    p.add_argument("source", nargs="?", default="0",
                   help="Path to image/video, or webcam index like 0 (default: 0)")
    p.add_argument("--db", default="people.db", help="Path to the SQLite database")
    p.add_argument("--name", help="Name of the person to check for")
    p.add_argument("--id", type=int, help="ID of the person to check for")
    p.add_argument("--device", default="auto", help="auto | cpu | gpu | index (default: auto)")
    p.add_argument("--model", default=None, help="YOLO weights file")
    p.add_argument("--imgsz", type=int, default=None, help="Inference size in pixels")
    p.add_argument("--vid-stride", type=int, default=None, help="Process every Nth frame")
    p.add_argument("--faces-only", action="store_true", help="Skip YOLO and only detect/recognize faces")
    p.add_argument("--face-model", default=None, help="InsightFace pack: buffalo_l or buffalo_s")
    p.add_argument("--face-det-size", type=int, default=None, help="Face detector input size")
    p.add_argument("--match-thresh", type=float, default=0.4, help="Cosine similarity threshold")
    p.add_argument("--min-face", type=int, default=40, help="Smallest face width in pixels")
    return p.parse_args()

def read_frames(source, stride):
    webcam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if webcam else source)
    if not cap.isOpened():
        sys.exit(f"Error: Could not open source {source}.")

    if webcam:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield frame
            i += 1
    finally:
        cap.release()

def main():
    args = parse_args()
    if not args.name and args.id is None:
        sys.exit("Error: You must provide either --name or --id to check for.")

    # Initialize Database
    db = FaceDB(args.db)
    check_face_model(db, args.face_model or ("buffalo_l" if torch.cuda.is_available() else "buffalo_s"))

    # Resolve target ID if name was provided
    target_id = args.id
    if args.name:
        found = False
        for pid, data in db.people.items():
            name_val = data.get("name") if isinstance(data, dict) else None
            if name_val and name_val.lower() == args.name.lower():
                target_id = pid
                found = True
                break
        if not found:
            sys.exit(f"Error: No person named '{args.name}' found in database '{args.db}'.")

    target_name = db.people.get(target_id, {}).get("name") or f"ID {target_id}"
    print(f"Looking for: {target_name} (ID: {target_id}) on source {args.source}...")

    # Resolve device and defaults
    device = resolve_device(args.device)
    on_gpu = device.startswith("cuda")
    half = on_gpu

    if args.model is None:
        args.model = "yolo11m.pt" if on_gpu else "yolo11n.pt"
    if args.imgsz is None:
        args.imgsz = 960 if on_gpu else 480
    if args.vid_stride is None:
        args.vid_stride = 1 if on_gpu else 2
    if args.face_model is None:
        args.face_model = "buffalo_l" if on_gpu else "buffalo_s"
    if args.face_det_size is None:
        args.face_det_size = 960 if on_gpu else 480

    # Setup models
    face_app = load_face_app(args.face_model, device, "video", args.face_det_size)
    recognizer = Recognizer(db, face_app, args.match_thresh, min_face=args.min_face, enroll=False)
    
    model = None if args.faces_only else YOLO(args.model)

    print("Press 'q' in the video window to quit.")
    cv2.namedWindow("Person Check", cv2.WINDOW_NORMAL)

    try:
        for frame in read_frames(args.source, args.vid_stride):
            # Process faces in the current frame
            faces = recognizer.process(frame)
            person_found = any(face.pid == target_id for face in faces)

            # Output True or False directly to the console per frame
            print(str(person_found).lower())

            # Optional visual overlay
            display_frame = frame.copy()
            for r in faces:
                x1, y1, x2, y2 = r.bbox
                color = (0, 200, 0) if r.pid == target_id else (160, 160, 160)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, r.label, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            status_text = f"Target ({target_name}): {'FOUND' if person_found else 'Not Found'}"
            color = (0, 255, 0) if person_found else (0, 0, 255)
            cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Person Check", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        db.close()

if __name__ == "__main__":
    main()
"""
Needs identity.py and database.py in the same folder.

Examples
--------
# Auto-detect everything (GPU if available, image/video from file extension)
python detect_people.py park-photo-1.jpg
python detect_people.py park-video.mp4

# Laptop (CPU), small and fast
python detect_people.py park-video.mp4 --device cpu --imgsz 640 --model yolo11n.pt

# Desktop (GPU), high quality on a single image
python detect_people.py park-photo-1.jpg --device gpu --imgsz 1280 --model yolo11x.pt

# Video with tracking (persistent IDs, recovers missed people)
python detect_people.py park-video.mp4 --track

# Webcam (index 0), mirrored like a selfie camera, at 1280x720
python detect_people.py --webcam --mirror --cam-size 1280x720

# Second webcam with face recognition
python detect_people.py --webcam 1 --db people.db

# Recognize faces and remember them in a database (new faces get a new ID)
python detect_people.py park-video.mp4 --db people.db

# Faces only (skips YOLO, faster), recognition without adding new people
python detect_people.py park-video.mp4 --db people.db --faces-only --no-enroll

# Give a saved person a name, or list everyone in the database
python detect_people.py --db people.db --rename 3 Alice
python detect_people.py --db people.db --list-db

# Fix duplicates: fold person 8 into person 7, or delete a person entirely
python detect_people.py --db people.db --merge 7 8
python detect_people.py --db people.db --delete-person 8

# Purge blurry entries from database below a sharpness threshold
python detect_people.py --db people.db --purge-blurred 100

# Show a saved person's best close-up, optionally exporting it to a file
python detect_people.py --db people.db --show-person 3
python detect_people.py --db people.db --show-person 3 --export-thumb alice.jpg
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import sqlite3

from database import FaceDB, show_person
from identity import Recognizer, check_face_model, load_face_app

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}

STATUS_COLORS = {"known": (0, 200, 0), "new": (0, 140, 255), "unknown": (160, 160, 160)}


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


def resolve_mode(source: str, mode: str) -> str:
    """Decide 'image' or 'video'. Webcam indices count as video."""
    if mode in ("image", "video"):
        return mode
    if source.isdigit():
        return "video"
    ext = Path(source).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    sys.exit(f"Can't tell if '{source}' is an image or video. Use --mode image|video.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detect people in an image or video with YOLO11 (CPU or GPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("source", nargs="?",
                   help="Path to image/video, or webcam index like 0")
    p.add_argument("--mode", choices=["auto", "image", "video"], default="auto",
                   help="Input type (default: auto-detect from file extension)")
    p.add_argument("--device", default="auto",
                   help="auto | cpu | gpu | GPU index (0, 1) | cuda:0 (default: auto)")
    p.add_argument("--imgsz", type=int, default=None,
                   help="Inference size in pixels. Default: 480 on CPU, 960 on GPU")
    p.add_argument("--model", default=None,
                   help="Weights, e.g. yolo11n.pt / yolo11m.pt / yolo11x.pt. "
                        "Default: yolo11n on CPU, yolo11m on GPU")
    p.add_argument("--conf", type=float, default=0.15,
                   help="Confidence threshold (default: 0.15)")
    p.add_argument("--iou", type=float, default=0.7,
                   help="NMS IoU threshold; higher keeps more overlapping people (default: 0.7)")
    p.add_argument("--all-classes", action="store_true",
                   help="Detect all COCO classes instead of people only")
    p.add_argument("--track", action="store_true",
                   help="Video only: use ByteTrack for persistent IDs")
    p.add_argument("--vid-stride", type=int, default=None,
                   help="Video only: process every Nth frame (default: 2 on CPU, 1 on GPU)")
    p.add_argument("--webcam", nargs="?", type=int, const=0, default=None, metavar="INDEX",
                   help="Use a webcam instead of a file (index defaults to 0)")
    p.add_argument("--cam-size", default=None, metavar="WxH",
                   help="Webcam only: requested resolution, e.g. 1280x720")
    p.add_argument("--mirror", action="store_true",
                   help="Webcam only: flip the picture horizontally")

    f = p.add_argument_group("face recognition")
    f.add_argument("--purge-low-conf", type=float, metavar="THRESHOLD",
                   help="Delete saved faces/people with detection confidence below this value and exit")
    f.add_argument("--purge-blurred", type=float, nargs="?", const=100.0, metavar="THRESHOLD",
                   help="Delete saved faces/people with thumbnail sharpness below this value and exit (default: 100.0)")
    f.add_argument("--db", default=None,
                   help="SQLite file of known faces. Giving this turns face recognition on")
    f.add_argument("--faces-only", action="store_true",
                   help="Skip YOLO and only detect/recognize faces (needs --db)")
    f.add_argument("--no-enroll", action="store_true",
                   help="Recognize known people but never add new ones")
    f.add_argument("--match-thresh", type=float, default=0.4,
                   help="Cosine similarity needed to count as the same person (default: 0.4)")
    f.add_argument("--min-face", type=int, default=40,
                   help="Smallest face width in pixels allowed to enroll or update (default: 40)")
    f.add_argument("--face-model", default=None,
                   help="InsightFace pack: buffalo_l (accurate) or buffalo_s (faster). Default: buffalo_s on CPU, buffalo_l on GPU")
    f.add_argument("--face-det-size", type=int, default=None,
                   help="Face detector input size, multiple of 32. Default: 480 on CPU, 960 on GPU")
    f.add_argument("--face-stride", type=int, default=None,
                   help="Run face recognition every Nth frame (default: 3 on CPU, 1 on GPU)")
    f.add_argument("--list-db", action="store_true",
                   help="Print everyone in --db and exit")
    f.add_argument("--rename", nargs=2, metavar=("ID", "NAME"),
                   help="Set the name of a saved person and exit")
    f.add_argument("--merge", nargs=2, type=int, metavar=("KEEP", "REMOVE"),
                   help="Merge duplicate person REMOVE into person KEEP and exit")
    f.add_argument("--delete-person", type=int, metavar="ID",
                   help="Delete a saved person and exit")
    f.add_argument("--show-person", type=int, metavar="ID",
                   help="Show the saved close-up of a person and exit")
    f.add_argument("--export-thumb", metavar="PATH",
                   help="With --show-person: also write the close-up to this file")
    return p


def draw_overlays(img, results):
    scale = max(max(img.shape[:2]) / 1000, 1.0)
    font = 0.6 * scale
    thick = max(2, int(round(2 * scale)))
    pad = int(4 * scale)
    for r in results:
        x1, y1, x2, y2 = r.bbox
        color = STATUS_COLORS[r.status]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        (tw, th), base = cv2.getTextSize(r.label, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        top = max(y1 - th - base - pad, 0)
        cv2.rectangle(img, (x1, top), (x1 + tw + 2 * pad, top + th + base + pad), color, -1)
        cv2.putText(img, r.label, (x1 + pad, top + th + pad // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), thick)


def print_faces(results):
    if not results:
        print("No faces detected. Try a larger --face-det-size (e.g. 640 or 960).")
        return
    print(f"Faces found: {len(results)}")
    for r in results:
        x1, y1, x2, y2 = r.bbox
        print(f"  {r.label:<16} face width {x2 - x1}px at ({x1}, {y1})")


def parse_size(text):
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        sys.exit("--cam-size must look like 1280x720")


def read_frames(source, mode, stride, size=None, mirror=False):
    if mode == "image":
        frame = cv2.imread(source)
        if frame is None:
            sys.exit(f"Could not read image: {source}")
        yield frame
        return

    webcam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if webcam else source)
    if not cap.isOpened():
        sys.exit(f"Could not open webcam {source}. Is it plugged in, not in use by another "
                 f"program, and the right index?" if webcam else f"Could not open video: {source}")

    if webcam:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Webcam {source}: {w}x{h}")

    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield cv2.flip(frame, 1) if (webcam and mirror) else frame
            i += 1
    finally:
        cap.release()


def frame_stream(model, args, mode, device, half):
    webcam = args.source.isdigit()
    if args.faces_only:
        for frame in read_frames(args.source, mode, args.vid_stride, args.cam_size, args.mirror):
            yield frame, frame, None
        return

    kwargs = dict(
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        classes=None if args.all_classes else [0],
        device=device,
        half=half,
        verbose=False,
    )

    if webcam:
        for frame in read_frames(args.source, mode, args.vid_stride, args.cam_size, args.mirror):
            if args.track:
                r = model.track(tracker="bytetrack.yaml", persist=True, **kwargs)[0]
            else:
                r = model.predict(frame, **kwargs)[0]
            yield r.plot(), r.orig_img, len(r.boxes)
        return

    kwargs.update(source=args.source, vid_stride=args.vid_stride, stream=True, save=False)
    if args.track and mode == "video":
        results = model.track(tracker="bytetrack.yaml", persist=True, **kwargs)
    else:
        results = model.predict(**kwargs)

    for r in results:
        yield r.plot(), r.orig_img, len(r.boxes)


def run(args, mode, model, device, half, recognizer, db):
    is_image = mode == "image"
    window = "Detections (press any key to close)" if is_image else "Detections (press q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frames, peak_people, peak_faces, faces = 0, 0, 0, []
    t0 = time.time()
    try:
        for shown, orig, count in frame_stream(model, args, mode, device, half):
            frames += 1
            if count is not None:
                peak_people = max(peak_people, count)

            if recognizer is not None:
                if (frames - 1) % args.face_stride == 0:
                    faces = recognizer.process(orig)
                    peak_faces = max(peak_faces, len(faces))
                    if is_image:
                        print_faces(faces)
                draw_overlays(shown, faces)
                if frames % 100 == 0:
                    db.save()

            if not is_image and frames % 30 == 0:
                fps = frames / (time.time() - t0)
                people = f"people: {count:3d} | " if count is not None else ""
                face_text = f"faces: {len(faces):3d} | " if recognizer is not None else ""
                print(f"frame {frames:5d} | {people}{face_text}{fps:5.1f} FPS")

            cv2.imshow(window, shown)
            key = cv2.waitKey(0 if is_image else 1) & 0xFF
            if is_image or key == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cv2.destroyAllWindows()
        if db is not None:
            db.close()

    elapsed = max(time.time() - t0, 1e-6)
    print(f"\nProcessed {frames} frame(s) in {elapsed:.1f}s ({frames / elapsed:.1f} FPS avg)")
    if not args.faces_only:
        print(f"Peak people in a single frame: {peak_people}")
    if recognizer is not None:
        print(f"Peak faces in a single frame: {peak_faces}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    utility = (args.list_db or args.rename or args.merge or
               args.delete_person is not None or args.show_person is not None or
               args.purge_low_conf is not None or args.purge_blurred is not None)
    if utility:
        if not args.db:
            parser.error("--list-db, --rename, --merge, --delete-person, "
                         "--purge-low-conf, --purge-blurred, and --show-person need --db PATH")
        db = FaceDB(args.db)
        if args.rename:
            pid, name = int(args.rename[0]), args.rename[1]
            if not db.rename(pid, name):
                sys.exit(f"No person with ID {pid} in {args.db}")
        if args.merge:
            keep, remove = args.merge
            if not db.merge(keep, remove):
                sys.exit("Both IDs must exist in the database and be different.")
            print(f"Merged ID {remove} into ID {keep}.")
        if args.delete_person is not None:
            if not db.delete(args.delete_person):
                sys.exit(f"No person with ID {args.delete_person} in {args.db}")
            print(f"Deleted ID {args.delete_person}.")
        if args.purge_low_conf is not None:
            conn = sqlite3.connect(args.db)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM people WHERE thumb_score < ?", (args.purge_low_conf,))
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            print(f"Purged {deleted} low-confidence entries below {args.purge_low_conf}.")
        if args.purge_blurred is not None:
            threshold = args.purge_blurred
            conn = sqlite3.connect(args.db)
            cursor = conn.cursor()
            cursor.execute("SELECT id, thumbnail FROM people")
            rows = cursor.fetchall()
            deleted_count = 0
            for pid, thumb_blob in rows:
                if thumb_blob is None:
                    continue
                arr = np.frombuffer(thumb_blob, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if sharpness < threshold:
                    cursor.execute("DELETE FROM people WHERE id = ?", (pid,))
                    deleted_count += 1
            conn.commit()
            conn.close()
            print(f"Purged {deleted_count} blurred entries below sharpness {threshold}.")
        elif args.show_person is not None:
            show_person(db, args.show_person, args.export_thumb)
        elif not args.purge_low_conf and not args.purge_blurred and not args.rename and not args.merge and args.delete_person is None:
            db.print_all()
            
        db.close()
        return

    if args.webcam is not None:
        if args.source:
            parser.error("give either a source file or --webcam, not both")
        args.source = str(args.webcam)
        args.mode = "video"
    args.cam_size = parse_size(args.cam_size) if args.cam_size else None

    if not args.source:
        parser.error("source is required (a file path, or --webcam)")
    if args.faces_only and not args.db:
        parser.error("--faces-only needs --db PATH")

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
    if args.face_stride is None:
        args.face_stride = 1 if on_gpu else 3

    mode = resolve_mode(args.source, args.mode)

    if mode == "image" and not Path(args.source).exists():
        sys.exit(f"File not found: {args.source}")
    if mode == "image" and args.track:
        print("[note] --track is ignored for images.")

    gpu_name = torch.cuda.get_device_name(device) if on_gpu else "n/a"
    print(f"Mode: {mode} | Device: {device} ({gpu_name}) | Model: "
          f"{'none (faces only)' if args.faces_only else args.model} | "
          f"imgsz: {args.imgsz} | FP16: {half}")

    model = None if args.faces_only else YOLO(args.model)

    recognizer, db = None, None
    if args.db:
        db = FaceDB(args.db)
        check_face_model(db, args.face_model)
        face_app = load_face_app(args.face_model, device, mode, args.face_det_size)
        recognizer = Recognizer(db, face_app, args.match_thresh, args.min_face,
                                enroll=not args.no_enroll)
        print(f"Face recognition on | Database: {args.db} ({len(db.people)} known people)")

    run(args, mode, model, device, half, recognizer, db)


if __name__ == "__main__":
    main()
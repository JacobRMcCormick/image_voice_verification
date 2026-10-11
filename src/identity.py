import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from database import make_thumbnail

MIN_DET_SCORE = 0.6
MIN_SHARPNESS = 35.0  
UPDATE_MARGIN = 0.1
MAX_SAMPLES = 100


@dataclass
class FaceResult:
    bbox: tuple
    label: str
    status: str  
    pid: Optional[int] = None
    similarity: float = 0.0


def compute_sharpness(frame, bbox) -> float:
    """Calculate the variance of the Laplacian on the face crop as a measure of sharpness."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_face_model(db, model_name):
    saved = db.get_meta("face_model")
    if saved is None:
        assumed = model_name if not db.people else "buffalo_l"
        db.set_meta("face_model", assumed)
        if db.people:
            print(f"[note] This database didn't record its face model. Assuming '{assumed}'.")
        saved = assumed
    if saved != model_name:
        sys.exit(
            f"This database was built with face model '{saved}', but you asked for "
            f"'{model_name}'. Embeddings from different models can't be compared, so "
            f"every face would be saved as a new person. Use --face-model {saved}, or "
            f"start a new database."
        )


def load_face_app(model_name, device, mode, det_size=None):
    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
    except ImportError:
        sys.exit("Face recognition needs: pip install insightface onnxruntime "
                 "(onnxruntime-gpu on the GPU machine)")

    wants_gpu = device.startswith("cuda")
    has_gpu = "CUDAExecutionProvider" in ort.get_available_providers()
    if wants_gpu and not has_gpu:
        print("[warn] onnxruntime-gpu not installed; face recognition will run on CPU.")
    use_gpu = wants_gpu and has_gpu

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    ctx_id = int(device.split(":")[1]) if use_gpu and ":" in device else (0 if use_gpu else -1)
    if mode == "image":
        default_size = 1280 if use_gpu else 960
    else:
        default_size = 960 if use_gpu else 640
    size = det_size or default_size

    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(size, size))
    return app


class Recognizer:
    def __init__(self, db, face_app, match_thresh=0.4, min_face=40, enroll=True):
        self.db = db
        self.face_app = face_app
        self.match_thresh = match_thresh
        self.min_face = min_face
        self.enroll = enroll

    def best_candidate(self, vec):
        best_id, best_sim = None, -1.0
        for pid, p in self.db.people.items():
            sim = float(vec @ p["vec"])
            if sim > best_sim:
                best_id, best_sim = pid, sim
        return best_id, best_sim

    def process(self, frame):
        results = []
        for face in self.face_app.get(frame):
            bbox = tuple(int(v) for v in face.bbox)
            width = bbox[2] - bbox[0]
            vec = face.normed_embedding
            
            sharpness = compute_sharpness(frame, bbox)
            good = (face.det_score >= MIN_DET_SCORE and 
                    width >= self.min_face and 
                    sharpness >= MIN_SHARPNESS)
            
            quality = float(face.det_score) * min(width, 200) / 200

            best_id, sim = self.best_candidate(vec)
            pid = best_id if sim >= self.match_thresh else None
            if pid is not None:
                if good and sim >= self.match_thresh + UPDATE_MARGIN:
                    self._learn(pid, vec)
                    self._maybe_upgrade_thumbnail(pid, frame, bbox, quality)
                else:
                    self.db.seen(pid)
                label = f"{self.db.label(pid)} {sim:.2f}"
                results.append(FaceResult(bbox, label, "known", pid, sim))
            elif good and self.enroll:
                pid = self.db.add(vec, make_thumbnail(frame, bbox), quality)
                note = f" ({sim:.2f} vs {self.db.label(best_id)})" if best_id is not None else ""
                results.append(FaceResult(bbox, f"{self.db.label(pid)} NEW{note}", "new", pid,
                                          sim if best_id is not None else 0.0))
            else:
                results.append(FaceResult(bbox, "unknown", "unknown"))
        return results

    def _learn(self, pid, vec):
        p = self.db.people[pid]
        n = min(p["n"], MAX_SAMPLES)
        v = (p["vec"] * n + vec) / (n + 1)
        self.db.update_embedding(pid, v / np.linalg.norm(v))

    def _maybe_upgrade_thumbnail(self, pid, frame, bbox, quality):
        if quality <= self.db.thumbnail_quality(pid):
            return
        jpg = make_thumbnail(frame, bbox)
        if jpg is not None:
            self.db.set_thumbnail(pid, jpg, quality)
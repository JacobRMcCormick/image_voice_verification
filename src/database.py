"""
Database calls for known people: face embeddings, voice embeddings, names, timestamps and thumbnails.

Storage only. Deciding who someone is lives in identity.py.
"""

import sqlite3
from datetime import datetime

import cv2
import numpy as np

VOICE_DIM = 192  


def make_thumbnail(frame, bbox, margin=0.4, max_side=256):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    pad_x, pad_y = int((x2 - x1) * margin), int((y2 - y1) * margin)
    crop = frame[max(y1 - pad_y, 0):min(y2 + pad_y, h), max(x1 - pad_x, 0):min(x2 + pad_x, w)]
    if crop.size == 0:
        return None
    side = max(crop.shape[:2])
    if side > max_side:
        s = max_side / side
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes() if ok else None


def _to_voice_vec(blob):
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy().flatten()


def _to_voice_samples(blob):
    """Stored as flat float32; reshape to (k, VOICE_DIM)."""
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float32).copy()
    if arr.size == 0 or arr.size % VOICE_DIM != 0:
        return None
    return arr.reshape(-1, VOICE_DIM)


class FaceDB:
    def __init__(self, path):
        self.con = sqlite3.connect(path)

        self.con.execute(
            "CREATE TABLE IF NOT EXISTS people ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, embedding BLOB, "
            "voice_embedding BLOB, voice_samples BLOB, "
            "samples INTEGER, first_seen TEXT, last_seen TEXT, "
            "thumbnail BLOB, thumb_score REAL DEFAULT 0)"
        )
        self.con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")

        # Migrations for older databases
        cols = {row[1] for row in self.con.execute("PRAGMA table_info(people)")}
        if "thumbnail" not in cols:
            self.con.execute("ALTER TABLE people ADD COLUMN thumbnail BLOB")
        if "thumb_score" not in cols:
            self.con.execute("ALTER TABLE people ADD COLUMN thumb_score REAL DEFAULT 0")
        if "voice_embedding" not in cols:
            self.con.execute("ALTER TABLE people ADD COLUMN voice_embedding BLOB")
        if "voice_samples" not in cols:
            self.con.execute("ALTER TABLE people ADD COLUMN voice_samples BLOB")
        self.con.commit()

        self.people = {}
        rows = self.con.execute(
            "SELECT id, name, embedding, voice_embedding, voice_samples, samples, "
            "COALESCE(thumb_score, 0) FROM people"
        )
        for pid, name, blob, voice_blob, samples_blob, n, thumb_score in rows:
            vec = np.frombuffer(blob, dtype=np.float32).copy() if blob else None
            voice_vec = _to_voice_vec(voice_blob)
            if voice_vec is not None and len(voice_vec) != VOICE_DIM:
                print(f"[Warning] ID {pid} voice vector has length {len(voice_vec)}, "
                      f"expected {VOICE_DIM}. Re-enroll this person's voice.")
            self.people[pid] = {
                "name": name,
                "vec": vec,
                "voice_vec": voice_vec,
                "voice_samples": _to_voice_samples(samples_blob),
                "n": n,
                "thumb_score": thumb_score,
            }
        self.dirty = set()

    @staticmethod
    def _now():
        return datetime.now().isoformat(timespec="seconds")

    def label(self, pid):
        return self.people[pid]["name"] or f"ID {pid}"

    def add(self, vec, thumb=None, quality=0.0, voice_vec=None):
        vec = vec.astype(np.float32) if vec is not None else None
        voice_vec = voice_vec.astype(np.float32).flatten() if voice_vec is not None else None
        now = self._now()
        score = quality if thumb else 0.0

        cur = self.con.execute(
            "INSERT INTO people (name, embedding, voice_embedding, samples, first_seen, last_seen, "
            "thumbnail, thumb_score) VALUES (NULL, ?, ?, 1, ?, ?, ?, ?)",
            (vec.tobytes() if vec is not None else None,
             voice_vec.tobytes() if voice_vec is not None else None,
             now, now, thumb, score),
        )
        self.con.commit()
        pid = cur.lastrowid
        self.people[pid] = {
            "name": None,
            "vec": vec.copy() if vec is not None else None,
            "voice_vec": voice_vec.copy() if voice_vec is not None else None,
            "voice_samples": None,
            "n": 1,
            "thumb_score": score,
        }
        return pid

    def update_embedding(self, pid, vec):
        p = self.people[pid]
        p["vec"] = vec.astype(np.float32)
        p["n"] += 1
        self.dirty.add(pid)

    def update_voice_embedding(self, pid, voice_vec, voice_samples=None):
        """
        Enroll or replace the voice profile.
        voice_vec: (192,) centroid. voice_samples: optional (k, 192) individual takes.
        Written straight to the DB so it doesn't depend on save().
        """
        if pid not in self.people:
            return False

        voice_vec = np.asarray(voice_vec, dtype=np.float32).flatten()
        if len(voice_vec) != VOICE_DIM:
            print(f"[Warning] Voice vector length {len(voice_vec)} != {VOICE_DIM}")

        samples = None
        sample_bytes = None
        if voice_samples is not None:
            samples = np.asarray(voice_samples, dtype=np.float32).reshape(-1, voice_vec.size)
            sample_bytes = samples.tobytes()

        self.con.execute(
            "UPDATE people SET voice_embedding=?, voice_samples=? WHERE id=?",
            (voice_vec.tobytes(), sample_bytes, pid),
        )
        self.con.commit()

        self.people[pid]["voice_vec"] = voice_vec
        self.people[pid]["voice_samples"] = samples
        return True

    def seen(self, pid):
        self.dirty.add(pid)

    def thumbnail_quality(self, pid):
        return self.people[pid]["thumb_score"]

    def set_thumbnail(self, pid, jpg, quality):
        self.people[pid]["thumb_score"] = quality
        self.con.execute(
            "UPDATE people SET thumbnail=?, thumb_score=? WHERE id=?", (jpg, quality, pid)
        )
        self.dirty.add(pid)

    def get_thumbnail(self, pid):
        row = self.con.execute("SELECT thumbnail FROM people WHERE id=?", (pid,)).fetchone()
        if not row or row[0] is None:
            return None
        return cv2.imdecode(np.frombuffer(row[0], np.uint8), cv2.IMREAD_COLOR)

    def save(self):
        """Persist face embedding, sample count and last_seen for people marked dirty."""
        now = self._now()
        for pid in self.dirty:
            p = self.people[pid]
            vec_bytes = p["vec"].tobytes() if p["vec"] is not None else None
            # COALESCE keeps whatever is already stored if memory has no value
            self.con.execute(
                "UPDATE people SET embedding=COALESCE(?, embedding), "
                "voice_embedding=COALESCE(?, voice_embedding), samples=?, last_seen=? WHERE id=?",
                (vec_bytes,
                 p["voice_vec"].tobytes() if p["voice_vec"] is not None else None,
                 p["n"], now, pid),
            )
        self.con.commit()
        self.dirty.clear()

    def rename(self, pid, name):
        if pid not in self.people:
            return False
        self.people[pid]["name"] = name
        self.con.execute("UPDATE people SET name=? WHERE id=?", (name, pid))
        self.con.commit()
        return True

    def get_meta(self, key):
        row = self.con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key, value):
        self.con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        self.con.commit()

    def delete(self, pid):
        if pid not in self.people:
            return False
        self.con.execute("DELETE FROM people WHERE id=?", (pid,))
        self.con.commit()
        del self.people[pid]
        self.dirty.discard(pid)
        return True

    def close(self):
        if self.con:
            self.con.close()
            self.con = None

    def merge(self, keep, remove):
        if keep == remove or keep not in self.people or remove not in self.people:
            return False
        a, b = self.people[keep], self.people[remove]

        if a["vec"] is not None and b["vec"] is not None:
            v = a["vec"] * a["n"] + b["vec"] * b["n"]
            a["vec"] = (v / np.linalg.norm(v)).astype(np.float32)
        elif a["vec"] is None:
            a["vec"] = b["vec"]

        sa, sb = a["voice_samples"], b["voice_samples"]
        if sa is not None and sb is not None:
            a["voice_samples"] = np.vstack([sa, sb])
            v_voice = a["voice_samples"].mean(axis=0)
            a["voice_vec"] = (v_voice / np.linalg.norm(v_voice)).astype(np.float32)
        elif a["voice_vec"] is None and b["voice_vec"] is not None:
            a["voice_vec"] = b["voice_vec"]
            a["voice_samples"] = sb
        elif a["voice_vec"] is not None and b["voice_vec"] is not None:
            v_voice = a["voice_vec"] + b["voice_vec"]
            a["voice_vec"] = (v_voice / np.linalg.norm(v_voice)).astype(np.float32)

        a["n"] += b["n"]
        if a["name"] is None and b["name"]:
            a["name"] = b["name"]

        rows = {r[0]: r for r in self.con.execute(
            "SELECT id, first_seen FROM people WHERE id IN (?, ?)", (keep, remove))}

        self.con.execute(
            "UPDATE people SET name=?, embedding=?, voice_embedding=?, voice_samples=?, "
            "samples=?, first_seen=?, last_seen=? WHERE id=?",
            (a["name"],
             a["vec"].tobytes() if a["vec"] is not None else None,
             a["voice_vec"].tobytes() if a["voice_vec"] is not None else None,
             a["voice_samples"].tobytes() if a["voice_samples"] is not None else None,
             a["n"],
             min(rows[keep][1], rows[remove][1]), self._now(), keep),
        )
        self.con.commit()
        return True
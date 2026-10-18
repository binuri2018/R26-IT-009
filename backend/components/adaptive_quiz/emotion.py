"""YOLO inference for webcam frames (``backend/model/best.pt``).

- **Classify** checkpoints: uses top-1 ``probs``.
- **Detect** checkpoints: area-weighted per-class scores with a smile-friendly uplift.

Ported from the standalone Adaptive-quiz project; config now comes from
``backend.common.config.settings``.
"""
from __future__ import annotations

import base64
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from backend.common.config import settings

log = logging.getLogger(__name__)

SUPPORTED_EMOTIONS = ["neutral", "happy", "confused", "frustrated", "surprised", "unknown"]

_RAW_TO_EXPRESSION: dict[str, str] = {
    "neutral": "neutral",
    "calm": "neutral",
    "positive": "happy",
    "positivity": "happy",
    "negative": "frustrated",
    "negativity": "frustrated",
    "happy": "happy",
    "smile": "happy",
    "happiness": "happy",
    "joy": "happy",
    "sad": "confused",
    "sadness": "confused",
    "fear": "confused",
    "fearful": "confused",
    "angry": "frustrated",
    "anger": "frustrated",
    "disgust": "frustrated",
    "confused": "confused",
    "confusion": "confused",
    "frustrated": "frustrated",
    "frustration": "frustrated",
    "surprise": "surprised",
    "surprised": "surprised",
    "unknown": "unknown",
}

_model: Any = None
_model_lock = threading.Lock()
_load_error: str | None = None

# This 3-class detector is light (~25-30 ms/frame on CPU at imgsz=256). Run it on a
# dedicated single worker so a burst of webcam frames can never starve the pool used
# by /predict and Mongo, and so frames are analysed one at a time (single-flight).
emotion_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="emotion")


def _merged_label_map() -> dict[str, str]:
    m = dict(_RAW_TO_EXPRESSION)
    extra_json = (settings.emotion_label_map or "").strip()
    if extra_json:
        try:
            extra = json.loads(extra_json)
            if isinstance(extra, dict):
                m.update({str(k).lower().strip(): str(v).strip() for k, v in extra.items()})
        except json.JSONDecodeError as e:
            log.warning("emotion_label_map is not valid JSON: %s", e)
    return m


def _normalize_key(label: str) -> str:
    return label.lower().strip().replace(" ", "_").replace("-", "_")


def map_to_expression(raw: str, label_map: dict[str, str]) -> str:
    key = _normalize_key(str(raw))
    out = label_map.get(key, label_map.get(str(raw).lower().strip(), "neutral"))
    if out not in SUPPORTED_EMOTIONS:
        return "neutral"
    return out


def _detect_pick_class(
    names: dict[Any, Any],
    confs: np.ndarray,
    clss: np.ndarray,
    xyxy: np.ndarray,
    label_map: dict[str, str],
) -> tuple[int, float]:
    confs = np.asarray(confs, dtype=np.float64).reshape(-1)
    clss = np.asarray(clss, dtype=np.int64).reshape(-1)
    n = len(clss)
    if n == 0:
        return -1, 0.0

    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    areas = np.maximum(w * h, 1e-6)
    max_area = float(np.max(areas))
    geo = np.sqrt(areas / max_area)

    scores_by_class: dict[int, float] = {}
    conf_by_class: dict[int, float] = {}
    for j in range(n):
        cid = int(clss[j])
        c = float(confs[j])
        wscore = float(geo[j]) * c
        if wscore > scores_by_class.get(cid, -1.0):
            scores_by_class[cid] = wscore
            conf_by_class[cid] = c

    def raw_name(cid: int) -> str:
        raw = names.get(cid, names.get(str(cid), str(cid)))
        if isinstance(raw, (list, tuple)) and raw:
            raw = raw[0]
        return str(raw)

    # Valence priors: this 3-class detector (positive/neutral/negative) under-fires
    # smiles and frowns relative to the large "neutral face" box. Lift both affects
    # so happy/frustrated can actually win, then only demote a weak negative.
    _PRIOR = {"happy": 1.28, "frustrated": 1.12, "neutral": 1.0, "confused": 1.0, "surprised": 1.0}

    def ranked(cid: int) -> float:
        expr = map_to_expression(raw_name(cid), label_map)
        return scores_by_class[cid] * _PRIOR.get(expr, 1.0)

    best_cid = max(scores_by_class, key=ranked)
    best_score = scores_by_class[best_cid]

    happy_ids = [
        cid for cid in scores_by_class
        if map_to_expression(raw_name(cid), label_map) == "happy"
    ]
    if happy_ids:
        best_happy = max(happy_ids, key=lambda c: scores_by_class[c])
        hs = scores_by_class[best_happy]
        if hs >= best_score * 0.42:
            best_cid = best_happy

    if map_to_expression(raw_name(best_cid), label_map) == "frustrated":
        frustrated_conf = conf_by_class.get(best_cid, 0.0)
        neutral_ids = [
            cid for cid in scores_by_class
            if map_to_expression(raw_name(cid), label_map) == "neutral"
        ]
        if neutral_ids and frustrated_conf < 0.22:
            best_neutral = max(neutral_ids, key=lambda c: scores_by_class[c])
            if scores_by_class[best_neutral] >= scores_by_class[best_cid] * 0.95:
                best_cid = best_neutral

    return best_cid, float(conf_by_class.get(best_cid, 0.0))


def _resolve_model_path() -> Path:
    return settings.resolved_emotion_model_path()


def get_model():
    """Lazy-load YOLO model (thread-safe). Returns None if weights/deps are missing."""
    global _model, _load_error
    with _model_lock:
        if _model is not None:
            return _model
        if _load_error is not None:
            return None

        path = _resolve_model_path()
        if not path.is_file():
            log.warning("Emotion model not found: %s — place best.pt here or set EMOTION_MODEL_PATH", path)
            return None

        try:
            from ultralytics import YOLO

            _model = YOLO(str(path))
            log.info("Loaded emotion model from %s (task=%s)", path, getattr(_model, "task", "?"))
            return _model
        except Exception as e:  # noqa: BLE001
            _load_error = str(e)
            log.exception("Failed to load YOLO model: %s", e)
            return None


_face_detector: Any = None
_face_load_error: str | None = None


def _get_face_detector():
    """Lazy-load OpenCV YuNet. Returns None (gating disabled) if the model file is missing."""
    global _face_detector, _face_load_error
    with _model_lock:
        if _face_detector is not None or _face_load_error is not None:
            return _face_detector
        path = settings.resolved_face_model_path()
        if not path.is_file():
            _face_load_error = f"Face model not found: {path}"
            log.warning("%s — emotion results are NOT face-gated", _face_load_error)
            return None
        try:
            import cv2

            _face_detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), settings.face_min_score)
            log.info("Loaded face detector from %s", path)
        except Exception as e:  # noqa: BLE001
            _face_load_error = str(e)
            log.exception("Failed to load face detector: %s", e)
        return _face_detector


def _find_face(bgr: np.ndarray) -> np.ndarray | None:
    """Largest detected face as [x1, y1, x2, y2], or None when nobody is in front of the camera."""
    detector = _get_face_detector()
    h, w = bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(bgr)
    if faces is None or not len(faces):
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])[:4]
    return np.array([x, y, x + fw, y + fh], dtype=np.float64)


def _on_face(xyxy: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Mask of expression boxes that belong to the detected face (centre on the face, padded 30%)."""
    fx1, fy1, fx2, fy2 = face
    pad_x, pad_y = (fx2 - fx1) * 0.3, (fy2 - fy1) * 0.3
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
    return (cx >= fx1 - pad_x) & (cx <= fx2 + pad_x) & (cy >= fy1 - pad_y) & (cy <= fy2 + pad_y)


def _gate_expression(expr: str, conf: float) -> tuple[str, float]:
    """Keep a weak read as neutral; allow typical webcam valence confidences through."""
    if expr == "neutral":
        return expr, conf
    floor = settings.emotion_negative_min_conf if expr == "frustrated" else settings.emotion_min_conf
    if conf < floor:
        return "neutral", conf
    return expr, conf


def analyze_frame(frame_b64: str) -> tuple[str, float]:
    """Decode a base64 JPEG frame, run YOLO, return ``(expression, confidence)``."""
    label_map = _merged_label_map()
    if not frame_b64:
        return "neutral", 0.0

    model = get_model()
    if model is None:
        return "neutral", 0.35

    try:
        import cv2

        blob = base64.b64decode(frame_b64)
        arr = np.frombuffer(blob, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return "neutral", 0.0

        # No face in front of the camera → no emotion at all (the expression model
        # alone happily "finds" frustrated/happy boxes in an empty room).
        gated = _get_face_detector() is not None
        face = _find_face(bgr) if gated else None
        if gated and face is None:
            if settings.emotion_debug:
                log.warning("emotion-debug no face -> unknown")
            return "unknown", 0.0

        predict_kw: dict[str, Any] = dict(
            source=bgr, verbose=False, imgsz=settings.emotion_predict_imgsz
        )
        if getattr(model, "task", None) == "detect":
            predict_kw["conf"] = settings.emotion_yolo_conf

        results = model.predict(**predict_kw)
        if not results:
            return "neutral", 0.0

        r = results[0]
        probs = getattr(r, "probs", None)
        if probs is not None:
            top1 = int(probs.top1)
            conf = float(probs.top1conf)
            names = getattr(model, "names", {}) or {}
            raw = names.get(top1, str(top1))
            if isinstance(raw, (list, tuple)) and raw:
                raw = raw[0]
            expr = map_to_expression(str(raw), label_map)
            return _gate_expression(expr, round(min(max(conf, 0.0), 1.0), 4))

        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes):
            import torch

            confs, clss, xyxy = boxes.conf, boxes.cls, boxes.xyxy
            if isinstance(confs, torch.Tensor):
                confs = confs.cpu().numpy()
            if isinstance(clss, torch.Tensor):
                clss = clss.cpu().numpy()
            if isinstance(xyxy, torch.Tensor):
                xyxy = xyxy.cpu().numpy()
            confs = np.asarray(confs, dtype=np.float64).reshape(-1)
            clss = np.asarray(clss, dtype=np.int64).reshape(-1)
            xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)

            names = getattr(model, "names", {}) or {}
            all_boxes = (confs, clss, xyxy)
            if face is not None:                    # ignore boxes on the background
                keep = _on_face(xyxy, face)
                confs, clss, xyxy = confs[keep], clss[keep], xyxy[keep]
            cid, conf = _detect_pick_class(names, confs, clss, xyxy, label_map)
            if cid < 0:
                result = ("neutral", 0.0)           # face visible, no confident expression
            else:
                raw = names.get(cid, names.get(str(cid), str(cid)))
                if isinstance(raw, (list, tuple)) and raw:
                    raw = raw[0]
                expr = map_to_expression(str(raw), label_map)
                result = _gate_expression(expr, round(min(max(conf, 0.0), 1.0), 4))
            if settings.emotion_debug:
                _log_debug(names, all_boxes, int(len(confs)), result)
            return result

        if settings.emotion_debug:
            log.warning("emotion-debug face, no boxes -> neutral")
        return "neutral", 0.0
    except Exception:  # noqa: BLE001
        log.exception("Emotion inference failed")
        return "neutral", 0.0


def _log_debug(names, all_boxes, kept: int, result) -> None:
    """Numbers only (never images): every box's label/confidence and how many sat on the face."""
    confs, clss, _ = all_boxes
    boxes = [(str(names.get(int(c), c)), round(float(p), 2)) for c, p in zip(clss, confs)]
    log.warning("emotion-debug boxes=%s on_face=%d -> %s", boxes, kept, result)


def emotion_model_status() -> dict[str, Any]:
    path = _resolve_model_path()
    return {
        "path": str(path),
        "exists": path.is_file(),
        "loaded": _model is not None,
        "error": _load_error,
        "backend": "ultralytics_yolo",
        "faceGate": {
            "path": str(settings.resolved_face_model_path()),
            "loaded": _face_detector is not None,
            "error": _face_load_error,
        },
    }

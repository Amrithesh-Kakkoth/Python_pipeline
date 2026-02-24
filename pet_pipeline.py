#!/usr/bin/env python3
"""Production pet re-identification pipeline.

Takes raw pet photos, runs detection + embedding + clustering, and outputs
organized folders of clustered images. Supports batch and incremental modes.

Usage:
    # Batch mode — cluster all photos from scratch
    python pet_pipeline.py batch --input /path/to/photos --species dog --output /path/to/clusters --gpu

    # Incremental mode — add new photos to existing clusters
    python pet_pipeline.py add --input /path/to/new_photos --state /path/to/clusters/state.npz --output /path/to/clusters --gpu

Dependencies:
    pip install opencv-python numpy onnxruntime ultralytics hdbscan tqdm
    # For GPU: pip install onnxruntime-gpu
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

_MODEL_DIR = Path.home() / "pet_embedding" / "models"

logger = logging.getLogger("pet_pipeline")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SpeciesConfig:
    """Species-specific thresholds and model paths."""
    species: str
    face_weight: float
    body_rescue_threshold: float
    min_body_agreements: int
    body_merge_threshold: float
    face_veto_threshold: float
    face_contradiction_threshold: float
    coco_class: int
    face_detect_model: str = ""
    face_embed_model: str = ""
    body_embed_model: str = ""
    body_detect_model: str = ""
    min_cluster_size: int = 2
    min_samples: int = 2
    min_body_overlap_ratio: float = 0.5

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "SpeciesConfig":
        return cls(**json.loads(s))


SPECIES_DEFAULTS = {
    "dog": SpeciesConfig(
        species="dog",
        face_weight=0.5,
        body_rescue_threshold=0.25,
        min_body_agreements=3,
        body_merge_threshold=0.35,
        face_veto_threshold=0.05,
        face_contradiction_threshold=0.10,
        coco_class=16,
        face_detect_model=str(_MODEL_DIR / "face_detect" / "yolov5n_petface_v2.onnx"),
        face_embed_model=str(_MODEL_DIR / "face_128" / "dog_byol_128.onnx"),
        body_embed_model=str(_MODEL_DIR / "body" / "dog_body.onnx"),
        body_detect_model=str(Path.home() / "pet_embedding" / "yolov8n.pt"),
    ),
    "cat": SpeciesConfig(
        species="cat",
        face_weight=0.3,
        body_rescue_threshold=0.20,
        min_body_agreements=2,
        body_merge_threshold=0.30,
        face_veto_threshold=0.05,
        face_contradiction_threshold=0.10,
        coco_class=15,
        face_detect_model=str(_MODEL_DIR / "face_detect" / "yolov5n_petface_v2.onnx"),
        face_embed_model=str(_MODEL_DIR / "face_128" / "cat_byol_128.onnx"),
        body_embed_model=str(_MODEL_DIR / "body" / "cat_body.onnx"),
        body_detect_model=str(Path.home() / "pet_embedding" / "yolov8n.pt"),
    ),
}


# ---------------------------------------------------------------------------
# ONNX provider selection
# ---------------------------------------------------------------------------

def get_onnx_providers(gpu: bool = False) -> list:
    """Return ONNX execution providers, preferring GPU if requested and available."""
    if gpu:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.warning("CUDA requested but CUDAExecutionProvider not available, falling back to CPU")
    return ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Detection utilities (from pet_identifier.py — verbatim)
# ---------------------------------------------------------------------------

def _preprocess_crop(bgr_crop: np.ndarray, size: int = 224) -> np.ndarray:
    """Preprocess a BGR crop for embedding model input."""
    img = cv2.resize(bgr_crop, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)
    return img[np.newaxis]


def _align_face(image: np.ndarray, bbox: list, left_eye: tuple, right_eye: tuple,
                output_size: int = 224) -> Optional[np.ndarray]:
    """Align face by rotating to make eyes horizontal, then crop."""
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    eye_dist = np.sqrt(dx**2 + dy**2)

    if eye_dist < 5:
        return None

    x1, y1, x2, y2 = bbox
    angle = np.degrees(np.arctan2(dy, dx))

    if abs(angle) < 1.0:
        crop = image[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (output_size, output_size))

    face_center = ((x1 + x2) / 2, (y1 + y2) / 2)
    M = cv2.getRotationMatrix2D(face_center, angle, 1.0)
    h, w = image.shape[:2]
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    bw, bh = x2 - x1, y2 - y1
    expand = 0.1
    nx1 = max(0, int(x1 - bw * expand))
    ny1 = max(0, int(y1 - bh * expand))
    nx2 = min(w, int(x2 + bw * expand))
    ny2 = min(h, int(y2 + bh * expand))

    crop = rotated[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (output_size, output_size))


def _crop_body(image: np.ndarray, box: np.ndarray, padding: float = 0.1) -> np.ndarray:
    """Crop body region with padding."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * padding), int(bh * padding)
    x1 = max(0, int(x1) - pad_x)
    y1 = max(0, int(y1) - pad_y)
    x2 = min(w, int(x2) + pad_x)
    y2 = min(h, int(y2) + pad_y)
    return image[y1:y2, x1:x2]


def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Pure numpy NMS. Returns indices to keep."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)


# ---------------------------------------------------------------------------
# FaceDetector (from pet_identifier.py — verbatim)
# ---------------------------------------------------------------------------

class FaceDetector:
    """Pet face detection using YOLOv5-face ONNX model with 3-point landmarks."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        input_size: int = 640,
        providers: Optional[list] = None,
    ):
        import onnxruntime as ort
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        if providers is None:
            providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def _letterbox(self, img: np.ndarray) -> tuple:
        """Resize with padding to maintain aspect ratio."""
        h, w = img.shape[:2]
        target = self.input_size
        scale = min(target / h, target / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = target - new_w, target - new_h
        top, left = pad_h // 2, pad_w // 2
        padded = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, scale, left, top

    def detect(self, image: np.ndarray) -> list:
        """Detect pet faces in a BGR image.

        Returns list of dicts with 'bbox' [x1,y1,x2,y2], 'conf', and
        'landmarks' np.ndarray of shape (3, 2) for [left_eye, right_eye, nose].
        """
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        padded, scale, pad_left, pad_top = self._letterbox(img_rgb)

        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        outputs = self._session.run(None, {self._input_name: blob})
        pred = outputs[0]

        if pred.ndim == 3:
            pred = pred[0]

        mask = pred[:, 4] > self.conf_threshold
        pred = pred[mask]
        if len(pred) == 0:
            return []

        boxes = _xywh2xyxy(pred[:, :4])
        obj_conf = pred[:, 4]
        landmarks_raw = pred[:, 5:11]

        cls_conf = pred[:, 11:]
        if cls_conf.shape[1] > 0:
            cls_max = cls_conf.max(axis=1)
            scores = obj_conf * cls_max
        else:
            scores = obj_conf

        keep = _nms_numpy(boxes, scores, self.iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        landmarks_raw = landmarks_raw[keep]

        faces = []
        for i in range(len(boxes)):
            box = boxes[i].copy()
            lm = landmarks_raw[i].copy().reshape(3, 2)

            box[[0, 2]] = (box[[0, 2]] - pad_left) / scale
            box[[1, 3]] = (box[[1, 3]] - pad_top) / scale
            lm[:, 0] = (lm[:, 0] - pad_left) / scale
            lm[:, 1] = (lm[:, 1] - pad_top) / scale

            box[0] = max(0, box[0]); box[1] = max(0, box[1])
            box[2] = min(w, box[2]); box[3] = min(h, box[3])
            lm[:, 0] = np.clip(lm[:, 0], 0, w)
            lm[:, 1] = np.clip(lm[:, 1], 0, h)

            faces.append({
                "bbox": box.astype(int).tolist(),
                "conf": float(scores[i]),
                "landmarks": lm,
                "left_eye": tuple(lm[0]),
                "right_eye": tuple(lm[1]),
            })

        return faces


# ---------------------------------------------------------------------------
# BodyDetector (adapted from pet_identifier.py)
# ---------------------------------------------------------------------------

class BodyDetector:
    """Pet body detection using ultralytics YOLOv8."""

    def __init__(self, model_path: str, coco_class: int, conf_threshold: float = 0.3):
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._target_class = coco_class
        self._conf = conf_threshold

    def detect(self, image: np.ndarray) -> Optional[dict]:
        """Detect the best body bounding box in a BGR image.

        Returns dict with 'bbox' (x1,y1,x2,y2 ndarray) and 'conf', or None.
        """
        results = self._model(image, conf=self._conf, classes=[self._target_class], verbose=False)
        boxes = results[0].boxes
        if len(boxes) == 0:
            return None
        confs = boxes.conf.cpu().numpy()
        best_idx = confs.argmax()
        return {
            "bbox": boxes.xyxy[best_idx].cpu().numpy(),
            "conf": float(confs[best_idx]),
        }

    def detect_batch_species(self, images: list, sample_size: int = 50) -> str:
        """Auto-detect species from a sample of images. Returns 'dog' or 'cat'."""
        import random
        sample = random.sample(images, min(sample_size, len(images)))
        dog_count = 0
        cat_count = 0
        for img_path in sample:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            results = self._model(img, conf=0.3, classes=[15, 16], verbose=False)
            boxes = results[0].boxes
            if len(boxes) == 0:
                continue
            classes = boxes.cls.cpu().numpy().astype(int)
            for c in classes:
                if c == 16:
                    dog_count += 1
                elif c == 15:
                    cat_count += 1
        detected = "dog" if dog_count >= cat_count else "cat"
        logger.info(f"Species auto-detect: dog={dog_count}, cat={cat_count} -> {detected}")
        return detected


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """Face and body embedding extraction via ONNX."""

    def __init__(self, face_model: str, body_model: str, providers: Optional[list] = None):
        import onnxruntime as ort
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self._face_session = ort.InferenceSession(face_model, providers=providers)
        self._face_input = self._face_session.get_inputs()[0].name
        self.face_dim = self._probe_dim(self._face_session, self._face_input)

        self._body_session = ort.InferenceSession(body_model, providers=providers)
        self._body_input = self._body_session.get_inputs()[0].name
        self.body_dim = self._probe_dim(self._body_session, self._body_input)

    @staticmethod
    def _probe_dim(session, input_name: str) -> int:
        dim = session.get_outputs()[0].shape[-1]
        if isinstance(dim, int):
            return dim
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        out = session.run(None, {input_name: dummy})[0]
        return out.shape[1]

    def embed_face(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Embed an aligned face crop (BGR). Returns L2-normalized embedding."""
        inp = _preprocess_crop(aligned_bgr)
        emb = self._face_session.run(None, {self._face_input: inp})[0][0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def embed_body(self, body_bgr: np.ndarray) -> np.ndarray:
        """Embed a body crop (BGR). Returns L2-normalized embedding."""
        inp = _preprocess_crop(body_bgr)
        emb = self._body_session.run(None, {self._body_input: inp})[0][0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb


# ---------------------------------------------------------------------------
# ImageResult & ImageProcessor
# ---------------------------------------------------------------------------

@dataclass
class ImageResult:
    """Processing result for a single image."""
    path: str
    face_emb: Optional[np.ndarray] = None
    body_emb: Optional[np.ndarray] = None
    has_face: bool = False
    has_body: bool = False


class ImageProcessor:
    """Detect → align/crop → embed for each image."""

    def __init__(self, face_detector: FaceDetector, body_detector: BodyDetector,
                 embedder: Embedder):
        self.face_detector = face_detector
        self.body_detector = body_detector
        self.embedder = embedder

    def process(self, image_path: str) -> ImageResult:
        """Process a single image. Returns ImageResult with embeddings."""
        result = ImageResult(path=image_path)
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return result

        # Face detection + alignment + embedding
        faces = self.face_detector.detect(img)
        if faces:
            best_face = max(faces, key=lambda f: f["conf"])
            bbox = [int(v) for v in best_face["bbox"]]
            left_eye = best_face.get("left_eye")
            right_eye = best_face.get("right_eye")

            if left_eye is not None and right_eye is not None:
                aligned = _align_face(img, bbox, left_eye, right_eye)
            else:
                x1, y1, x2, y2 = bbox
                crop = img[max(0, y1):y2, max(0, x1):x2]
                aligned = cv2.resize(crop, (224, 224)) if crop.size > 0 else None

            if aligned is not None:
                result.face_emb = self.embedder.embed_face(aligned)
                result.has_face = True

        # Body detection + crop + embedding
        body = self.body_detector.detect(img)
        if body is not None:
            crop = _crop_body(img, body["bbox"], padding=0.1)
            if crop.shape[0] >= 32 and crop.shape[1] >= 32:
                result.body_emb = self.embedder.embed_body(crop)
                result.has_body = True

        return result


# ---------------------------------------------------------------------------
# Clustering Engine (adapted from fused_clustering.py — unified index space)
# ---------------------------------------------------------------------------

class ClusterEngine:
    """3-phase fused clustering: face HDBSCAN → body rescue → cross-cluster merge."""

    def __init__(self, config: SpeciesConfig):
        self.config = config

    def phase1_face_cluster(self, face_embeddings: np.ndarray,
                            has_face: np.ndarray) -> np.ndarray:
        """Phase 1: HDBSCAN on face embeddings.

        Only images with faces participate. Others start as -1.
        Returns labels array of length N.
        """
        import hdbscan

        n = len(face_embeddings)
        labels = np.full(n, -1, dtype=np.int32)

        face_indices = np.where(has_face)[0]
        if len(face_indices) < 2:
            logger.warning("Fewer than 2 face images — skipping face clustering")
            return labels

        face_embs = face_embeddings[face_indices]
        sim = face_embs @ face_embs.T
        distance = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float64)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            metric="precomputed",
            cluster_selection_method="eom",
        )
        face_labels = clusterer.fit_predict(distance)

        # Map back to global indices
        for i, global_idx in enumerate(face_indices):
            labels[global_idx] = face_labels[i]

        n_clusters = len(set(face_labels) - {-1})
        n_noise = (face_labels == -1).sum()
        logger.info(f"  HDBSCAN: {n_clusters} clusters, {n_noise} unclustered (from {len(face_indices)} face images)")
        return labels

    def phase2_body_rescue(self, labels: np.ndarray, face_embeddings: np.ndarray,
                           body_embeddings: np.ndarray, has_face: np.ndarray,
                           has_body: np.ndarray,
                           face_centroids: dict) -> tuple:
        """Phase 2: Rescue unclustered images using body similarity.

        Simplified from fused_clustering.py — unified index, no face_to_body mapping.
        """
        updated_labels = labels.copy()
        rescue_log = []

        # Build body embeddings per cluster
        cluster_body_embs = defaultdict(list)
        for idx in range(len(labels)):
            if labels[idx] == -1:
                continue
            if has_body[idx]:
                cluster_body_embs[labels[idx]].append(body_embeddings[idx])

        cluster_body_stacked = {}
        for c, embs in cluster_body_embs.items():
            if len(embs) >= 1:
                cluster_body_stacked[c] = np.stack(embs)

        unclustered = np.where(labels == -1)[0]
        logger.info(f"  Body rescue: {len(unclustered)} unclustered images...")

        rescued = 0
        vetoed = 0
        no_body = 0
        insufficient = 0

        for img_idx in unclustered:
            if not has_body[img_idx]:
                no_body += 1
                continue

            candidate_body = body_embeddings[img_idx]
            best_cluster = -1
            best_avg_sim = -1
            best_n_agree = 0

            for cluster_id, stacked in cluster_body_stacked.items():
                sims = stacked @ candidate_body
                n_above = int((sims > self.config.body_rescue_threshold).sum())
                avg_sim = float(sims.mean())

                if n_above >= self.config.min_body_agreements and avg_sim > best_avg_sim:
                    best_cluster = cluster_id
                    best_avg_sim = avg_sim
                    best_n_agree = n_above

            if best_cluster == -1:
                insufficient += 1
                continue

            # Face veto
            if has_face[img_idx]:
                face_emb = face_embeddings[img_idx]
                face_norm = np.linalg.norm(face_emb)
                if face_norm > 0.1:
                    face_sim = float(face_emb @ face_centroids[best_cluster])
                    if face_sim < self.config.face_veto_threshold:
                        vetoed += 1
                        rescue_log.append({
                            "idx": int(img_idx), "action": "vetoed",
                            "reason": f"face_sim={face_sim:.4f}",
                            "proposed_cluster": int(best_cluster),
                        })
                        continue

            updated_labels[img_idx] = best_cluster
            rescued += 1
            rescue_log.append({
                "idx": int(img_idx), "action": "rescued",
                "cluster": int(best_cluster),
                "body_avg_sim": round(best_avg_sim, 4),
                "n_agree": best_n_agree,
            })

        logger.info(f"  Rescued: {rescued}, Vetoed: {vetoed}, No body: {no_body}, Insufficient: {insufficient}")
        return updated_labels, rescue_log

    def phase2b_body_cluster(self, labels: np.ndarray, body_embeddings: np.ndarray,
                             has_body: np.ndarray) -> tuple:
        """Phase 2b: Form new clusters from body-only unclustered images.

        After face clustering (Phase 1) and body rescue (Phase 2), some images
        remain unclustered because they had no face AND no existing cluster to
        rescue into. This phase runs HDBSCAN on their body embeddings to form
        new body-only clusters.

        These clusters are lower confidence than face-based ones (body is less
        discriminative), but better than leaving them unclustered.
        """
        import hdbscan

        updated_labels = labels.copy()
        body_cluster_log = []

        # Find images still unclustered that have body embeddings
        still_unclustered = np.where((labels == -1) & has_body)[0]

        if len(still_unclustered) < self.config.min_cluster_size:
            logger.info(f"  Body clustering: only {len(still_unclustered)} body-only unclustered images, need {self.config.min_cluster_size}. Skipping.")
            return updated_labels, body_cluster_log

        logger.info(f"  Body clustering: running HDBSCAN on {len(still_unclustered)} unclustered body images...")

        # HDBSCAN on body embeddings
        body_embs = body_embeddings[still_unclustered]
        sim = body_embs @ body_embs.T
        distance = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float64)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            metric="precomputed",
            cluster_selection_method="eom",
        )
        body_labels = clusterer.fit_predict(distance)

        n_new_clusters = len(set(body_labels) - {-1})
        n_assigned = (body_labels != -1).sum()

        if n_new_clusters == 0:
            logger.info("  Body clustering: no clusters formed")
            return updated_labels, body_cluster_log

        # Offset new cluster IDs above existing max
        existing_max = max(set(labels) - {-1}) if (labels != -1).any() else -1

        for i, global_idx in enumerate(still_unclustered):
            if body_labels[i] != -1:
                new_id = body_labels[i] + existing_max + 1
                updated_labels[global_idx] = new_id
                body_cluster_log.append({
                    "idx": int(global_idx),
                    "action": "body_clustered",
                    "cluster": int(new_id),
                })

        logger.info(f"  Body clustering: {n_new_clusters} new clusters, {n_assigned} images assigned")
        return updated_labels, body_cluster_log

    def phase3_cross_merge(self, labels: np.ndarray, face_embeddings: np.ndarray,
                           body_embeddings: np.ndarray, has_face: np.ndarray,
                           has_body: np.ndarray) -> tuple:
        """Phase 3: Merge clusters that are separate in face space but similar in body space.

        Simplified — unified index, direct access to body via has_body mask.
        """
        updated_labels = labels.copy()
        merge_log = []

        unique_clusters = sorted(set(labels) - {-1})
        if len(unique_clusters) < 2:
            return updated_labels, merge_log

        # Build per-cluster embeddings
        cluster_face = {}
        cluster_body = {}
        for c in unique_clusters:
            mask = labels == c
            indices = np.where(mask)[0]
            cluster_face[c] = face_embeddings[indices[has_face[indices]]]

            body_mask = has_body[indices]
            if body_mask.any():
                cluster_body[c] = body_embeddings[indices[body_mask]]
            else:
                cluster_body[c] = None

        # Pre-screen using body centroids
        body_centroids = {}
        for c in unique_clusters:
            if cluster_body[c] is not None:
                centroid = np.median(cluster_body[c], axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                body_centroids[c] = centroid

        centroid_screen_thresh = self.config.body_merge_threshold * 0.7
        candidate_pairs = []
        clusters_with_body = [c for c in unique_clusters if c in body_centroids]

        if len(clusters_with_body) > 1:
            centroid_matrix = np.stack([body_centroids[c] for c in clusters_with_body])
            centroid_sims = centroid_matrix @ centroid_matrix.T
            for i in range(len(clusters_with_body)):
                for j in range(i + 1, len(clusters_with_body)):
                    if centroid_sims[i, j] > centroid_screen_thresh:
                        candidate_pairs.append(
                            (clusters_with_body[i], clusters_with_body[j]))

        logger.info(f"  Cross-merge: {len(candidate_pairs)} candidate pairs from {len(clusters_with_body)} clusters")

        merge_pairs = []
        for c1, c2 in candidate_pairs:
            body_sims = cluster_body[c1] @ cluster_body[c2].T
            avg_body_sim = float(body_sims.mean())
            ratio_above = float((body_sims > self.config.body_merge_threshold).mean())

            if avg_body_sim < self.config.body_merge_threshold:
                continue
            if ratio_above < self.config.min_body_overlap_ratio:
                continue

            # Face contradiction check
            if len(cluster_face[c1]) > 0 and len(cluster_face[c2]) > 0:
                face_sims = cluster_face[c1] @ cluster_face[c2].T
                avg_face_sim = float(face_sims.mean())
            else:
                avg_face_sim = 1.0  # no face data, no contradiction

            if avg_face_sim < self.config.face_contradiction_threshold:
                merge_log.append({
                    "action": "blocked", "clusters": [int(c1), int(c2)],
                    "reason": f"face_contradiction={avg_face_sim:.4f}",
                    "avg_body_sim": round(avg_body_sim, 4),
                })
                continue

            merge_pairs.append((c1, c2, avg_body_sim, avg_face_sim))

        merge_pairs.sort(key=lambda x: -x[2])

        # Union-find
        parent = {c: c for c in unique_clusters}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        merges_applied = 0
        for c1, c2, avg_body, avg_face in merge_pairs:
            if find(c1) != find(c2):
                union(c1, c2)
                merges_applied += 1
                merge_log.append({
                    "action": "merged", "clusters": [int(c1), int(c2)],
                    "avg_body_sim": round(avg_body, 4),
                    "avg_face_sim": round(avg_face, 4),
                })

        if merges_applied > 0:
            for i in range(len(updated_labels)):
                if updated_labels[i] != -1:
                    updated_labels[i] = find(updated_labels[i])

        logger.info(f"  Cross-merge: merged {merges_applied} pairs")
        return updated_labels, merge_log

    def compute_face_centroids(self, face_embeddings: np.ndarray,
                               labels: np.ndarray, has_face: np.ndarray) -> dict:
        """Compute L2-normalized face centroids per cluster (median-based)."""
        centroids = {}
        unique_labels = set(labels) - {-1}
        for c in unique_labels:
            mask = (labels == c) & has_face
            if not mask.any():
                # Use all cluster members (zero-embedding faces included)
                mask = labels == c
            cluster_embs = face_embeddings[mask]
            centroid = np.median(cluster_embs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[c] = centroid
        return centroids


# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """Full pipeline state for save/load."""
    image_paths: np.ndarray  # (N,) dtype=object, string paths
    face_embeddings: np.ndarray  # (N, face_dim)
    body_embeddings: np.ndarray  # (N, body_dim)
    has_face: np.ndarray  # (N,) bool
    has_body: np.ndarray  # (N,) bool
    cluster_labels: np.ndarray  # (N,) int32, -1=unclustered
    species: str
    face_dim: int
    body_dim: int
    config: SpeciesConfig


class StateManager:
    """Save/load pipeline state as .npz."""

    @staticmethod
    def save(state: PipelineState, path: str):
        np.savez(
            path,
            image_paths=np.array([str(p) for p in state.image_paths], dtype=object),
            face_embeddings=state.face_embeddings.astype(np.float32),
            body_embeddings=state.body_embeddings.astype(np.float32),
            has_face=state.has_face.astype(np.bool_),
            has_body=state.has_body.astype(np.bool_),
            cluster_labels=state.cluster_labels.astype(np.int32),
            species=np.array(state.species),
            face_dim=np.array(state.face_dim),
            body_dim=np.array(state.body_dim),
            config_json=np.array(state.config.to_json()),
        )
        logger.info(f"State saved to {path}")

    @staticmethod
    def load(path: str) -> PipelineState:
        data = np.load(path, allow_pickle=True)
        try:
            config = SpeciesConfig.from_json(str(data["config_json"]))
        except Exception as e:
            raise ValueError(f"Corrupt state file — could not parse config: {e}")

        return PipelineState(
            image_paths=data["image_paths"],
            face_embeddings=data["face_embeddings"],
            body_embeddings=data["body_embeddings"],
            has_face=data["has_face"],
            has_body=data["has_body"],
            cluster_labels=data["cluster_labels"],
            species=str(data["species"]),
            face_dim=int(data["face_dim"]),
            body_dim=int(data["body_dim"]),
            config=config,
        )


# ---------------------------------------------------------------------------
# Output Organizer
# ---------------------------------------------------------------------------

class OutputOrganizer:
    """Create cluster folders with symlinks to original images."""

    @staticmethod
    def organize(output_dir: Path, image_paths: np.ndarray,
                 labels: np.ndarray, use_symlinks: bool = True):
        """Create output folder structure from labels.

        Creates cluster_001/, cluster_002/, ..., unclustered/ dirs.
        Uses symlinks by default, falls back to copy if symlinks fail.
        """
        # Clean existing cluster dirs
        for item in output_dir.iterdir():
            if item.is_dir() and (item.name.startswith("cluster_") or item.name == "unclustered"):
                shutil.rmtree(item)

        # Renumber labels to contiguous 0..K-1
        unique_labels = sorted(set(labels) - {-1})
        label_map = {old: new for new, old in enumerate(unique_labels)}

        # Track filenames to handle duplicates
        folder_names = defaultdict(set)

        for idx, (path, label) in enumerate(zip(image_paths, labels)):
            path = str(path)
            if label == -1:
                folder = output_dir / "unclustered"
            else:
                new_label = label_map[label]
                folder = output_dir / f"cluster_{new_label + 1:03d}"

            folder.mkdir(parents=True, exist_ok=True)

            filename = Path(path).name
            # Handle duplicate filenames within same folder
            base_name = filename
            if filename in folder_names[folder]:
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                counter = 1
                while filename in folder_names[folder]:
                    filename = f"{stem}_{counter}{suffix}"
                    counter += 1

            folder_names[folder].add(filename)
            dest = folder / filename

            if use_symlinks:
                try:
                    dest.symlink_to(Path(path).resolve())
                except OSError:
                    shutil.copy2(path, dest)
            else:
                shutil.copy2(path, dest)

        n_clusters = len(unique_labels)
        n_unclustered = int((labels == -1).sum())
        logger.info(f"Organized {len(image_paths)} images into {n_clusters} clusters + {n_unclustered} unclustered")
        return n_clusters

    @staticmethod
    def write_summary(output_dir: Path, image_paths: np.ndarray,
                      labels: np.ndarray, config: SpeciesConfig,
                      elapsed: float, extra: Optional[dict] = None):
        """Write summary.json with cluster statistics."""
        unique_labels = sorted(set(labels) - {-1})
        label_map = {old: new for new, old in enumerate(unique_labels)}

        cluster_sizes = {}
        for label in unique_labels:
            size = int((labels == label).sum())
            cluster_sizes[f"cluster_{label_map[label] + 1:03d}"] = size

        summary = {
            "species": config.species,
            "total_images": len(image_paths),
            "n_clusters": len(unique_labels),
            "n_unclustered": int((labels == -1).sum()),
            "cluster_sizes": cluster_sizes,
            "elapsed_seconds": round(elapsed, 2),
            "config": asdict(config),
        }
        if extra:
            summary.update(extra)

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary written to {summary_path}")


# ---------------------------------------------------------------------------
# Utility: renumber labels
# ---------------------------------------------------------------------------

def renumber_labels(labels: np.ndarray) -> np.ndarray:
    """Renumber cluster labels to contiguous 0..K-1, preserving -1."""
    out = labels.copy()
    unique = sorted(set(labels) - {-1})
    mapping = {old: new for new, old in enumerate(unique)}
    for i in range(len(out)):
        if out[i] != -1:
            out[i] = mapping[out[i]]
    return out


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

class PetPipeline:
    """Production pet clustering pipeline."""

    def __init__(self, config: SpeciesConfig, gpu: bool = False):
        self.config = config
        self.gpu = gpu
        self._providers = get_onnx_providers(gpu)

        logger.info(f"Initializing pipeline: species={config.species}, gpu={gpu}")
        logger.info(f"  Face detect: {config.face_detect_model}")
        logger.info(f"  Face embed:  {config.face_embed_model}")
        logger.info(f"  Body embed:  {config.body_embed_model}")
        logger.info(f"  Body detect: {config.body_detect_model}")

        self.face_detector = FaceDetector(
            model_path=config.face_detect_model,
            providers=self._providers,
        )
        self.body_detector = BodyDetector(
            model_path=config.body_detect_model,
            coco_class=config.coco_class,
        )
        self.embedder = Embedder(
            face_model=config.face_embed_model,
            body_model=config.body_embed_model,
            providers=self._providers,
        )
        self.processor = ImageProcessor(self.face_detector, self.body_detector, self.embedder)
        self.cluster_engine = ClusterEngine(config)

    def _scan_images(self, input_dir: Path) -> list:
        """Scan directory recursively for image files."""
        paths = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(input_dir.rglob(f"*{ext}"))
            paths.extend(input_dir.rglob(f"*{ext.upper()}"))
        # Deduplicate (case-insensitive extensions on case-insensitive FS)
        seen = set()
        unique = []
        for p in sorted(paths):
            resolved = str(p.resolve())
            if resolved not in seen:
                seen.add(resolved)
                unique.append(p)
        return unique

    def _process_images(self, image_paths: list) -> list:
        """Process images with progress bar."""
        from tqdm import tqdm
        results = []
        for path in tqdm(image_paths, desc="Processing images"):
            try:
                result = self.processor.process(str(path))
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process {path}: {e}")
                results.append(ImageResult(path=str(path)))
        return results

    def _build_matrices(self, results: list) -> tuple:
        """Build embedding matrices from ImageResult list."""
        n = len(results)
        face_dim = self.embedder.face_dim
        body_dim = self.embedder.body_dim

        face_embs = np.zeros((n, face_dim), dtype=np.float32)
        body_embs = np.zeros((n, body_dim), dtype=np.float32)
        has_face = np.zeros(n, dtype=np.bool_)
        has_body = np.zeros(n, dtype=np.bool_)
        paths = []

        for i, r in enumerate(results):
            paths.append(r.path)
            if r.has_face and r.face_emb is not None:
                face_embs[i] = r.face_emb
                has_face[i] = True
            if r.has_body and r.body_emb is not None:
                body_embs[i] = r.body_emb
                has_body[i] = True

        return np.array(paths, dtype=object), face_embs, body_embs, has_face, has_body

    def run_batch(self, input_dir: Path, output_dir: Path):
        """Full batch pipeline: scan → detect → embed → cluster → organize."""
        start = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Scan
        print(f"\n{'='*60}")
        print(f"  PET PIPELINE — BATCH MODE ({self.config.species.upper()})")
        print(f"{'='*60}\n")

        image_paths = self._scan_images(input_dir)
        print(f"[1/6] Found {len(image_paths)} images in {input_dir}")
        if not image_paths:
            print("ERROR: No images found.")
            sys.exit(1)

        # Step 2: Process
        print(f"[2/6] Detecting and embedding...")
        results = self._process_images(image_paths)

        # Step 3: Build matrices
        paths, face_embs, body_embs, has_face, has_body = self._build_matrices(results)
        n_faces = has_face.sum()
        n_bodies = has_body.sum()
        print(f"  Faces detected: {n_faces}/{len(paths)} ({100*n_faces/len(paths):.1f}%)")
        print(f"  Bodies detected: {n_bodies}/{len(paths)} ({100*n_bodies/len(paths):.1f}%)")

        if n_faces == 0:
            print("ERROR: Zero faces detected in entire batch. Cannot cluster.")
            sys.exit(1)

        # Step 4: Phase 1 — Face clustering
        print(f"\n[3/7] Phase 1: Face-based clustering...")
        labels = self.cluster_engine.phase1_face_cluster(face_embs, has_face)
        face_centroids = self.cluster_engine.compute_face_centroids(face_embs, labels, has_face)

        # Step 5: Phase 2 — Body rescue
        print(f"[4/7] Phase 2: Body rescue...")
        labels, rescue_log = self.cluster_engine.phase2_body_rescue(
            labels, face_embs, body_embs, has_face, has_body, face_centroids)

        # Step 6: Phase 2b — Body-only clustering
        print(f"[5/7] Phase 2b: Body-only clustering...")
        labels, body_cluster_log = self.cluster_engine.phase2b_body_cluster(
            labels, body_embs, has_body)

        # Step 7: Phase 3 — Cross merge
        print(f"[6/7] Phase 3: Cross-cluster merge...")
        labels, merge_log = self.cluster_engine.phase3_cross_merge(
            labels, face_embs, body_embs, has_face, has_body)

        # Renumber
        labels = renumber_labels(labels)

        # Organize output
        print(f"[7/7] Organizing output...")
        n_clusters = OutputOrganizer.organize(output_dir, paths, labels)

        elapsed = time.time() - start

        # Save state
        state = PipelineState(
            image_paths=paths,
            face_embeddings=face_embs,
            body_embeddings=body_embs,
            has_face=has_face,
            has_body=has_body,
            cluster_labels=labels,
            species=self.config.species,
            face_dim=self.embedder.face_dim,
            body_dim=self.embedder.body_dim,
            config=self.config,
        )
        StateManager.save(state, str(output_dir / "state.npz"))

        OutputOrganizer.write_summary(
            output_dir, paths, labels, self.config, elapsed,
            extra={"rescue_count": len([r for r in rescue_log if r["action"] == "rescued"]),
                   "body_cluster_count": len([b for b in body_cluster_log if b["action"] == "body_clustered"]),
                   "merge_count": len([m for m in merge_log if m["action"] == "merged"])})

        # Print summary
        n_unclustered = int((labels == -1).sum())
        print(f"\n{'='*60}")
        print(f"  RESULTS")
        print(f"{'='*60}")
        print(f"  Total images:   {len(paths)}")
        print(f"  Clusters:       {n_clusters}")
        print(f"  Clustered:      {len(paths) - n_unclustered} ({100*(len(paths)-n_unclustered)/len(paths):.1f}%)")
        print(f"  Unclustered:    {n_unclustered}")
        print(f"  Time:           {elapsed:.1f}s")
        print(f"  Output:         {output_dir}")
        print()

    def run_incremental(self, input_dir: Path, state_path: str, output_dir: Path):
        """Incremental mode: add new images to existing clusters."""
        start = time.time()

        print(f"\n{'='*60}")
        print(f"  PET PIPELINE — INCREMENTAL MODE ({self.config.species.upper()})")
        print(f"{'='*60}\n")

        # Step 1: Load existing state
        print(f"[1/7] Loading existing state from {state_path}...")
        state = StateManager.load(state_path)
        existing_paths = set(str(p) for p in state.image_paths)
        print(f"  Existing: {len(state.image_paths)} images, "
              f"{len(set(state.cluster_labels) - {-1})} clusters")

        # Step 2: Scan for new images
        image_paths = self._scan_images(input_dir)
        new_paths = [p for p in image_paths if str(p.resolve()) not in existing_paths
                     and str(p) not in existing_paths]
        print(f"[2/7] Found {len(new_paths)} new images ({len(image_paths)} total scanned)")

        if not new_paths:
            print("No new images to process.")
            return

        # Step 3: Process new images
        print(f"[3/7] Processing new images...")
        results = self._process_images(new_paths)
        new_path_arr, new_face, new_body, new_has_face, new_has_body = self._build_matrices(results)

        n_faces = new_has_face.sum()
        n_bodies = new_has_body.sum()
        print(f"  New faces: {n_faces}/{len(new_paths)}, New bodies: {n_bodies}/{len(new_paths)}")

        # Step 4: Assign to existing clusters
        print(f"[4/7] Assigning new images to existing clusters...")
        existing_labels = state.cluster_labels
        existing_face = state.face_embeddings
        existing_body = state.body_embeddings
        existing_has_face = state.has_face
        existing_has_body = state.has_body

        # Compute centroids of existing clusters
        existing_clusters = sorted(set(existing_labels) - {-1})
        face_centroids = {}
        body_centroids = {}
        for c in existing_clusters:
            c_mask = existing_labels == c
            # Face centroid
            c_face_mask = c_mask & existing_has_face
            if c_face_mask.any():
                centroid = np.median(existing_face[c_face_mask], axis=0)
                norm = np.linalg.norm(centroid)
                face_centroids[c] = centroid / norm if norm > 0 else centroid
            # Body centroid
            c_body_mask = c_mask & existing_has_body
            if c_body_mask.any():
                centroid = np.median(existing_body[c_body_mask], axis=0)
                norm = np.linalg.norm(centroid)
                body_centroids[c] = centroid / norm if norm > 0 else centroid

        new_labels = np.full(len(new_paths), -1, dtype=np.int32)
        assigned = 0
        fw = self.config.face_weight
        bw = 1.0 - fw
        assignment_threshold = self.config.body_rescue_threshold  # reuse as assignment threshold

        for i in range(len(new_paths)):
            best_score = -1.0
            best_cluster = -1

            for c in existing_clusters:
                score = 0.0
                n_modalities = 0

                if new_has_face[i] and c in face_centroids:
                    face_sim = float(new_face[i] @ face_centroids[c])
                    score += fw * face_sim
                    n_modalities += 1

                if new_has_body[i] and c in body_centroids:
                    body_sim = float(new_body[i] @ body_centroids[c])
                    score += bw * body_sim
                    n_modalities += 1

                if n_modalities == 0:
                    continue

                # Normalize if only one modality
                if n_modalities == 1:
                    if new_has_face[i] and c in face_centroids:
                        score = float(new_face[i] @ face_centroids[c])
                    elif new_has_body[i] and c in body_centroids:
                        score = float(new_body[i] @ body_centroids[c])

                if score > best_score:
                    best_score = score
                    best_cluster = c

            if best_score > assignment_threshold and best_cluster >= 0:
                new_labels[i] = best_cluster
                assigned += 1

        print(f"  Assigned {assigned}/{len(new_paths)} to existing clusters")
        unassigned_mask = new_labels == -1

        # Step 5: Cluster remaining unassigned
        n_unassigned = unassigned_mask.sum()
        print(f"[5/7] Clustering {n_unassigned} unassigned images...")

        if n_unassigned >= self.config.min_cluster_size:
            # Offset new cluster IDs above existing max
            max_existing = max(existing_clusters) if existing_clusters else -1
            unassigned_idx = np.where(unassigned_mask)[0]

            sub_face = new_face[unassigned_idx]
            sub_body = new_body[unassigned_idx]
            sub_has_face = new_has_face[unassigned_idx]
            sub_has_body = new_has_body[unassigned_idx]

            sub_labels = self.cluster_engine.phase1_face_cluster(sub_face, sub_has_face)

            if (sub_labels != -1).any():
                sub_centroids = self.cluster_engine.compute_face_centroids(
                    sub_face, sub_labels, sub_has_face)
                sub_labels, _ = self.cluster_engine.phase2_body_rescue(
                    sub_labels, sub_face, sub_body, sub_has_face, sub_has_body, sub_centroids)

            # Phase 2b: body-only clustering for remaining unclustered
            sub_labels, _ = self.cluster_engine.phase2b_body_cluster(
                sub_labels, sub_body, sub_has_body)

            # Offset new cluster IDs
            for j in range(len(sub_labels)):
                if sub_labels[j] != -1:
                    new_labels[unassigned_idx[j]] = sub_labels[j] + max_existing + 1

        # Step 6: Merge state
        print(f"[6/7] Merging state...")
        all_paths = np.concatenate([state.image_paths, new_path_arr])
        all_face = np.concatenate([existing_face, new_face])
        all_body = np.concatenate([existing_body, new_body])
        all_has_face = np.concatenate([existing_has_face, new_has_face])
        all_has_body = np.concatenate([existing_has_body, new_has_body])
        all_labels = np.concatenate([existing_labels, new_labels])

        # Cross-merge between new and existing clusters
        all_labels, merge_log = self.cluster_engine.phase3_cross_merge(
            all_labels, all_face, all_body, all_has_face, all_has_body)

        all_labels = renumber_labels(all_labels)

        # Step 7: Re-organize output
        print(f"[7/7] Reorganizing output...")
        output_dir.mkdir(parents=True, exist_ok=True)
        n_clusters = OutputOrganizer.organize(output_dir, all_paths, all_labels)

        elapsed = time.time() - start

        updated_state = PipelineState(
            image_paths=all_paths,
            face_embeddings=all_face,
            body_embeddings=all_body,
            has_face=all_has_face,
            has_body=all_has_body,
            cluster_labels=all_labels,
            species=self.config.species,
            face_dim=state.face_dim,
            body_dim=state.body_dim,
            config=self.config,
        )
        StateManager.save(updated_state, str(output_dir / "state.npz"))
        OutputOrganizer.write_summary(
            output_dir, all_paths, all_labels, self.config, elapsed,
            extra={"mode": "incremental", "new_images": len(new_paths),
                   "assigned_to_existing": assigned})

        n_unclustered = int((all_labels == -1).sum())
        print(f"\n{'='*60}")
        print(f"  RESULTS")
        print(f"{'='*60}")
        print(f"  Total images:      {len(all_paths)} (+{len(new_paths)} new)")
        print(f"  Clusters:          {n_clusters}")
        print(f"  Clustered:         {len(all_paths) - n_unclustered}")
        print(f"  Unclustered:       {n_unclustered}")
        print(f"  Assigned existing: {assigned}")
        print(f"  Time:              {elapsed:.1f}s")
        print(f"  Output:            {output_dir}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_config(args) -> SpeciesConfig:
    """Build SpeciesConfig from CLI args, starting from species defaults."""
    config = SpeciesConfig(**asdict(SPECIES_DEFAULTS[args.species]))

    # Override model paths if provided
    if hasattr(args, 'face_detect_model') and args.face_detect_model:
        config.face_detect_model = args.face_detect_model
    if hasattr(args, 'face_embed_model') and args.face_embed_model:
        config.face_embed_model = args.face_embed_model
    if hasattr(args, 'body_embed_model') and args.body_embed_model:
        config.body_embed_model = args.body_embed_model
    if hasattr(args, 'body_detect_model') and args.body_detect_model:
        config.body_detect_model = args.body_detect_model

    return config


def auto_detect_species(input_dir: Path, gpu: bool = False) -> str:
    """Auto-detect species by running body detector on a sample."""
    default_body_model = str(Path.home() / "pet_embedding" / "yolov8n.pt")
    detector = BodyDetector(model_path=default_body_model, coco_class=16)

    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(input_dir.rglob(f"*{ext}"))
        image_paths.extend(input_dir.rglob(f"*{ext.upper()}"))

    if not image_paths:
        print("No images found for species detection, defaulting to 'dog'")
        return "dog"

    species = detector.detect_batch_species(image_paths, sample_size=50)
    print(f"Auto-detected species: {species}")
    return species


def main():
    parser = argparse.ArgumentParser(
        description="Production pet re-identification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python pet_pipeline.py batch --input /photos --species dog --output /clusters --gpu
  python pet_pipeline.py add --input /new_photos --state /clusters/state.npz --output /clusters
""")

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    sub = parser.add_subparsers(dest="command")

    # Batch subcommand
    batch_p = sub.add_parser("batch", help="Cluster all photos from scratch")
    batch_p.add_argument("--input", "-i", required=True, help="Input image directory")
    batch_p.add_argument("--output", "-o", required=True, help="Output cluster directory")
    batch_p.add_argument("--species", "-s", choices=["dog", "cat"], default=None,
                         help="Species (auto-detected if not given)")
    batch_p.add_argument("--gpu", action="store_true", help="Use GPU for ONNX inference")
    batch_p.add_argument("--face-detect-model", default=None, help="Custom face detection model")
    batch_p.add_argument("--face-embed-model", default=None, help="Custom face embedding model")
    batch_p.add_argument("--body-embed-model", default=None, help="Custom body embedding model")
    batch_p.add_argument("--body-detect-model", default=None, help="Custom body detection model")

    # Incremental subcommand
    add_p = sub.add_parser("add", help="Add new photos to existing clusters")
    add_p.add_argument("--input", "-i", required=True, help="Input directory with new images")
    add_p.add_argument("--state", required=True, help="Path to existing state.npz")
    add_p.add_argument("--output", "-o", required=True, help="Output cluster directory")
    add_p.add_argument("--gpu", action="store_true", help="Use GPU for ONNX inference")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "batch":
        input_dir = Path(args.input)
        output_dir = Path(args.output)

        if not input_dir.exists():
            print(f"ERROR: Input directory does not exist: {input_dir}")
            sys.exit(1)

        # Auto-detect species if needed
        species = args.species
        if species is None:
            species = auto_detect_species(input_dir, args.gpu)
        args.species = species

        config = build_config(args)
        pipeline = PetPipeline(config, gpu=args.gpu)
        pipeline.run_batch(input_dir, output_dir)

    elif args.command == "add":
        input_dir = Path(args.input)
        state_path = args.state
        output_dir = Path(args.output)

        if not input_dir.exists():
            print(f"ERROR: Input directory does not exist: {input_dir}")
            sys.exit(1)
        if not Path(state_path).exists():
            print(f"ERROR: State file does not exist: {state_path}")
            sys.exit(1)

        # Load config from state
        state = StateManager.load(state_path)
        config = state.config
        pipeline = PetPipeline(config, gpu=args.gpu)
        pipeline.run_incremental(input_dir, state_path, output_dir)


if __name__ == "__main__":
    main()

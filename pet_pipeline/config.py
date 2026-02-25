"""Constants, species configuration, and ONNX provider selection."""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

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
    cluster_algorithm: str = "hdbscan"
    agglomerative_threshold: float = 0.77
    cw_threshold: float = 0.45

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

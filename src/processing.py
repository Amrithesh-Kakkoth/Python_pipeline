"""Image processing: detection, alignment, cropping, and embedding."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import logger
from .detection import FaceDetector, BodyDetector, _align_face, _crop_body
from .embedding import Embedder


@dataclass
class ImageResult:
    """Processing result for a single image."""
    path: str
    face_emb: Optional[np.ndarray] = None
    body_emb: Optional[np.ndarray] = None
    has_face: bool = False
    has_body: bool = False


class ImageProcessor:
    """Detect -> align/crop -> embed for each image."""

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

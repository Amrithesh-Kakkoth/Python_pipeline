"""Face and body detection: FaceDetector, BodyDetector, and utility functions."""

from typing import Optional

import cv2
import numpy as np

from .config import IMAGENET_MEAN, IMAGENET_STD, logger


# ---------------------------------------------------------------------------
# Detection utilities
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
# FaceDetector
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
# BodyDetector
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

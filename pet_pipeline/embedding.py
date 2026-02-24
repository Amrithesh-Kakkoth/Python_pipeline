"""Face and body embedding extraction via ONNX."""

from typing import Optional

import numpy as np

from .detection import _preprocess_crop


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

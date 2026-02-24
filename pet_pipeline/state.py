"""Pipeline state save/load."""

from dataclasses import dataclass

import numpy as np

from .config import SpeciesConfig, logger


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

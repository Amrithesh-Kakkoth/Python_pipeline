"""Output organization: create cluster folders and summary."""

import json
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .config import SpeciesConfig, logger


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

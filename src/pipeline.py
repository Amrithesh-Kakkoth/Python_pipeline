"""Main PetPipeline class: batch and incremental clustering."""

import sys
import time
from pathlib import Path

import numpy as np

from .config import SpeciesConfig, IMAGE_EXTENSIONS, get_onnx_providers, logger
from .detection import FaceDetector, BodyDetector
from .embedding import Embedder
from .processing import ImageResult, ImageProcessor
from .clustering import ClusterEngine, renumber_labels
from .state import PipelineState, StateManager
from .output import OutputOrganizer


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
        """Full batch pipeline: scan -> detect -> embed -> cluster -> organize."""
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

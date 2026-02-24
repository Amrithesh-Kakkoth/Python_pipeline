"""Clustering engine: 3-phase fused clustering with body rescue and cross-merge."""

from collections import defaultdict

import numpy as np

from .config import SpeciesConfig, logger


class ClusterEngine:
    """3-phase fused clustering: face HDBSCAN -> body rescue -> cross-cluster merge."""

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
        """Phase 2: Rescue unclustered images using body similarity."""
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
        """Phase 3: Merge clusters that are separate in face space but similar in body space."""
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


def renumber_labels(labels: np.ndarray) -> np.ndarray:
    """Renumber cluster labels to contiguous 0..K-1, preserving -1."""
    out = labels.copy()
    unique = sorted(set(labels) - {-1})
    mapping = {old: new for new, old in enumerate(unique)}
    for i in range(len(out)):
        if out[i] != -1:
            out[i] = mapping[out[i]]
    return out

# Clustering Engine Documentation

The clustering engine (`src/clustering.py`) is the core intelligence behind the pet re-identification pipeline. It groups photos of the same individual pet together using a **3-phase fused clustering** approach that combines face and body embeddings.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Phase 1: Face-Based Clustering (HDBSCAN)](#phase-1-face-based-clustering-hdbscan)
- [Phase 2: Body Rescue](#phase-2-body-rescue)
- [Phase 2b: Body-Only Clustering](#phase-2b-body-only-clustering)
- [Phase 3: Cross-Cluster Merge](#phase-3-cross-cluster-merge)
- [Helper Functions](#helper-functions)
- [Species-Specific Configuration](#species-specific-configuration)
- [Execution Flow Diagram](#execution-flow-diagram)

---

## Overview

The pipeline receives two types of embeddings (numerical fingerprints) for each image:

| Embedding | Source | Dimension | Reliability |
|-----------|--------|-----------|-------------|
| **Face** | Aligned face crop → ONNX model | 128-d | High — faces are highly discriminative |
| **Body** | Body crop → ONNX model | Variable | Medium — body/fur patterns are less unique |

Not every image has both. A photo might have a detected face but no body, or vice versa. The clustering engine handles this gracefully across its phases.

**Label Convention:**
- `0, 1, 2, ...` → Assigned cluster IDs
- `-1` → Unclustered / noise

---

## Prerequisites

Before clustering begins, each image has been processed by the `ImageProcessor` to produce:

- `face_embeddings` — `(N, face_dim)` float32 matrix, L2-normalized
- `body_embeddings` — `(N, body_dim)` float32 matrix, L2-normalized
- `has_face` — `(N,)` boolean array, whether a face was detected
- `has_body` — `(N,)` boolean array, whether a body was detected

Because all embeddings are L2-normalized, **cosine similarity = dot product**, simplifying all similarity computations.

---

## Phase 1: Face-Based Clustering (HDBSCAN)

**Method:** `ClusterEngine.phase1_face_cluster()`

### Purpose

Form initial clusters using only face embeddings — the most discriminative signal for individual pet identification.

### Algorithm

1. **Select** only images where `has_face = True`.
2. **Compute pairwise cosine distance matrix:**
   ```
   similarity = face_embs @ face_embs.T       # (M, M) dot product
   distance   = clip(1.0 - similarity, 0, 2)   # convert to distance
   ```
3. **Run HDBSCAN** with `metric="precomputed"`:
   - `min_cluster_size` — minimum number of images to form a cluster (default: 2)
   - `min_samples` — controls density (default: 2)
   - `cluster_selection_method = "eom"` — Excess of Mass, tends to produce more fine-grained clusters
4. **Map face-level labels back** to the global index array.

### Why HDBSCAN?

- **No need to specify K** (number of clusters) — the algorithm discovers it automatically.
- **Handles noise** — data points that don't fit any cluster are labeled `-1`, which is exactly what we want for ambiguous images.
- **Density-based** — works well with embedding clusters of varying sizes and shapes.

### Output

A labels array of length N, where images without faces remain at `-1`.

---

## Phase 2: Body Rescue

**Method:** `ClusterEngine.phase2_body_rescue()`

### Purpose

Assign unclustered images (label = -1) to existing clusters using body embedding similarity, with a face-based veto mechanism to prevent incorrect assignments.

### Algorithm

1. **Build per-cluster body stacks** — for each cluster, collect all body embeddings of its members.
2. **For each unclustered image** that has a body embedding:
   a. Compute body similarity against **every member** of every cluster.
   b. Count how many members have similarity above `body_rescue_threshold`.
   c. If the count ≥ `min_body_agreements` **and** the average similarity is the highest seen → this is the best candidate cluster.
3. **Face Veto Check** (if the image also has a face):
   - Compute face similarity between the image's face and the candidate cluster's face centroid.
   - If `face_sim < face_veto_threshold` → **veto** the rescue. The face evidence says this is a different pet.
4. If not vetoed → **assign** the image to the best candidate cluster.

### Face Veto — Why?

Body embeddings are less discriminative than faces. Two different dogs with similar fur color might have similar body embeddings. The face veto prevents the body signal from overriding clear face-level disagreement.

### Tracking

The method returns a `rescue_log` — a list of dicts recording each rescue action:
- `"rescued"` — successfully assigned
- `"vetoed"` — face contradicted the body match

---

## Phase 2b: Body-Only Clustering

**Method:** `ClusterEngine.phase2b_body_cluster()`

### Purpose

After Phase 1 and Phase 2, some images are still unclustered. These typically have:
- No face detected (so Phase 1 skipped them)
- No strong body match to any existing cluster (so Phase 2 couldn't rescue them)

If enough of these body-only images exist, we cluster them independently.

### Algorithm

1. **Select** images where `label == -1` AND `has_body == True`.
2. If count < `min_cluster_size` → skip (not enough data).
3. **Compute pairwise cosine distance** on their body embeddings.
4. **Run HDBSCAN** (same parameters as Phase 1).
5. **Offset new cluster IDs** above the current maximum to avoid collisions.
   - e.g., if existing clusters are `{0, 1, 2}` and HDBSCAN produces `{0, 1}`, the new IDs become `{3, 4}`.

### Confidence Level

These clusters are **lower confidence** than face-based clusters since body embeddings are less unique. However, they are better than leaving images unclustered.

---

## Phase 3: Cross-Cluster Merge

**Method:** `ClusterEngine.phase3_cross_merge()`

### Purpose

HDBSCAN may over-split — creating two separate clusters for the same pet (e.g., "Max playing" vs "Max sleeping"). This phase merges clusters that are likely the same pet by comparing body embeddings across clusters, with a face contradiction check to prevent merging different pets.

### Algorithm

1. **Compute body centroids** (L2-normalized median) for each cluster.
2. **Pre-screen pairs** — only consider cluster pairs whose centroid similarity exceeds `body_merge_threshold × 0.7` (fast filter).
3. **For each candidate pair (C1, C2):**
   a. Compute **all pairwise body similarities** between C1 and C2 members.
   b. Check:
      - `avg_body_sim ≥ body_merge_threshold` ✓
      - `ratio of pairs above threshold ≥ min_body_overlap_ratio` ✓
   c. **Face contradiction check:**
      - Compute average face similarity between C1 and C2 face members.
      - If `avg_face_sim < face_contradiction_threshold` → **block** the merge.
4. **Sort** approved merge pairs by descending body similarity.
5. **Apply merges** using **Union-Find** (disjoint set):
   - Efficiently handles transitive merges (if A merges with B, and B merges with C, then all three become one cluster).
   - Uses path compression for performance.

### Why Union-Find?

Union-Find ensures that if Cluster A merges with B, and separately B merges with C, all three end up with the same label — without needing to re-check every combination.

### Face Contradiction — Why?

Two clusters might have similar body embeddings (e.g., two golden retrievers) but very different faces. The face contradiction check prevents merging genuinely different pets that just happen to look similar from the body.

---

## Helper Functions

### `compute_face_centroids()`

Computes the representative face embedding for each cluster.

- Uses **median** (not mean) — more robust to outliers (e.g., one bad face crop in a cluster).
- L2-normalizes the result.
- If a cluster has no face members, falls back to using all members (including zero-vector placeholders).

### `renumber_labels()`

After merging, cluster IDs can be sparse (e.g., `{0, 3, 7}`). This function renumbers them to contiguous `{0, 1, 2}` while preserving `-1` for unclustered images.

---

## Species-Specific Configuration

The clustering thresholds are tuned per species because dogs and cats have different visual characteristics:

| Parameter | Dog | Cat | Description |
|-----------|-----|-----|-------------|
| `face_weight` | 0.5 | 0.3 | Weight of face vs body in similarity scoring |
| `body_rescue_threshold` | 0.25 | 0.20 | Minimum body similarity for rescue |
| `min_body_agreements` | 3 | 2 | Minimum cluster members that must agree |
| `body_merge_threshold` | 0.35 | 0.30 | Minimum body similarity for cross-merge |
| `face_veto_threshold` | 0.05 | 0.05 | Below this face similarity → veto rescue |
| `face_contradiction_threshold` | 0.10 | 0.10 | Below this face similarity → block merge |
| `min_cluster_size` | 2 | 2 | HDBSCAN minimum cluster size |
| `min_samples` | 2 | 2 | HDBSCAN density parameter |
| `min_body_overlap_ratio` | 0.5 | 0.5 | Fraction of pairs that must exceed threshold |

**Why different thresholds?** Cat faces are generally harder to differentiate than dog faces, so `face_weight` is lower for cats (0.3 vs 0.5), relying more on body information.

---

## Execution Flow Diagram

```
Input: face_embeddings (N × 128), body_embeddings (N × D), has_face, has_body
        │
        ▼
┌──────────────────────────────────────────────┐
│  Phase 1: Face HDBSCAN                       │
│  ─ Only images with faces                    │
│  ─ Cosine distance → HDBSCAN                 │
│  ─ Output: initial cluster labels            │
│  ─ Some images remain at -1 (unclustered)    │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Phase 2: Body Rescue                        │
│  ─ For each unclustered image with body      │
│  ─ Compare body to all cluster members       │
│  ─ Assign if enough agreements               │
│  ─ Face veto prevents wrong assignments      │
│  ─ Output: fewer unclustered images          │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Phase 2b: Body-Only Clustering              │
│  ─ Still-unclustered images with bodies      │
│  ─ HDBSCAN on body embeddings                │
│  ─ New cluster IDs offset above existing max │
│  ─ Output: additional body-only clusters     │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Phase 3: Cross-Cluster Merge                │
│  ─ Compare all cluster pairs via body        │
│  ─ Pre-screen with centroid similarity       │
│  ─ Face contradiction blocks bad merges      │
│  ─ Union-Find for transitive merging         │
│  ─ Output: final merged cluster labels       │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  renumber_labels()                           │
│  ─ Sparse IDs → contiguous 0, 1, 2, ...     │
│  ─ -1 preserved for unclustered              │
└──────────────────────────────────────────────┘
                   │
                   ▼
Output: labels array (N,) — one cluster ID per image
```

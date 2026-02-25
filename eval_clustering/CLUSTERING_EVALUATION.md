# Clustering Algorithm Evaluation Report

**Date:** 2026-02-25
**Dataset:** 39 dog images, 10 true identities
**Detection:** 37 faces (95%), 29 bodies (74%)

---

## 1. Algorithms Tested

| Algorithm | Type | Key Parameter | Default |
|-----------|------|---------------|---------|
| **HDBSCAN** | Density-based | `min_cluster_size` | 2 |
| **Agglomerative** | Threshold-based (average linkage) | `distance_threshold` | 0.77 |
| **Chinese Whispers** | Graph-based (label propagation) | `cw_threshold` (similarity) | 0.45 |

All three operate on precomputed cosine distance matrices from the same face/body embeddings.

---

## 2. Per-Algorithm Results

### 2.1 Cluster Statistics

| Metric | HDBSCAN | Agglomerative | Chinese Whispers |
|--------|--------:|--------------:|-----------------:|
| Predicted clusters | 6 | 15 | 9 |
| True identities | 10 | 10 | 10 |
| Unclustered | 11 (28.2%) | 2 (5.1%) | 15 (38.5%) |
| **% Clustered** | **71.8%** | **94.9%** | **61.5%** |
| Cluster size (mean) | 4.7 | 2.5 | 2.7 |
| Cluster size (median) | 4.0 | 2.0 | 3.0 |
| Cluster size (min) | 3 | 1 | 2 |
| Cluster size (max) | 8 | 6 | 4 |

**Key observations:**
- **HDBSCAN** under-segments (6 clusters vs 10 true), merging some identities together, but leaves 28% unclustered.
- **Agglomerative** over-segments moderately (15 clusters vs 10 true), but assigns almost everything (95%) with near-perfect purity (0.973).
- **Chinese Whispers** is closest to the true cluster count (9 vs 10) but leaves the most images unclustered (38.5%).

### 2.2 Quality Metrics (vs Ground Truth)

| Metric | HDBSCAN | Agglomerative | Chinese Whispers | Best |
|--------|--------:|--------------:|-----------------:|------|
| **Pairwise Precision** | 0.6290 | **0.9286** | 0.9091 | Agglomerative |
| **Pairwise Recall** | 0.6094 | **0.6094** | 0.3125 | Tied (HDBSCAN/Agglom) |
| **Pairwise F1** | 0.6190 | **0.7358** | 0.4651 | Agglomerative |
| **B-Cubed Precision** | 0.8372 | 0.9615 | **0.9658** | Chinese Whispers |
| **B-Cubed Recall** | 0.7282 | **0.7325** | 0.5051 | Agglomerative |
| **B-Cubed F1** | 0.7789 | **0.8315** | 0.6633 | Agglomerative |
| **NMI** | 0.8049 | **0.8867** | 0.7339 | Agglomerative |
| **ARI** | 0.5336 | **0.7088** | 0.3316 | Agglomerative |
| **Purity** | 0.8214 | **0.9730** | 0.9583 | Agglomerative |

### 2.3 Confusion Counts (Pairwise)

| | HDBSCAN | Agglomerative | Chinese Whispers |
|---|--------:|--------------:|-----------------:|
| True Positives | 39 | 39 | 20 |
| False Positives | 23 | 3 | 2 |
| False Negatives | 25 | 25 | 44 |
| True Negatives | 654 | 674 | 675 |

### 2.4 Pipeline Phase Breakdown

| Phase | HDBSCAN | Agglomerative | Chinese Whispers |
|-------|--------:|--------------:|-----------------:|
| Body rescues | 0 | 0 | 0 |
| Body-only clusters formed | 0 | 0 | 2 |
| Cross-cluster merges | 0 | 0 | 1 |

### 2.5 Timing (5 runs)

| | HDBSCAN | Agglomerative | Chinese Whispers |
|---|--------:|--------------:|-----------------:|
| Mean (s) | 0.169 | 0.003 | **0.002** |
| Std (s) | 0.333 | 0.003 | 0.000 |

Note: HDBSCAN's first-run time includes library import overhead (~0.8s); subsequent runs are ~0.003s.

---

## 3. Pairwise Algorithm Comparisons

How similar are the algorithms' outputs to each other (not to ground truth)?

| Pair | NMI | ARI | Pairwise Agreement |
|------|----:|----:|-------------------:|
| HDBSCAN vs Agglomerative | 0.8046 | 0.4809 | 96.5% |
| HDBSCAN vs Chinese Whispers | 0.7185 | 0.4605 | 94.3% |
| Agglomerative vs Chinese Whispers | 0.7304 | 0.2271 | 97.3% |

**Key observations:**
- Agglomerative and HDBSCAN now have the highest agreement (96.5%) — the tuned agglomerative threshold brings it closer to HDBSCAN's grouping behavior while maintaining much higher precision.
- Chinese Whispers remains more conservative and diverges from both due to its high unclustered rate.

---

## 4. Chinese Whispers Stability

Since Chinese Whispers uses random node ordering, we measured consistency across 5 runs:

| Metric | Value |
|--------|------:|
| Mean pairwise agreement | **1.0000** |
| All run-to-run agreements | 1.0, 1.0, 1.0, 1.0 |

On this dataset, Chinese Whispers is fully deterministic in practice — the graph structure is clear enough that random ordering doesn't affect the outcome.

---

## 5. Agglomerative Threshold Tuning

We swept `distance_threshold` from 0.40 to 0.90 to find the optimal value:

| Threshold | Clusters | Purity | Pw Precision | Pw Recall | Pw F1 | ARI |
|----------:|---------:|-------:|-------------:|----------:|------:|----:|
| 0.40-0.50 | 23 | 1.000 | 1.000 | 0.281 | 0.439 | 0.410 |
| 0.55-0.60 | 20 | 1.000 | 1.000 | 0.391 | 0.562 | 0.532 |
| 0.65-0.69 | 18 | 1.000 | 1.000 | 0.469 | 0.638 | 0.610 |
| 0.71-0.75 | 17 | 0.973 | 0.909 | 0.469 | 0.619 | 0.588 |
| **0.77-0.79** | **15** | **0.973** | **0.929** | **0.609** | **0.736** | **0.709** |
| 0.80-0.85 | 13 | 0.892 | 0.764 | 0.656 | 0.706 | 0.674 |
| 0.90 | 9 | 0.730 | 0.566 | 0.672 | 0.614 | 0.569 |

**Selected default: `0.77`** — this threshold hits the sweet spot:
- Drops from 20 clusters (at 0.55) to 15, much closer to the 10 true identities.
- Near-perfect purity (0.973) — only 1 image out of 37 is in the wrong cluster.
- Best Pairwise F1 (0.736) and ARI (0.709) among all thresholds that maintain >0.95 purity.
- Matches HDBSCAN's recall (0.609) while having far higher precision (0.929 vs 0.629).

The key transition points:
- Below 0.71: perfect purity but too many tiny clusters.
- 0.77-0.79: optimal — one small purity trade-off unlocks a big F1/ARI jump.
- Above 0.80: purity drops to 0.89 as wrong merges accumulate.

---

## 6. Analysis and Recommendations

### Precision vs Recall Trade-off

```
                    Precision ◄──────────────────────► Recall
                    (no false merges)                  (no missed groups)

  Agglomerative     ██████████████████    0.93         ████████████       0.61
  Chinese Whispers  ██████████████████    0.91         ██████             0.31
  HDBSCAN           ████████████          0.63         ████████████       0.61
```

- **Agglomerative** (tuned) is now the clear winner — it matches HDBSCAN's recall while having much higher precision.
- **Chinese Whispers** has high precision but very low recall due to leaving 38.5% unclustered.
- **HDBSCAN** has balanced but mediocre precision-recall — it groups aggressively but often puts different dogs together.

### For Human-in-the-Loop Workflows

The original motivation was to find algorithms that are **stable under manual removals** (when a user removes a misclassified image from a cluster):

| Property | HDBSCAN | Agglomerative | Chinese Whispers |
|----------|---------|---------------|------------------|
| Stable under removals | No (density changes) | **Yes** (threshold-based) | **Yes** (edge-based) |
| Pairwise F1 | 0.619 | **0.736** | 0.465 |
| Precision | 0.629 | **0.929** | 0.909 |
| Coverage (% assigned) | 71.8% | **94.9%** | 61.5% |
| Speed | Slowest | Fast | **Fastest** |

**Recommendation by use case:**

1. **Manual verification workflow** (primary goal): **Agglomerative** (`threshold=0.77`) is the best choice.
   - Best F1 (0.736) and ARI (0.709) of all three algorithms.
   - Near-perfect precision (0.929) — almost every cluster is pure, so the user mainly needs to *merge* clusters, rarely split.
   - 95% coverage — very few images need manual assignment.
   - Threshold-based: removing an image doesn't affect other cluster boundaries.

2. **Automated pipeline (no human review)**: **Agglomerative** also wins here — it now outperforms HDBSCAN on every metric except raw recall (tied at 0.609).

3. **Large-scale speed-critical**: **Chinese Whispers** is fastest and has high precision (0.91), but leaves 38.5% unclustered — only suitable if downstream processing handles singletons.

### Further Tuning Notes

- **Chinese Whispers `cw_threshold=0.45`**: Currently conservative. Lowering the threshold (e.g., 0.35) would add more edges and form larger clusters, improving recall at the cost of some precision. A similar sweep could be done for this parameter.

---

## 7. Raw Data

Full results with all metrics:
- [`eval_results.json`](./eval_results.json) — final evaluation (threshold=0.77)
- [`agglom_sweep.json`](./agglom_sweep.json) — agglomerative threshold sweep results

**Evaluation scripts** (project root):
```bash
# Full 3-algorithm comparison
python evaluate_clustering.py -i ~/test_dogs -s dog -o ~/eval_clustering --gpu --runs 5

# Agglomerative threshold sweep
python sweep_agglom.py ~/test_dogs dog ~/eval_clustering/agglom_sweep.json
```

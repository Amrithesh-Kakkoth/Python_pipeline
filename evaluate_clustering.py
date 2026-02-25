#!/usr/bin/env python3
"""Comprehensive evaluation of clustering algorithms.

Runs all three algorithms (HDBSCAN, Agglomerative, Chinese Whispers) on the
test dataset, computes per-algorithm metrics, pairwise comparisons, and
writes a full report.

Ground truth: folder structure in the input directory (each subfolder = one identity).
"""

import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from pet_pipeline.clustering import ClusterEngine, renumber_labels
from pet_pipeline.config import SPECIES_DEFAULTS, SpeciesConfig
from pet_pipeline.pipeline import PetPipeline


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_pairwise_matrices(labels_pred, labels_true):
    """Compute TP, FP, FN, TN from pairwise same/different decisions."""
    n = len(labels_pred)
    tp = fp = fn = tn = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_pred = (labels_pred[i] == labels_pred[j]) and labels_pred[i] != -1
            same_true = (labels_true[i] == labels_true[j]) and labels_true[i] != -1
            if same_pred and same_true:
                tp += 1
            elif same_pred and not same_true:
                fp += 1
            elif not same_pred and same_true:
                fn += 1
            else:
                tn += 1
    return tp, fp, fn, tn


def pairwise_precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def pairwise_recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def pairwise_f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def bcubed_precision_recall(labels_pred, labels_true):
    """Compute B-Cubed precision and recall (per-item, then averaged)."""
    n = len(labels_pred)
    precisions = []
    recalls = []
    for i in range(n):
        if labels_pred[i] == -1:
            # Treat unclustered as singleton
            pred_cluster = {i}
        else:
            pred_cluster = {j for j in range(n) if labels_pred[j] == labels_pred[i]}

        true_cluster = {j for j in range(n) if labels_true[j] == labels_true[i]}

        correct = len(pred_cluster & true_cluster)
        precisions.append(correct / len(pred_cluster))
        recalls.append(correct / len(true_cluster))

    avg_p = np.mean(precisions)
    avg_r = np.mean(recalls)
    return float(avg_p), float(avg_r)


def normalized_mutual_info(labels_pred, labels_true):
    """NMI between two label arrays (treats -1 as a valid label)."""
    # Map to contiguous ints
    def _map(labels):
        uniq = sorted(set(labels))
        m = {v: i for i, v in enumerate(uniq)}
        return np.array([m[v] for v in labels])

    a = _map(labels_pred)
    b = _map(labels_true)
    n = len(a)

    # Contingency table
    max_a, max_b = a.max() + 1, b.max() + 1
    contingency = np.zeros((max_a, max_b), dtype=np.float64)
    for i in range(n):
        contingency[a[i], b[i]] += 1

    # Marginals
    row_sum = contingency.sum(axis=1)
    col_sum = contingency.sum(axis=0)

    # Entropies
    def entropy(counts):
        p = counts[counts > 0] / counts.sum()
        return -np.sum(p * np.log(p))

    h_a = entropy(row_sum)
    h_b = entropy(col_sum)

    # Mutual information
    mi = 0.0
    for i in range(max_a):
        for j in range(max_b):
            if contingency[i, j] > 0:
                mi += contingency[i, j] / n * np.log(
                    n * contingency[i, j] / (row_sum[i] * col_sum[j])
                )

    # NMI
    denom = (h_a + h_b) / 2
    if denom == 0:
        return 1.0 if h_a == h_b == 0 else 0.0
    return float(mi / denom)


def adjusted_rand_index(labels_pred, labels_true):
    """ARI between two label arrays."""
    def _map(labels):
        uniq = sorted(set(labels))
        m = {v: i for i, v in enumerate(uniq)}
        return np.array([m[v] for v in labels])

    a = _map(labels_pred)
    b = _map(labels_true)
    n = len(a)

    max_a, max_b = a.max() + 1, b.max() + 1
    contingency = np.zeros((max_a, max_b), dtype=np.int64)
    for i in range(n):
        contingency[a[i], b[i]] += 1

    row_sum = contingency.sum(axis=1)
    col_sum = contingency.sum(axis=0)

    def comb2(x):
        return x * (x - 1) / 2

    sum_comb_c = sum(comb2(contingency[i, j]) for i in range(max_a) for j in range(max_b))
    sum_comb_a = sum(comb2(r) for r in row_sum)
    sum_comb_b = sum(comb2(c) for c in col_sum)

    comb_n = comb2(n)
    expected = sum_comb_a * sum_comb_b / comb_n if comb_n > 0 else 0
    max_index = (sum_comb_a + sum_comb_b) / 2
    denom = max_index - expected

    if denom == 0:
        return 1.0
    return float((sum_comb_c - expected) / denom)


def cluster_purity(labels_pred, labels_true):
    """Average purity: for each predicted cluster, fraction of dominant true label."""
    pred_clusters = defaultdict(list)
    for i, lbl in enumerate(labels_pred):
        if lbl != -1:
            pred_clusters[lbl].append(labels_true[i])

    if not pred_clusters:
        return 0.0

    total = 0
    correct = 0
    for members in pred_clusters.values():
        counter = Counter(members)
        total += len(members)
        correct += counter.most_common(1)[0][1]

    return correct / total if total > 0 else 0.0


def compute_all_metrics(labels_pred, labels_true, elapsed):
    """Compute the full metrics suite for one algorithm."""
    pred = np.array(labels_pred)
    true = np.array(labels_true)

    n_pred_clusters = len(set(pred) - {-1})
    n_true_clusters = len(set(true) - {-1})
    n_unclustered = int((pred == -1).sum())
    n_total = len(pred)

    # Cluster sizes
    sizes = []
    for c in sorted(set(pred) - {-1}):
        sizes.append(int((pred == c).sum()))

    # Pairwise
    tp, fp, fn, tn = compute_pairwise_matrices(pred, true)
    pw_prec = pairwise_precision(tp, fp)
    pw_rec = pairwise_recall(tp, fn)
    pw_f1 = pairwise_f1(pw_prec, pw_rec)

    # B-Cubed
    bc_prec, bc_rec = bcubed_precision_recall(pred, true)
    bc_f1 = pairwise_f1(bc_prec, bc_rec)

    # NMI & ARI
    nmi = normalized_mutual_info(pred, true)
    ari = adjusted_rand_index(pred, true)

    # Purity
    purity = cluster_purity(pred, true)

    return {
        "n_images": n_total,
        "n_predicted_clusters": n_pred_clusters,
        "n_true_identities": n_true_clusters,
        "n_unclustered": n_unclustered,
        "pct_clustered": round(100 * (n_total - n_unclustered) / n_total, 1),
        "cluster_sizes": sizes,
        "cluster_size_mean": round(np.mean(sizes), 1) if sizes else 0,
        "cluster_size_median": round(float(np.median(sizes)), 1) if sizes else 0,
        "cluster_size_min": int(min(sizes)) if sizes else 0,
        "cluster_size_max": int(max(sizes)) if sizes else 0,
        "pairwise_precision": round(pw_prec, 4),
        "pairwise_recall": round(pw_rec, 4),
        "pairwise_f1": round(pw_f1, 4),
        "bcubed_precision": round(bc_prec, 4),
        "bcubed_recall": round(bc_rec, 4),
        "bcubed_f1": round(bc_f1, 4),
        "nmi": round(nmi, 4),
        "ari": round(ari, 4),
        "purity": round(purity, 4),
        "elapsed_seconds": round(elapsed, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compare_two_algorithms(labels_a, labels_b, name_a, name_b):
    """Compare two algorithm outputs against each other."""
    a = np.array(labels_a)
    b = np.array(labels_b)

    nmi = normalized_mutual_info(a, b)
    ari = adjusted_rand_index(a, b)

    # Agreement rate: fraction of pairs where both agree (same/different)
    n = len(a)
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_a = (a[i] == a[j]) and a[i] != -1
            same_b = (b[i] == b[j]) and b[i] != -1
            if same_a == same_b:
                agree += 1
            total += 1

    agreement = agree / total if total > 0 else 0.0

    return {
        "algorithms": [name_a, name_b],
        "nmi": round(nmi, 4),
        "ari": round(ari, 4),
        "pairwise_agreement": round(agreement, 4),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def extract_embeddings(input_dir, species, gpu=False):
    """Run the pipeline's detection+embedding, return embeddings and ground truth."""
    from dataclasses import asdict as _ad
    config = SpeciesConfig(**_ad(SPECIES_DEFAULTS[species]))

    pipeline = PetPipeline(config, gpu=gpu)

    # Scan
    image_paths = pipeline._scan_images(input_dir)
    if not image_paths:
        return None

    # Process (detect + embed)
    results = pipeline._process_images(image_paths)

    # Build matrices
    paths, face_embs, body_embs, has_face, has_body = pipeline._build_matrices(results)

    # Ground truth from folder names
    folder_names = sorted(set(Path(p).parent.name for p in paths))
    folder_map = {name: i for i, name in enumerate(folder_names)}
    gt_labels = np.array([folder_map[Path(p).parent.name] for p in paths], dtype=np.int32)

    return {
        "paths": paths,
        "face_embs": face_embs,
        "body_embs": body_embs,
        "has_face": has_face,
        "has_body": has_body,
        "gt_labels": gt_labels,
    }


def run_clustering_only(face_embs, body_embs, has_face, has_body, species, algorithm):
    """Re-run just the clustering phases (skip detection/embedding)."""
    from dataclasses import asdict as _ad
    config = SpeciesConfig(**_ad(SPECIES_DEFAULTS[species]))
    config.cluster_algorithm = algorithm

    engine = ClusterEngine(config)

    t0 = time.time()
    labels = engine.phase1_face_cluster(face_embs, has_face)
    face_centroids = engine.compute_face_centroids(face_embs, labels, has_face)
    labels, rescue_log = engine.phase2_body_rescue(
        labels, face_embs, body_embs, has_face, has_body, face_centroids)
    labels, body_cluster_log = engine.phase2b_body_cluster(labels, body_embs, has_body)
    labels, merge_log = engine.phase3_cross_merge(
        labels, face_embs, body_embs, has_face, has_body)
    labels = renumber_labels(labels)
    elapsed = time.time() - t0

    return labels, elapsed, {
        "rescue_count": len([r for r in rescue_log if r["action"] == "rescued"]),
        "body_cluster_count": len([b for b in body_cluster_log if b["action"] == "body_clustered"]),
        "merge_count": len([m for m in merge_log if m["action"] == "merged"]),
    }


def main():
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Evaluate clustering algorithms")
    parser.add_argument("--input", "-i", required=True, help="Input directory (with identity subfolders)")
    parser.add_argument("--species", "-s", required=True, choices=["dog", "cat"])
    parser.add_argument("--output", "-o", required=True, help="Output directory for evaluation results")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per algorithm for timing stability (default: 5)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("pet_pipeline")

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    algorithms = ["hdbscan", "agglomerative", "chinese_whispers"]

    print("=" * 70)
    print("CLUSTERING ALGORITHM EVALUATION")
    print("=" * 70)
    print(f"Input:   {input_dir}")
    print(f"Species: {args.species}")
    print(f"Runs:    {args.runs}")
    print()

    # --- Step 1: Extract embeddings (shared across all algorithms) ---
    print("[1/4] Detecting and embedding (shared step)...")
    data = extract_embeddings(input_dir, args.species, gpu=args.gpu)

    if data is None:
        print("ERROR: No images found!")
        sys.exit(1)

    face_embs = data["face_embs"]
    body_embs = data["body_embs"]
    has_face = data["has_face"]
    has_body = data["has_body"]
    paths = data["paths"]
    gt_labels = data["gt_labels"]

    n_true = len(set(gt_labels))
    print(f"  Images: {len(paths)}")
    print(f"  Faces:  {has_face.sum()}")
    print(f"  Bodies: {has_body.sum()}")
    print(f"  True identities: {n_true}")
    print()

    # --- Step 2: Run each algorithm multiple times ---
    print(f"[2/4] Running each algorithm ({args.runs} runs each)...")
    all_results = {}

    for algo in algorithms:
        print(f"\n  --- {algo.upper()} ---")
        run_labels = []
        run_times = []
        run_extras = []

        for run_i in range(args.runs):
            labels, elapsed, extras = run_clustering_only(
                face_embs, body_embs, has_face, has_body, args.species, algo)
            run_labels.append(labels)
            run_times.append(elapsed)
            run_extras.append(extras)
            print(f"    Run {run_i+1}: {len(set(labels)-{-1})} clusters, "
                  f"{(labels==-1).sum()} unclustered, {elapsed:.3f}s")

        # Use the first run's labels for metrics (deterministic for hdbscan/agglom,
        # representative for CW)
        best_idx = 0
        all_results[algo] = {
            "labels": run_labels[best_idx],
            "all_labels": run_labels,
            "times": run_times,
            "extras": run_extras[best_idx],
            "mean_time": np.mean(run_times),
            "std_time": np.std(run_times),
        }

    # --- Step 3: Compute metrics ---
    print(f"\n[3/4] Computing metrics...")
    metrics = {}
    for algo in algorithms:
        r = all_results[algo]
        m = compute_all_metrics(r["labels"], gt_labels, r["mean_time"])
        m["timing_std"] = round(r["std_time"], 4)
        m["rescue_count"] = r["extras"]["rescue_count"]
        m["body_cluster_count"] = r["extras"]["body_cluster_count"]
        m["merge_count"] = r["extras"]["merge_count"]
        metrics[algo] = m
        print(f"  {algo}: F1(pw)={m['pairwise_f1']}, F1(bc)={m['bcubed_f1']}, "
              f"NMI={m['nmi']}, ARI={m['ari']}, purity={m['purity']}")

    # --- Step 4: Pairwise comparisons ---
    print(f"\n[4/4] Computing pairwise comparisons...")
    comparisons = []
    for a1, a2 in combinations(algorithms, 2):
        comp = compare_two_algorithms(
            all_results[a1]["labels"], all_results[a2]["labels"], a1, a2)
        comparisons.append(comp)
        print(f"  {a1} vs {a2}: NMI={comp['nmi']}, ARI={comp['ari']}, "
              f"agreement={comp['pairwise_agreement']}")

    # Chinese Whispers stability (across runs)
    cw_labels_runs = all_results["chinese_whispers"]["all_labels"]
    if len(cw_labels_runs) > 1:
        cw_stability = []
        for i in range(1, len(cw_labels_runs)):
            comp = compare_two_algorithms(
                cw_labels_runs[0], cw_labels_runs[i], "cw_run0", f"cw_run{i}")
            cw_stability.append(comp["pairwise_agreement"])
        cw_mean_stability = float(np.mean(cw_stability))
    else:
        cw_stability = []
        cw_mean_stability = 1.0

    # --- Build full report ---
    report = {
        "dataset": {
            "input_dir": str(input_dir),
            "species": args.species,
            "n_images": len(paths),
            "n_true_identities": n_true,
            "n_faces_detected": int(has_face.sum()),
            "n_bodies_detected": int(has_body.sum()),
            "identity_sizes": {
                name: int((gt_labels == i).sum())
                for name, i in sorted(
                    {Path(p).parent.name: gt_labels[idx]
                     for idx, p in enumerate(paths)}.items(),
                    key=lambda x: x[1])
            },
        },
        "per_algorithm": metrics,
        "pairwise_comparisons": comparisons,
        "chinese_whispers_stability": {
            "n_runs": len(cw_labels_runs),
            "run_agreements": [round(s, 4) for s in cw_stability],
            "mean_stability": round(cw_mean_stability, 4),
        },
    }

    # Save JSON
    report_path = output_dir / "eval_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {report_path}")

    # --- Print formatted report ---
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    print(f"\nDataset: {args.species} | {len(paths)} images | {n_true} true identities")
    print(f"  Faces detected: {has_face.sum()} ({100*has_face.mean():.0f}%)")
    print(f"  Bodies detected: {has_body.sum()} ({100*has_body.mean():.0f}%)")

    print(f"\n{'':4s}{'Metric':<24s}", end="")
    for algo in algorithms:
        print(f"{algo:>18s}", end="")
    print()
    print("    " + "-" * (24 + 18 * len(algorithms)))

    rows = [
        ("Predicted clusters", "n_predicted_clusters", "d"),
        ("Unclustered", "n_unclustered", "d"),
        ("% Clustered", "pct_clustered", ".1f"),
        ("Cluster size (mean)", "cluster_size_mean", ".1f"),
        ("Cluster size (median)", "cluster_size_median", ".1f"),
        ("Cluster size (min)", "cluster_size_min", "d"),
        ("Cluster size (max)", "cluster_size_max", "d"),
        ("", None, None),
        ("Pairwise Precision", "pairwise_precision", ".4f"),
        ("Pairwise Recall", "pairwise_recall", ".4f"),
        ("Pairwise F1", "pairwise_f1", ".4f"),
        ("", None, None),
        ("B-Cubed Precision", "bcubed_precision", ".4f"),
        ("B-Cubed Recall", "bcubed_recall", ".4f"),
        ("B-Cubed F1", "bcubed_f1", ".4f"),
        ("", None, None),
        ("NMI", "nmi", ".4f"),
        ("ARI", "ari", ".4f"),
        ("Purity", "purity", ".4f"),
        ("", None, None),
        ("Rescued (body)", "rescue_count", "d"),
        ("Body-only clusters", "body_cluster_count", "d"),
        ("Cross-merges", "merge_count", "d"),
        ("", None, None),
        ("Time (mean, s)", "elapsed_seconds", ".3f"),
        ("Time (std, s)", "timing_std", ".4f"),
    ]

    for label, key, fmt in rows:
        if key is None:
            print()
            continue
        print(f"    {label:<24s}", end="")
        for algo in algorithms:
            val = metrics[algo][key]
            print(f"{val:>18{fmt}}", end="")
        print()

    # Best algorithm per metric (higher is better for most)
    higher_better = ["pairwise_precision", "pairwise_recall", "pairwise_f1",
                     "bcubed_precision", "bcubed_recall", "bcubed_f1",
                     "nmi", "ari", "purity", "pct_clustered"]
    lower_better = ["elapsed_seconds"]

    print(f"\n{'':4s}{'Best per metric':}")
    print("    " + "-" * 50)
    for label, key, fmt in rows:
        if key is None or key not in higher_better + lower_better:
            continue
        vals = {algo: metrics[algo][key] for algo in algorithms}
        if key in higher_better:
            best = max(vals, key=vals.get)
        else:
            best = min(vals, key=vals.get)
        print(f"    {label:<24s} -> {best}")

    print(f"\nPairwise Algorithm Comparisons:")
    print("    " + "-" * 60)
    print(f"    {'Pair':<40s} {'NMI':>8s} {'ARI':>8s} {'Agree':>8s}")
    for comp in comparisons:
        pair = f"{comp['algorithms'][0]} vs {comp['algorithms'][1]}"
        print(f"    {pair:<40s} {comp['nmi']:>8.4f} {comp['ari']:>8.4f} {comp['pairwise_agreement']:>8.4f}")

    print(f"\nChinese Whispers Stability ({len(cw_labels_runs)} runs):")
    print(f"    Mean pairwise agreement across runs: {cw_mean_stability:.4f}")

    print("\n" + "=" * 70)
    print("END OF REPORT")
    print("=" * 70)


if __name__ == "__main__":
    main()

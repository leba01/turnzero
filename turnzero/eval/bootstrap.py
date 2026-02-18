"""Cluster-aware bootstrap confidence intervals.

Resamples core_cluster_a groups with replacement and recomputes all
metrics via compute_metrics() to produce percentile CIs.

Reference: docs/PROJECT_BIBLE.md Section 6.4
"""

from __future__ import annotations

import sys

import numpy as np

from turnzero.eval.metrics import compute_metrics


def cluster_bootstrap_ci(
    probs: np.ndarray,
    action90_true: np.ndarray,
    lead2_true: np.ndarray,
    bring4_observed: np.ndarray,
    is_mirror: np.ndarray,
    cluster_ids: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Cluster-aware bootstrap CIs for all metrics.

    Resampling strategy:
    - Get unique cluster IDs.
    - For each bootstrap iteration:
      * Sample cluster IDs with replacement.
      * Gather all examples belonging to sampled clusters.
      * Compute all metrics via compute_metrics().
    - Report percentile CIs.

    Args:
        probs: (N, 90) predicted 90-way action probabilities.
        action90_true: (N,) ground-truth action90 ids.
        lead2_true: (N,) ground-truth lead-pair index (0..14).
        bring4_observed: (N,) bool — True for Tier 1 examples.
        is_mirror: (N,) bool — True when core_cluster_a == core_cluster_b.
        cluster_ids: (N,) str or int — core_cluster_a per example.
        n_bootstrap: number of bootstrap iterations.
        ci_level: confidence level (default 0.95 for 95% CIs).
        seed: random seed for reproducibility.

    Returns:
        Dict keyed by metric name (e.g. ``"overall/top1_action90"``) with
        values ``{"mean": ..., "lo": ..., "hi": ..., "std": ...}``.
    """
    probs = np.asarray(probs, dtype=np.float64)
    action90_true = np.asarray(action90_true, dtype=np.int64)
    lead2_true = np.asarray(lead2_true, dtype=np.int64)
    bring4_observed = np.asarray(bring4_observed, dtype=bool)
    is_mirror = np.asarray(is_mirror, dtype=bool)

    N = len(probs)
    assert probs.shape == (N, 90), f"Expected (N, 90) probs, got {probs.shape}"

    # Build cluster -> example indices mapping
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    cluster_to_indices: dict[int, np.ndarray] = {}
    for i, cid in enumerate(unique_clusters):
        cluster_to_indices[i] = np.where(cluster_ids == cid)[0]

    alpha = 1.0 - ci_level
    rng = np.random.default_rng(seed)

    # Collect bootstrap metric samples
    all_samples: dict[str, list[float]] = {}

    for b in range(n_bootstrap):
        if (b + 1) % 100 == 0 or b == 0:
            print(
                f"  Bootstrap iteration {b + 1}/{n_bootstrap}",
                flush=True,
                file=sys.stderr,
            )

        # Sample cluster indices with replacement
        sampled_cluster_idx = rng.integers(0, n_clusters, size=n_clusters)

        # Gather all example indices for sampled clusters
        idx_parts = [cluster_to_indices[ci] for ci in sampled_cluster_idx]
        boot_idx = np.concatenate(idx_parts)

        # Recompute metrics on bootstrapped sample
        metrics = compute_metrics(
            probs=probs[boot_idx],
            action90_true=action90_true[boot_idx],
            lead2_true=lead2_true[boot_idx],
            bring4_observed=bring4_observed[boot_idx],
            is_mirror=is_mirror[boot_idx],
        )

        for key, val in metrics.items():
            # Skip count keys (n_examples, n_tier1) — not meaningful as CIs
            if key.endswith("/n_examples") or key.endswith("/n_tier1"):
                continue
            if val is None:
                continue
            if key not in all_samples:
                all_samples[key] = []
            all_samples[key].append(float(val))

    # Compute percentile CIs
    results: dict[str, dict[str, float]] = {}
    for key, samples in all_samples.items():
        arr = np.array(samples)
        results[key] = {
            "mean": float(np.mean(arr)),
            "lo": float(np.percentile(arr, 100 * alpha / 2)),
            "hi": float(np.percentile(arr, 100 * (1 - alpha / 2))),
            "std": float(np.std(arr)),
        }

    return results

#!/usr/bin/env python3
"""Evaluate popularity and logistic baselines on Regime A test split.

Produces:
  - outputs/baselines/metrics_popularity.json
  - outputs/baselines/metrics_popularity_cond.json
  - outputs/baselines/metrics_logistic.json
  - outputs/baselines/reliability_popularity.png/.pdf
  - outputs/baselines/reliability_popularity_cond.png/.pdf
  - outputs/baselines/reliability_logistic.png/.pdf
  - outputs/baselines/topk_comparison.png/.pdf
  - outputs/baselines/stratified_table.json
  - outputs/baselines/stratified_table.tex

Usage:
    python scripts/eval_baselines.py [--split_dir data/assembled/regime_a]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from turnzero.data.io_utils import read_jsonl
from turnzero.eval.metrics import compute_metrics
from turnzero.eval.plots import (
    reliability_diagram,
    stratified_table,
    topk_comparison_bar,
)
from turnzero.models.baselines import LogisticBaseline, PopularityBaseline


def _extract_arrays(
    examples: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Extract label arrays and cluster lists from raw example dicts."""
    from itertools import combinations

    lead_pairs = list(combinations(range(6), 2))
    lead_pair_to_idx = {pair: idx for idx, pair in enumerate(lead_pairs)}

    action90 = np.array([ex["label"]["action90_id"] for ex in examples], dtype=np.int64)
    lead2 = np.array(
        [lead_pair_to_idx[tuple(sorted(ex["label"]["lead2_idx"]))] for ex in examples],
        dtype=np.int64,
    )
    bring4_obs = np.array(
        [ex["label_quality"]["bring4_observed"] for ex in examples], dtype=bool
    )
    is_mirror = np.array(
        [ex.get("split_keys", {}).get("is_mirror", False) for ex in examples],
        dtype=bool,
    )
    cluster_a = [ex.get("split_keys", {}).get("core_cluster_a", "UNK") for ex in examples]
    cluster_b = [ex.get("split_keys", {}).get("core_cluster_b", "UNK") for ex in examples]

    return action90, lead2, bring4_obs, is_mirror, cluster_a, cluster_b


def _print_metrics(name: str, metrics: dict) -> None:
    """Pretty-print selected metrics."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for stratum in ["overall", "mirror", "non_mirror"]:
        n = metrics.get(f"{stratum}/n_examples", 0)
        n_t1 = metrics.get(f"{stratum}/n_tier1", 0)
        print(f"\n  [{stratum}] n={n}, tier1={n_t1}")

        # Action90 (Tier 1)
        t1 = metrics.get(f"{stratum}/top1_action90")
        t3 = metrics.get(f"{stratum}/top3_action90")
        t5 = metrics.get(f"{stratum}/top5_action90")
        nll = metrics.get(f"{stratum}/nll_action90")
        ece = metrics.get(f"{stratum}/ece_action90")
        if t1 is not None:
            print(f"    action90  top1={t1:.3%}  top3={t3:.3%}  top5={t5:.3%}  NLL={nll:.4f}  ECE={ece:.4f}")

        # Lead-2
        t1l = metrics.get(f"{stratum}/top1_lead2")
        t3l = metrics.get(f"{stratum}/top3_lead2")
        nlll = metrics.get(f"{stratum}/nll_lead2")
        ecel = metrics.get(f"{stratum}/ece_lead2")
        if t1l is not None:
            print(f"    lead2     top1={t1l:.3%}  top3={t3l:.3%}  NLL={nlll:.4f}  ECE={ecel:.4f}")


def _save_metrics(metrics: dict, path: Path) -> None:
    """Save metrics dict as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {path}")


@click.command()
@click.option(
    "--split_dir",
    default="data/assembled/regime_a",
    type=click.Path(exists=True),
    help="Directory containing train.jsonl, val.jsonl, test.jsonl.",
)
@click.option(
    "--out_dir",
    default="outputs/baselines",
    type=click.Path(),
    help="Output directory for metrics and plots.",
)
@click.option(
    "--logistic_C",
    default=1.0,
    type=float,
    help="Regularization strength for logistic regression.",
)
def main(split_dir: str, out_dir: str, logistic_c: float) -> None:
    split_dir = Path(split_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading training data...")
    t0 = time.time()
    train_examples = list(read_jsonl(split_dir / "train.jsonl"))
    print(f"  {len(train_examples)} train examples loaded in {time.time()-t0:.1f}s")

    print("Loading test data...")
    t0 = time.time()
    test_examples = list(read_jsonl(split_dir / "test.jsonl"))
    print(f"  {len(test_examples)} test examples loaded in {time.time()-t0:.1f}s")

    # Extract arrays
    train_a90, train_l2, train_b4, train_mir, train_ca, train_cb = _extract_arrays(train_examples)
    test_a90, test_l2, test_b4, test_mir, test_ca, test_cb = _extract_arrays(test_examples)

    all_results: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # 1. Popularity baseline (global)
    # ------------------------------------------------------------------
    print("\n--- Popularity Baseline (global) ---")
    pop = PopularityBaseline().fit(train_a90)
    pop_probs = pop.predict(len(test_examples))

    pop_metrics = compute_metrics(pop_probs, test_a90, test_l2, test_b4, test_mir)
    _print_metrics("Popularity (global)", pop_metrics)
    _save_metrics(pop_metrics, out_dir / "metrics_popularity.json")
    all_results["Popularity"] = pop_metrics

    reliability_diagram(
        pop_probs[test_b4],
        test_a90[test_b4],
        out_dir / "reliability_popularity",
        title="Reliability — Popularity (global)",
    )
    print("  Reliability diagram saved.")

    # ------------------------------------------------------------------
    # 2. Popularity baseline (core-conditional)
    # ------------------------------------------------------------------
    print("\n--- Popularity Baseline (core-conditional) ---")
    pop.fit_conditional(train_a90, train_ca, train_cb)
    pop_cond_probs = pop.predict_conditional_batch(test_ca, test_cb)

    pop_cond_metrics = compute_metrics(pop_cond_probs, test_a90, test_l2, test_b4, test_mir)
    _print_metrics("Popularity (core-conditional)", pop_cond_metrics)
    _save_metrics(pop_cond_metrics, out_dir / "metrics_popularity_cond.json")
    all_results["Pop. (cond.)"] = pop_cond_metrics

    reliability_diagram(
        pop_cond_probs[test_b4],
        test_a90[test_b4],
        out_dir / "reliability_popularity_cond",
        title="Reliability — Popularity (core-conditional)",
    )
    print("  Reliability diagram saved.")

    # ------------------------------------------------------------------
    # 3. Logistic regression baseline
    # ------------------------------------------------------------------
    print("\n--- Logistic Regression Baseline ---")
    t0 = time.time()
    lr = LogisticBaseline(C=logistic_c).fit(train_examples, train_a90)
    fit_time = time.time() - t0
    print(f"  Fit in {fit_time:.1f}s")

    t0 = time.time()
    lr_probs = lr.predict_proba(test_examples)
    pred_time = time.time() - t0
    print(f"  Predicted {len(test_examples)} examples in {pred_time:.1f}s")

    lr_metrics = compute_metrics(lr_probs, test_a90, test_l2, test_b4, test_mir)
    _print_metrics("Logistic Regression", lr_metrics)
    _save_metrics(lr_metrics, out_dir / "metrics_logistic.json")
    all_results["Logistic"] = lr_metrics

    reliability_diagram(
        lr_probs[test_b4],
        test_a90[test_b4],
        out_dir / "reliability_logistic",
        title="Reliability — Logistic Regression",
    )
    print("  Reliability diagram saved.")

    # ------------------------------------------------------------------
    # Comparison plots
    # ------------------------------------------------------------------
    print("\n--- Comparison Plots ---")
    topk_comparison_bar(all_results, out_dir / "topk_comparison")
    print("  Top-k comparison bar chart saved.")

    stratified_table(
        all_results,
        out_dir / "stratified_table.json",
        out_dir / "stratified_table.tex",
    )
    print("  Stratified table saved.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\nAll results saved to {out_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()

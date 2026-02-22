#!/usr/bin/env python3
"""Evaluate ablation ensembles and produce comparison table.

For each of the 3 ablation groups (a=action90_all, b=multitask, c=tier1_only):
1. Load 5 checkpoints → ensemble_predict()
2. compute_metrics() on Regime A test set
3. Save per-ablation metrics to outputs/eval/ablation_{a,b,c}/
4. Produce comparison table (JSON + printed)

Optionally run cluster_bootstrap_ci() on the winning ablation (--bootstrap flag).

Usage:
    cd /home/walter/CS229/turnzero
    .venv/bin/python scripts/eval_ablations.py
    .venv/bin/python scripts/eval_ablations.py --bootstrap  # adds CIs for best
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turnzero.data.dataset import Vocab, VGCDataset
from turnzero.eval.metrics import compute_metrics
from turnzero.uq.ensemble import ensemble_predict, save_ensemble_predictions

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_A = ROOT / "data" / "assembled" / "regime_a"
RUNS_DIR = ROOT / "outputs" / "runs"
EVAL_DIR = ROOT / "outputs" / "eval"
VOCAB_PATH = RUNS_DIR / "ablation_a_001" / "vocab.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
NUM_WORKERS = 4

ABLATION_GROUPS = {
    "a": {"label": "action90_all", "members": [f"ablation_a_{i:03d}" for i in range(1, 6)]},
    "b": {"label": "multitask",    "members": [f"ablation_b_{i:03d}" for i in range(1, 6)]},
    "c": {"label": "tier1_only",   "members": [f"ablation_c_{i:03d}" for i in range(1, 6)]},
}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true",
                        help="Run cluster-aware bootstrap CIs on the winning ablation.")
    parser.add_argument("--bootstrap_n", type=int, default=1000,
                        help="Number of bootstrap iterations (default 1000).")
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 70)
    print("  Ablation Ensemble Evaluation")
    print("=" * 70)

    # --- Build test DataLoader ---
    # Use vocab from first ablation (all share the same data/vocab)
    # Fall back to ensemble_001 if ablation hasn't been trained yet
    vp = VOCAB_PATH
    if not vp.exists():
        vp = RUNS_DIR / "ensemble_001" / "vocab.json"
    print(f"\nVocab: {vp}")
    vocab = Vocab.load(vp)
    test_ds = VGCDataset(DATA_A / "test.jsonl", vocab)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Test set: {len(test_ds)} examples, {len(test_loader)} batches")
    print(f"Device: {DEVICE}")

    # --- Evaluate each ablation ---
    all_results: dict[str, dict] = {}

    for group_key, group_info in ABLATION_GROUPS.items():
        label = group_info["label"]
        members = group_info["members"]
        ckpt_paths = [RUNS_DIR / m / "best.pt" for m in members]

        # Verify checkpoints exist
        missing = [p for p in ckpt_paths if not p.exists()]
        if missing:
            print(f"\n[SKIP] Ablation {group_key} ({label}): "
                  f"{len(missing)} missing checkpoints")
            for p in missing:
                print(f"  Missing: {p}")
            continue

        print(f"\n{'─' * 60}")
        print(f"Ablation {group_key}: {label}")
        print(f"{'─' * 60}")

        preds = ensemble_predict(ckpt_paths, test_loader, DEVICE)

        metrics = compute_metrics(
            probs=preds["probs"],
            action90_true=preds["action90_true"],
            lead2_true=preds["lead2_true"],
            bring4_observed=preds["bring4_observed"],
            is_mirror=preds["is_mirror"],
        )

        # Add uncertainty summary
        metrics["mean_confidence"] = float(preds["confidence"].mean())
        metrics["mean_entropy"] = float(preds["entropy"].mean())
        metrics["mean_mi"] = float(preds["mi"].mean())

        all_results[group_key] = metrics

        # Save per-ablation results
        out_dir = EVAL_DIR / f"ablation_{group_key}"
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics_json = {k: float(v) if hasattr(v, "__float__") else v
                        for k, v in metrics.items()}
        with open(out_dir / "test_metrics.json", "w") as f:
            json.dump(metrics_json, f, indent=2)

        save_ensemble_predictions(preds, out_dir / "ensemble_predictions.npz")

        print(f"  Results saved to {out_dir}")

    if not all_results:
        print("\nNo ablation groups have trained checkpoints. Run train_ablations.sh first.")
        sys.exit(1)

    # --- Comparison table ---
    print(f"\n{'=' * 70}")
    print("  ABLATION COMPARISON")
    print(f"{'=' * 70}")

    metric_keys = [
        ("overall/top1_action90", "Top-1 Act90", True),
        ("overall/top3_action90", "Top-3 Act90", True),
        ("overall/top5_action90", "Top-5 Act90", True),
        ("overall/top1_lead2",    "Top-1 Lead2", True),
        ("overall/top3_lead2",    "Top-3 Lead2", True),
        ("overall/nll_action90",  "NLL Act90",   False),
        ("overall/nll_lead2",     "NLL Lead2",   False),
        ("overall/ece_action90",  "ECE Act90",   False),
        ("mean_confidence",       "Confidence",   False),
        ("mean_entropy",          "Entropy",      False),
    ]

    # Header
    group_labels = {k: ABLATION_GROUPS[k]["label"] for k in all_results}
    header = f"{'Metric':<18}"
    for gk in sorted(all_results.keys()):
        header += f"  {group_labels[gk]:>16}"
    print(f"\n{header}")
    print("─" * len(header))

    for key, display, is_pct in metric_keys:
        row = f"{display:<18}"
        for gk in sorted(all_results.keys()):
            val = all_results[gk].get(key, 0)
            if is_pct:
                row += f"  {val * 100:>15.1f}%"
            else:
                row += f"  {val:>16.4f}"
        print(row)

    # --- Identify winner ---
    best_key = max(all_results, key=lambda k: all_results[k].get("overall/top1_action90", 0))
    best_label = ABLATION_GROUPS[best_key]["label"]
    best_t1 = all_results[best_key].get("overall/top1_action90", 0) * 100
    print(f"\nBest by Top-1 Action90: ablation_{best_key} ({best_label}) = {best_t1:.1f}%")

    # Save comparison table
    comparison = {
        "results": {k: {mk: float(v) if hasattr(v, "__float__") else v
                        for mk, v in metrics.items()}
                    for k, metrics in all_results.items()},
        "best_group": best_key,
        "best_loss_mode": best_label,
    }
    comp_path = EVAL_DIR / "ablation_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to {comp_path}")

    # --- Optional bootstrap CIs on winner ---
    if args.bootstrap:
        from turnzero.data.io_utils import read_jsonl
        from turnzero.eval.bootstrap import cluster_bootstrap_ci

        print(f"\n{'=' * 70}")
        print(f"  Bootstrap CIs for ablation_{best_key} ({best_label})")
        print(f"{'=' * 70}")

        # Load cluster IDs from test examples
        test_examples = list(read_jsonl(DATA_A / "test.jsonl"))
        cluster_ids = np.array([ex["split_keys"]["core_cluster_a"] for ex in test_examples])

        # Load saved predictions for winner
        winner_preds = np.load(EVAL_DIR / f"ablation_{best_key}" / "ensemble_predictions.npz")

        cis = cluster_bootstrap_ci(
            probs=winner_preds["probs"],
            action90_true=winner_preds["action90_true"],
            lead2_true=winner_preds["lead2_true"],
            bring4_observed=winner_preds["bring4_observed"],
            is_mirror=winner_preds["is_mirror"],
            cluster_ids=cluster_ids,
            n_bootstrap=args.bootstrap_n,
        )

        ci_path = EVAL_DIR / f"ablation_{best_key}" / "bootstrap_cis.json"
        with open(ci_path, "w") as f:
            json.dump(cis, f, indent=2)
        print(f"Bootstrap CIs saved to {ci_path}")

        # Print key CIs
        for key_name in ["overall/top1_action90", "overall/top3_action90", "overall/nll_action90"]:
            if key_name in cis:
                ci = cis[key_name]
                print(f"  {key_name}: {ci.get('mean', 0):.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]")

    dt = time.time() - t0
    print(f"\nTotal time: {dt:.1f}s ({dt / 60:.1f} min)")


if __name__ == "__main__":
    main()
